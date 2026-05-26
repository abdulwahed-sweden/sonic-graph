//! STFT pipeline: Hann window → FFT → log-magnitude (dB).
//!
//! Each STFT frame is independent, so we fan out frames across rayon workers.
//! Buffers are allocated once per worker via `for_each_init` and reused —
//! crucial because FFT scratch and the windowed input buffer are the only
//! per-frame allocations we cannot avoid.

use std::sync::Arc;

use anyhow::{anyhow, Result};
use rayon::prelude::*;
use rustfft::{num_complex::Complex, Fft, FftPlanner};

pub struct Spectrogram {
    /// Row-major: `data[frame * bins + bin]`, values in dBFS.
    pub data: Vec<f32>,
    pub frames: usize,
    pub bins: usize,
    pub hop_size: usize,
    pub fft_size: usize,
}

impl Spectrogram {
    /// Random-access read of a single cell. Hot loops in `render` index
    /// `self.data` directly to skip the per-access bounds check.
    #[allow(dead_code)]
    #[inline]
    pub fn cell(&self, frame: usize, bin: usize) -> f32 {
        debug_assert!(frame < self.frames && bin < self.bins);
        self.data[frame * self.bins + bin]
    }
}

pub fn compute_spectrogram(
    samples: &[f32],
    fft_size: usize,
    hop_size: usize,
) -> Result<Spectrogram> {
    if fft_size < 2 || !fft_size.is_power_of_two() {
        return Err(anyhow!(
            "fft_size must be a power of two ≥ 2 (got {})",
            fft_size
        ));
    }
    if hop_size == 0 {
        return Err(anyhow!("hop_size must be ≥ 1"));
    }
    if samples.len() < fft_size {
        return Err(anyhow!(
            "audio is shorter than one FFT window ({} samples < fft_size {})",
            samples.len(),
            fft_size
        ));
    }

    let window = hann_window(fft_size);
    // Normalize so that a unit-amplitude sinusoid reads ~0 dBFS at its peak bin.
    let window_sum: f32 = window.iter().sum();
    let mag_norm = 2.0 / window_sum.max(f32::EPSILON);

    let frames = (samples.len() - fft_size) / hop_size + 1;
    let bins = fft_size / 2 + 1;

    let mut planner = FftPlanner::<f32>::new();
    let fft: Arc<dyn Fft<f32>> = planner.plan_fft_forward(fft_size);
    let scratch_len = fft.get_inplace_scratch_len();

    let mut data = vec![0.0_f32; frames * bins];

    data.par_chunks_mut(bins)
        .enumerate()
        .for_each_init(
            || {
                (
                    vec![Complex::<f32>::new(0.0, 0.0); fft_size],
                    vec![Complex::<f32>::new(0.0, 0.0); scratch_len],
                )
            },
            |(buffer, scratch), (frame, out_row)| {
                let start = frame * hop_size;
                let chunk = &samples[start..start + fft_size];

                for (i, (&s, &w)) in chunk.iter().zip(window.iter()).enumerate() {
                    buffer[i] = Complex::new(s * w, 0.0);
                }

                fft.process_with_scratch(buffer, scratch);

                // Only the first `bins` outputs are unique for a real input.
                for (bin, c) in buffer.iter().take(bins).enumerate() {
                    let mag = c.norm() * mag_norm;
                    // 1e-10 floor → -200 dB, safe against log10(0).
                    out_row[bin] = 20.0 * mag.max(1e-10).log10();
                }
            },
        );

    Ok(Spectrogram {
        data,
        frames,
        bins,
        hop_size,
        fft_size,
    })
}

fn hann_window(n: usize) -> Vec<f32> {
    use std::f32::consts::PI;
    let denom = (n - 1).max(1) as f32;
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / denom).cos())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn hann_window_endpoints_are_zero() {
        let w = hann_window(64);
        assert_abs_diff_eq!(w[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(*w.last().unwrap(), 0.0, epsilon = 1e-6);
        // Symmetric around the centre.
        assert_abs_diff_eq!(w[32], 1.0, epsilon = 1e-3);
    }

    #[test]
    fn pure_sine_peaks_at_expected_bin() {
        let sample_rate: f32 = 44_100.0;
        let freq: f32 = 1_000.0;
        let fft_size: usize = 4096;
        let hop_size: usize = 1024;
        let n = sample_rate as usize; // one second

        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate).sin())
            .collect();

        let spec = compute_spectrogram(&samples, fft_size, hop_size).unwrap();
        let expected_bin = (freq * fft_size as f32 / sample_rate).round() as usize;

        // Find the loudest bin in a steady mid-signal frame.
        let frame = spec.frames / 2;
        let (max_bin, max_db) = (0..spec.bins)
            .map(|b| (b, spec.cell(frame, b)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        assert!(
            (max_bin as i32 - expected_bin as i32).abs() <= 1,
            "expected peak near bin {}, got bin {} ({:.1} dB)",
            expected_bin,
            max_bin,
            max_db
        );
        // A full-scale sine should peak close to 0 dBFS after window correction.
        assert!(
            max_db > -3.0,
            "peak {:.2} dB is too low for a full-scale sine",
            max_db
        );
    }

    #[test]
    fn rejects_non_power_of_two_fft() {
        let samples = vec![0.0_f32; 4096];
        assert!(compute_spectrogram(&samples, 1000, 256).is_err());
    }
}
