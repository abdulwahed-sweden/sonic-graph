//! WAV loading. Decodes integer and float WAVs and downmixes to mono `f32`.

use std::path::Path;

use anyhow::{anyhow, Context, Result};
use hound::{SampleFormat, WavReader};

pub struct Audio {
    /// Mono samples normalized to roughly `[-1.0, 1.0]`.
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

impl Audio {
    pub fn duration_sec(&self) -> f64 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.samples.len() as f64 / self.sample_rate as f64
        }
    }
}

pub fn load_wav<P: AsRef<Path>>(path: P) -> Result<Audio> {
    let mut reader = WavReader::open(path.as_ref()).context("opening WAV file")?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    if channels == 0 {
        return Err(anyhow!("WAV file declares zero channels"));
    }

    let interleaved: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("reading 32-bit float samples")?,
        (SampleFormat::Int, bits) if (8..=32).contains(&bits) => {
            // `1 << (bits-1)` is the magnitude of the smallest representable
            // signed value. Dividing by it maps the full int range to [-1, 1].
            let scale = 1.0_f32 / (1i64 << (bits - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<std::result::Result<Vec<_>, _>>()
                .context("reading integer samples")?
        }
        (fmt, bits) => {
            return Err(anyhow!(
                "unsupported WAV sample format ({:?}, {} bits)",
                fmt,
                bits
            ))
        }
    };

    let samples = if channels == 1 {
        interleaved
    } else {
        downmix_to_mono(&interleaved, channels)
    };

    Ok(Audio {
        samples,
        sample_rate: spec.sample_rate,
    })
}

fn downmix_to_mono(interleaved: &[f32], channels: usize) -> Vec<f32> {
    let frames = interleaved.len() / channels;
    let inv = 1.0_f32 / channels as f32;
    let mut out = Vec::with_capacity(frames);
    for f in 0..frames {
        let start = f * channels;
        let mut acc = 0.0_f32;
        for c in 0..channels {
            acc += interleaved[start + c];
        }
        out.push(acc * inv);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downmix_averages_channels() {
        let stereo = [0.5_f32, -0.5, 1.0, 0.0, 0.2, 0.4];
        let mono = downmix_to_mono(&stereo, 2);
        assert_eq!(mono, vec![0.0, 0.5, 0.3]);
    }
}
