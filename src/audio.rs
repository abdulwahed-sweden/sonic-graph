//! WAV loading. Parses the header with `hound`, then reads the entire data
//! chunk in one bulk `read_exact` and decodes bytes → `f32` in a tight loop.
//!
//! `hound`'s `samples::<T>()` iterator yields a `Result` per sample; for
//! large files that per-sample dispatch dominates the load time. Reading the
//! data chunk as raw bytes and decoding inline drops that overhead.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use hound::{SampleFormat, WavReader, WavSpec};

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
    // 1 MiB read buffer keeps the decode loop fed from the kernel page cache
    // with minimal syscalls even on the largest WAVs.
    let file = File::open(path.as_ref()).context("opening WAV file")?;
    let buffered = BufReader::with_capacity(1 << 20, file);

    let wav = WavReader::new(buffered).context("parsing WAV header")?;
    let spec = wav.spec();
    let channels = spec.channels as usize;
    if channels == 0 {
        return Err(anyhow!("WAV file declares zero channels"));
    }

    let total_samples = wav.len() as usize;
    let bytes_per_sample = bytes_per_sample(spec)?;
    let total_bytes = total_samples
        .checked_mul(bytes_per_sample)
        .ok_or_else(|| anyhow!("declared sample count overflows usize"))?;

    // After `WavReader::new`, the inner reader is positioned at the first
    // byte of the data chunk's sample payload.
    let mut inner = wav.into_inner();
    let mut raw = vec![0u8; total_bytes];
    inner
        .read_exact(&mut raw)
        .context("reading sample data chunk")?;

    let interleaved = decode_samples(&raw, spec)?;

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

fn bytes_per_sample(spec: WavSpec) -> Result<usize> {
    match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Int, 8) => Ok(1),
        (SampleFormat::Int, 16) => Ok(2),
        (SampleFormat::Int, 24) => Ok(3),
        (SampleFormat::Int, 32) => Ok(4),
        (SampleFormat::Float, 32) => Ok(4),
        (fmt, bits) => Err(anyhow!(
            "unsupported WAV sample format ({:?}, {} bits)",
            fmt,
            bits
        )),
    }
}

/// Bulk decoder. One match per format, then a single tight loop.
fn decode_samples(raw: &[u8], spec: WavSpec) -> Result<Vec<f32>> {
    match (spec.sample_format, spec.bits_per_sample) {
        // 8-bit WAVs are unsigned, centered at 128.
        (SampleFormat::Int, 8) => {
            const SCALE: f32 = 1.0 / 128.0;
            Ok(raw.iter().map(|&b| (b as i32 - 128) as f32 * SCALE).collect())
        }
        (SampleFormat::Int, 16) => {
            const SCALE: f32 = 1.0 / 32_768.0;
            if raw.len() % 2 != 0 {
                return Err(anyhow!("16-bit WAV data is not 2-byte aligned"));
            }
            let mut out = Vec::with_capacity(raw.len() / 2);
            for c in raw.chunks_exact(2) {
                let v = i16::from_le_bytes([c[0], c[1]]) as f32;
                out.push(v * SCALE);
            }
            Ok(out)
        }
        (SampleFormat::Int, 24) => {
            const SCALE: f32 = 1.0 / 8_388_608.0; // 2^23
            if raw.len() % 3 != 0 {
                return Err(anyhow!("24-bit WAV data is not 3-byte aligned"));
            }
            let mut out = Vec::with_capacity(raw.len() / 3);
            for c in raw.chunks_exact(3) {
                // Pack low 3 bytes, then sign-extend via arithmetic shift.
                let raw = (c[0] as i32) | ((c[1] as i32) << 8) | ((c[2] as i32) << 16);
                let v = (raw << 8) >> 8; // shift up, then arithmetic shift down
                out.push(v as f32 * SCALE);
            }
            Ok(out)
        }
        (SampleFormat::Int, 32) => {
            const SCALE: f32 = 1.0 / 2_147_483_648.0; // 2^31
            if raw.len() % 4 != 0 {
                return Err(anyhow!("32-bit int WAV data is not 4-byte aligned"));
            }
            let mut out = Vec::with_capacity(raw.len() / 4);
            for c in raw.chunks_exact(4) {
                let v = i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32;
                out.push(v * SCALE);
            }
            Ok(out)
        }
        (SampleFormat::Float, 32) => {
            if raw.len() % 4 != 0 {
                return Err(anyhow!("32-bit float WAV data is not 4-byte aligned"));
            }
            let mut out = Vec::with_capacity(raw.len() / 4);
            for c in raw.chunks_exact(4) {
                out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
            Ok(out)
        }
        (fmt, bits) => Err(anyhow!(
            "unsupported WAV sample format ({:?}, {} bits)",
            fmt,
            bits
        )),
    }
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
    use std::io::Cursor;

    #[test]
    fn downmix_averages_channels() {
        let stereo = [0.5_f32, -0.5, 1.0, 0.0, 0.2, 0.4];
        let mono = downmix_to_mono(&stereo, 2);
        assert_eq!(mono, vec![0.0, 0.5, 0.3]);
    }

    #[test]
    fn decode_16bit_round_trips_known_samples() {
        let raw: Vec<u8> = [0_i16, 1, -1, 32_767, -32_768]
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        let spec = WavSpec {
            channels: 1,
            sample_rate: 44_100,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let decoded = decode_samples(&raw, spec).unwrap();
        assert_eq!(decoded.len(), 5);
        assert!((decoded[0] - 0.0).abs() < 1e-6);
        assert!((decoded[1] - (1.0 / 32_768.0)).abs() < 1e-9);
        assert!((decoded[2] - (-1.0 / 32_768.0)).abs() < 1e-9);
        assert!((decoded[3] - (32_767.0 / 32_768.0)).abs() < 1e-6);
        assert!((decoded[4] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn decode_24bit_sign_extends_correctly() {
        // 0x000000 → 0, 0x7FFFFF → +max, 0x800000 → -max, 0xFFFFFF → -1
        let raw: Vec<u8> = vec![
            0x00, 0x00, 0x00, // 0
            0xFF, 0xFF, 0x7F, // +8388607
            0x00, 0x00, 0x80, // -8388608
            0xFF, 0xFF, 0xFF, // -1
        ];
        let spec = WavSpec {
            channels: 1,
            sample_rate: 48_000,
            bits_per_sample: 24,
            sample_format: SampleFormat::Int,
        };
        let decoded = decode_samples(&raw, spec).unwrap();
        assert_eq!(decoded.len(), 4);
        assert!((decoded[0]).abs() < 1e-9);
        assert!((decoded[1] - (8_388_607.0 / 8_388_608.0)).abs() < 1e-6);
        assert!((decoded[2] - (-1.0)).abs() < 1e-6);
        assert!((decoded[3] - (-1.0 / 8_388_608.0)).abs() < 1e-9);
    }

    #[test]
    fn decode_8bit_centers_on_128() {
        let spec = WavSpec {
            channels: 1,
            sample_rate: 8_000,
            bits_per_sample: 8,
            sample_format: SampleFormat::Int,
        };
        let raw = vec![128_u8, 0, 255, 129];
        let decoded = decode_samples(&raw, spec).unwrap();
        assert_eq!(decoded.len(), 4);
        assert!(decoded[0].abs() < 1e-6); // 128 → 0
        assert!((decoded[1] - (-1.0)).abs() < 1e-6); // 0 → -1
        assert!((decoded[2] - (127.0 / 128.0)).abs() < 1e-6); // 255 → ~+1
        assert!((decoded[3] - (1.0 / 128.0)).abs() < 1e-6); // 129 → ~+0.0078
    }

    #[test]
    fn round_trip_via_hound_writer_matches_batch_decoder() {
        // Synthesize a small 24-bit stereo WAV in memory, then decode it
        // through our pipeline. Compares against the values we wrote.
        let spec = WavSpec {
            channels: 2,
            sample_rate: 48_000,
            bits_per_sample: 24,
            sample_format: SampleFormat::Int,
        };
        let written: Vec<i32> = (0..200).map(|i| (i - 100) * 1000).collect();
        let mut buf: Vec<u8> = Vec::new();
        {
            let mut w = hound::WavWriter::new(Cursor::new(&mut buf), spec).unwrap();
            for &s in &written {
                w.write_sample(s).unwrap();
            }
            w.finalize().unwrap();
        }

        let wav = WavReader::new(Cursor::new(&buf)).unwrap();
        let total_samples = wav.len() as usize;
        let bps = bytes_per_sample(spec).unwrap();
        let mut inner = wav.into_inner();
        let mut raw = vec![0u8; total_samples * bps];
        inner.read_exact(&mut raw).unwrap();
        let decoded = decode_samples(&raw, spec).unwrap();
        assert_eq!(decoded.len(), written.len());
        for (i, (&w, &d)) in written.iter().zip(decoded.iter()).enumerate() {
            let expected = w as f32 / 8_388_608.0;
            assert!(
                (expected - d).abs() < 1e-6,
                "sample {} mismatch: written {}, decoded {}",
                i,
                expected,
                d
            );
        }
    }
}
