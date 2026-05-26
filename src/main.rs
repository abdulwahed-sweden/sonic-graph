//! sonic-graph — generate luxurious 4K spectrograms from WAV audio.

mod audio;
mod math;
mod render;

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = "sonic-graph",
    version,
    about = "Generate luxurious 4K spectrograms from WAV audio",
    long_about = "Reads a WAV file, computes a windowed Short-Time Fourier Transform, \
                  and renders a dark, mysterious, high-resolution spectrogram image."
)]
struct Cli {
    /// Input WAV file
    input: PathBuf,

    /// Output PNG file
    output: PathBuf,

    /// FFT window size (must be a power of two)
    #[arg(short = 'n', long, default_value_t = 2048)]
    fft_size: usize,

    /// Hop size in samples between successive frames
    #[arg(short = 'p', long, default_value_t = 512)]
    hop_size: usize,

    /// Lower dB clip used for color mapping
    #[arg(long, default_value_t = -90.0, allow_hyphen_values = true)]
    db_floor: f32,

    /// Upper dB clip used for color mapping
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    db_ceiling: f32,

    /// Maximum frequency to display (Hz). Defaults to the Nyquist frequency.
    #[arg(long)]
    max_freq: Option<f32>,

    /// Output image width in pixels
    #[arg(long, default_value_t = 3840)]
    width: u32,

    /// Output image height in pixels
    #[arg(long, default_value_t = 2160)]
    height: u32,

    /// Override the displayed title (defaults to the input file's stem)
    #[arg(long)]
    title: Option<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let started = Instant::now();

    eprintln!("sonic-graph :: loading {}", cli.input.display());
    let t0 = Instant::now();
    let audio = audio::load_wav(&cli.input)
        .with_context(|| format!("failed to load `{}`", cli.input.display()))?;
    eprintln!(
        "             {} samples, {} Hz, {:.2}s ({:?})",
        audio.samples.len(),
        audio.sample_rate,
        audio.duration_sec(),
        t0.elapsed()
    );

    eprintln!(
        "sonic-graph :: STFT  (fft={}, hop={})",
        cli.fft_size, cli.hop_size
    );
    let t1 = Instant::now();
    let spectrogram = math::compute_spectrogram(&audio.samples, cli.fft_size, cli.hop_size)
        .context("STFT computation failed")?;
    eprintln!(
        "             {} frames x {} bins ({:?})",
        spectrogram.frames,
        spectrogram.bins,
        t1.elapsed()
    );

    let title = cli.title.clone().unwrap_or_else(|| {
        cli.input
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("Spectrogram")
            .to_string()
    });

    eprintln!(
        "sonic-graph :: render {}x{} -> {}",
        cli.width,
        cli.height,
        cli.output.display()
    );
    let t2 = Instant::now();
    render::render_spectrogram(render::RenderOptions {
        spectrogram: &spectrogram,
        sample_rate: audio.sample_rate,
        output: &cli.output,
        width: cli.width,
        height: cli.height,
        title: &title,
        db_floor: cli.db_floor,
        db_ceiling: cli.db_ceiling,
        max_freq: cli.max_freq,
    })
    .with_context(|| format!("failed to render `{}`", cli.output.display()))?;
    eprintln!("             render done ({:?})", t2.elapsed());

    eprintln!("sonic-graph :: total {:?}", started.elapsed());
    Ok(())
}
