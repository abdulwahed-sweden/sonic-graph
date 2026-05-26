//! sonic-graph-web — local HTTP server that serves a single page for
//! uploading a WAV and viewing the rendered spectrogram in the browser.
//!
//! Synchronous (rouille) by design: rouille spawns one worker thread per
//! request and the rendering pipeline is already CPU-bound + parallelized
//! internally with rayon, so an async runtime would add no real benefit.

use std::io::Write;
use std::path::Path;
use std::str::FromStr;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use rouille::{post_input, router, Request, Response};
use sonic_graph::{audio, math, render};

const INDEX_HTML: &str = include_str!("web_index.html");
const MAX_UPLOAD_BYTES: u64 = 512 * 1024 * 1024; // 512 MiB

#[derive(Parser, Debug)]
#[command(name = "sonic-graph-web", version, about = "Web viewer for sonic-graph spectrograms")]
struct Cli {
    /// Bind address
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Port to listen on
    #[arg(long, default_value_t = 7878)]
    port: u16,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let addr = format!("{}:{}", cli.host, cli.port);
    eprintln!("sonic-graph-web :: listening on http://{}", addr);
    eprintln!("                  open it in a browser to render WAVs");
    rouille::start_server(addr.clone(), move |request| {
        let started = Instant::now();
        let response = handle(request);
        log_request(request, &response, started);
        response
    });
}

fn log_request(request: &Request, response: &Response, started: Instant) {
    eprintln!(
        "[{}] {} {} -> {} ({:?})",
        chrono_like_now(),
        request.method(),
        request.url(),
        response.status_code,
        started.elapsed()
    );
}

fn chrono_like_now() -> String {
    // Avoid pulling in chrono; rouille already brings time, but a tiny
    // wall-clock formatter from std is enough for request logging.
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let h = (secs / 3600) % 24;
    let m = (secs / 60) % 60;
    let s = secs % 60;
    format!("{:02}:{:02}:{:02}", h, m, s)
}

fn handle(request: &Request) -> Response {
    router!(request,
        (GET) (/) => { Response::html(INDEX_HTML) },
        (GET) (/health) => { Response::text("ok") },
        (POST) (/render) => { handle_render(request) },
        _ => Response::empty_404(),
    )
}

fn handle_render(request: &Request) -> Response {
    match try_render(request) {
        Ok(png) => Response::from_data("image/png", png),
        Err(e) => {
            // Walk the anyhow chain so the browser surfaces the root cause.
            let mut msg = format!("{}", e);
            for cause in e.chain().skip(1) {
                msg.push_str(&format!("\n  caused by: {}", cause));
            }
            Response::text(msg).with_status_code(400)
        }
    }
}

fn try_render(request: &Request) -> Result<Vec<u8>> {
    let data = post_input!(request, {
        wav: rouille::input::post::BufferedFile,
        colormap: String,
        title: String,
        fft_size: String,
        hop_size: String,
        db_floor: String,
        db_ceiling: String,
        max_freq: String,
        width: String,
        height: String,
    })
    .map_err(|e| anyhow!("invalid form upload: {}", e))?;

    if data.wav.data.is_empty() {
        return Err(anyhow!("WAV upload is empty"));
    }
    if data.wav.data.len() as u64 > MAX_UPLOAD_BYTES {
        return Err(anyhow!(
            "WAV upload too large ({} bytes; limit {} bytes)",
            data.wav.data.len(),
            MAX_UPLOAD_BYTES
        ));
    }

    let fft_size = parse_num(&data.fft_size, "fft_size", 2048_usize)?;
    let hop_size = parse_num(&data.hop_size, "hop_size", 512_usize)?;
    let db_floor = parse_num(&data.db_floor, "db_floor", -90.0_f32)?;
    let db_ceiling = parse_num(&data.db_ceiling, "db_ceiling", 0.0_f32)?;
    let width = parse_num(&data.width, "width", 3840_u32)?;
    let height = parse_num(&data.height, "height", 2160_u32)?;
    let max_freq = parse_optional_num::<f32>(&data.max_freq, "max_freq")?;
    let colormap = parse_colormap(&data.colormap)?;

    let original_name = data
        .wav
        .filename
        .as_deref()
        .unwrap_or("upload.wav")
        .to_string();
    let title = if data.title.trim().is_empty() {
        derive_title(&original_name)
    } else {
        data.title.trim().to_string()
    };

    // Hand the upload to the existing path-based loader via a tempfile.
    // Keeps audio.rs simple and incurs only a ~MB-scale disk write.
    let mut wav_temp = tempfile::Builder::new()
        .prefix("sonic-graph-")
        .suffix(".wav")
        .tempfile()
        .context("creating upload tempfile")?;
    wav_temp
        .as_file_mut()
        .write_all(&data.wav.data)
        .context("writing upload tempfile")?;
    wav_temp
        .as_file_mut()
        .flush()
        .context("flushing upload tempfile")?;

    let audio = audio::load_wav(wav_temp.path())
        .with_context(|| format!("decoding `{}`", original_name))?;
    let spectrogram = math::compute_spectrogram(&audio.samples, fft_size, hop_size)
        .context("STFT computation failed")?;

    let png_temp = tempfile::Builder::new()
        .prefix("sonic-graph-")
        .suffix(".png")
        .tempfile()
        .context("creating render tempfile")?;
    let png_path: &Path = png_temp.path();

    render::render_spectrogram(render::RenderOptions {
        spectrogram: &spectrogram,
        sample_rate: audio.sample_rate,
        output: png_path,
        width,
        height,
        title: &title,
        db_floor,
        db_ceiling,
        max_freq,
        colormap,
    })
    .context("render failed")?;

    let bytes = std::fs::read(png_path).context("reading rendered PNG")?;
    Ok(bytes)
}

fn parse_num<T: FromStr>(s: &str, name: &str, default: T) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    let trimmed = s.trim();
    if trimmed.is_empty() {
        Ok(default)
    } else {
        trimmed
            .parse::<T>()
            .map_err(|e| anyhow!("`{}` is not a valid number: {}", name, e))
    }
}

fn parse_optional_num<T: FromStr>(s: &str, name: &str) -> Result<Option<T>>
where
    T::Err: std::fmt::Display,
{
    let trimmed = s.trim();
    if trimmed.is_empty() {
        Ok(None)
    } else {
        trimmed
            .parse::<T>()
            .map(Some)
            .map_err(|e| anyhow!("`{}` is not a valid number: {}", name, e))
    }
}

fn parse_colormap(s: &str) -> Result<render::Colormap> {
    match s.trim().to_ascii_lowercase().as_str() {
        "luxe" => Ok(render::Colormap::Luxe),
        "magma" => Ok(render::Colormap::Magma),
        "inferno" => Ok(render::Colormap::Inferno),
        "viridis" => Ok(render::Colormap::Viridis),
        "mono" => Ok(render::Colormap::Mono),
        other => Err(anyhow!("unknown colormap `{}`", other)),
    }
}

fn derive_title(filename: &str) -> String {
    Path::new(filename)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Spectrogram")
        .to_string()
}
