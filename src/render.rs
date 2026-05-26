//! Rendering. Rasterizes the spectrogram into an RGB pixel buffer in parallel,
//! then composes axes, labels and a colorbar on top using `plotters`.

use std::path::Path;

use anyhow::{anyhow, Context, Result};
use plotters::element::BitMapElement;
use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, Pos, VPos};
use rayon::prelude::*;

use crate::math::Spectrogram;

const BACKGROUND: RGBColor = RGBColor(0x0B, 0x0C, 0x10);
const GOLD: RGBColor = RGBColor(0xD4, 0xAF, 0x37);
const SILVER: RGBColor = RGBColor(0xC5, 0xC6, 0xC7);
const FAINT_AXIS: RGBColor = RGBColor(0x3A, 0x33, 0x22);

pub struct RenderOptions<'a> {
    pub spectrogram: &'a Spectrogram,
    pub sample_rate: u32,
    pub output: &'a Path,
    pub width: u32,
    pub height: u32,
    pub title: &'a str,
    pub db_floor: f32,
    pub db_ceiling: f32,
    pub max_freq: Option<f32>,
}

pub fn render_spectrogram(opts: RenderOptions) -> Result<()> {
    let spec = opts.spectrogram;
    if spec.frames == 0 || spec.bins == 0 {
        return Err(anyhow!("spectrogram is empty; nothing to render"));
    }
    if opts.db_ceiling <= opts.db_floor {
        return Err(anyhow!(
            "db_ceiling ({}) must be greater than db_floor ({})",
            opts.db_ceiling,
            opts.db_floor
        ));
    }

    let nyquist = opts.sample_rate as f32 * 0.5;
    let display_max_freq = opts
        .max_freq
        .unwrap_or(nyquist)
        .min(nyquist)
        .max(1.0);

    let bins_f = spec.bins as f32 - 1.0;
    let visible_bin_count = (((display_max_freq / nyquist) * bins_f).ceil() as usize + 1)
        .min(spec.bins)
        .max(2);
    // Snap the visible frequency upper bound to the centre of the last visible bin.
    let visible_max_freq =
        (visible_bin_count - 1) as f32 / bins_f * nyquist;

    let duration_sec =
        ((spec.frames - 1) * spec.hop_size + spec.fft_size) as f64 / opts.sample_rate as f64;

    let root = BitMapBackend::new(opts.output, (opts.width, opts.height)).into_drawing_area();
    root.fill(&BACKGROUND).context("filling background")?;

    let caption_style = ("sans-serif", 72_i32)
        .into_font()
        .style(FontStyle::Bold)
        .color(&GOLD);
    let label_style = ("sans-serif", 32_i32).into_font().color(&SILVER);
    let axis_desc_style = ("sans-serif", 40_i32)
        .into_font()
        .style(FontStyle::Bold)
        .color(&GOLD);

    // Right side carries the colorbar; reserve room for it.
    const COLORBAR_AREA: u32 = 220;

    let mut chart = ChartBuilder::on(&root)
        .margin_top(60)
        .margin_bottom(60)
        .margin_left(60)
        .margin_right(COLORBAR_AREA)
        .caption(opts.title, caption_style)
        .x_label_area_size(130_i32)
        .y_label_area_size(200_i32)
        .build_cartesian_2d(0.0_f64..duration_sec, 0.0_f64..visible_max_freq as f64)
        .context("building chart")?;

    chart
        .configure_mesh()
        .disable_mesh()
        .axis_style(FAINT_AXIS.stroke_width(2))
        .x_desc("Time (s)")
        .y_desc("Frequency (Hz)")
        .label_style(label_style)
        .axis_desc_style(axis_desc_style)
        .x_label_formatter(&format_time)
        .y_label_formatter(&format_freq)
        .x_labels(10)
        .y_labels(8)
        .draw()
        .context("drawing axes")?;

    // Rasterize spectrogram → RGB buffer the size of the plot area, then blit.
    let plot_area = chart.plotting_area();
    let (plot_w, plot_h) = plot_area.dim_in_pixel();
    if plot_w == 0 || plot_h == 0 {
        return Err(anyhow!("plot area is zero-sized; output image is too small"));
    }

    let pixels = rasterize_spectrogram(
        spec,
        visible_bin_count,
        plot_w as usize,
        plot_h as usize,
        opts.db_floor,
        opts.db_ceiling,
    );

    let image = BitMapElement::with_owned_buffer(
        (0.0_f64, visible_max_freq as f64),
        (plot_w, plot_h),
        pixels,
    )
    .ok_or_else(|| anyhow!("internal buffer size mismatch for BitMapElement"))?;
    plot_area
        .draw(&image)
        .context("drawing spectrogram bitmap")?;

    draw_colorbar(
        &root,
        opts.width,
        opts.height,
        opts.db_floor,
        opts.db_ceiling,
    )
    .context("drawing colorbar")?;

    root.present().context("finalizing image")?;
    Ok(())
}

/// Build an RGB pixel buffer of the spectrogram, sized to `width x height`.
///
/// Row 0 is the top of the image (high frequency); column 0 is the start of
/// the audio (t = 0). We sample the (frame, bin) matrix with bilinear
/// interpolation in dB space — at 4K, this hides the underlying grid without
/// muddying details.
fn rasterize_spectrogram(
    spec: &Spectrogram,
    visible_bins: usize,
    width: usize,
    height: usize,
    db_floor: f32,
    db_ceiling: f32,
) -> Vec<u8> {
    let range = (db_ceiling - db_floor).max(1e-6);
    let inv_range = 1.0 / range;
    let frames_max = (spec.frames - 1) as f32;
    let bins_max = (visible_bins - 1) as f32;
    let inv_w = 1.0 / (width as f32 - 1.0).max(1.0);
    let inv_h = 1.0 / (height as f32 - 1.0).max(1.0);
    let bins = spec.bins;

    let mut buffer = vec![0u8; width * height * 3];

    buffer
        .par_chunks_mut(width * 3)
        .enumerate()
        .for_each(|(y, row)| {
            // y=0 is the top (highest freq), y=height-1 is the bottom (DC).
            let freq_t = 1.0 - y as f32 * inv_h;
            let bin_f = (freq_t * bins_max).clamp(0.0, bins_max);
            let b0 = bin_f.floor() as usize;
            let b1 = (b0 + 1).min(visible_bins - 1);
            let by = bin_f - b0 as f32;

            for x in 0..width {
                let time_t = x as f32 * inv_w;
                let frame_f = (time_t * frames_max).clamp(0.0, frames_max);
                let f0 = frame_f.floor() as usize;
                let f1 = (f0 + 1).min(spec.frames - 1);
                let fx = frame_f - f0 as f32;

                let v00 = spec.data[f0 * bins + b0];
                let v10 = spec.data[f1 * bins + b0];
                let v01 = spec.data[f0 * bins + b1];
                let v11 = spec.data[f1 * bins + b1];
                let v0 = v00 + (v10 - v00) * fx;
                let v1 = v01 + (v11 - v01) * fx;
                let v = v0 + (v1 - v0) * by;

                let t = ((v - db_floor) * inv_range).clamp(0.0, 1.0);
                let (r, g, b) = colormap(t);

                let o = x * 3;
                row[o] = r;
                row[o + 1] = g;
                row[o + 2] = b;
            }
        });

    buffer
}

/// Luxurious colormap: void black → midnight purple → crimson magenta →
/// molten amber → glowing gold → luminous cyan halo.
#[inline]
fn colormap(t: f32) -> (u8, u8, u8) {
    const STOPS: [(f32, [f32; 3]); 7] = [
        (0.00, [0.000, 0.000, 0.000]),
        (0.12, [0.078, 0.020, 0.157]),
        (0.30, [0.290, 0.055, 0.360]),
        (0.50, [0.760, 0.094, 0.310]),
        (0.70, [0.965, 0.490, 0.180]),
        (0.88, [0.965, 0.870, 0.420]),
        (1.00, [0.835, 1.000, 0.980]),
    ];

    let t = t.clamp(0.0, 1.0);

    for i in 0..STOPS.len() - 1 {
        let (t0, c0) = STOPS[i];
        let (t1, c1) = STOPS[i + 1];
        if t <= t1 {
            let k = ((t - t0) / (t1 - t0)).clamp(0.0, 1.0);
            let r = c0[0] + (c1[0] - c0[0]) * k;
            let g = c0[1] + (c1[1] - c0[1]) * k;
            let b = c0[2] + (c1[2] - c0[2]) * k;
            return ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8);
        }
    }
    let last = STOPS[STOPS.len() - 1].1;
    (
        (last[0] * 255.0) as u8,
        (last[1] * 255.0) as u8,
        (last[2] * 255.0) as u8,
    )
}

fn draw_colorbar(
    root: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
    canvas_w: u32,
    canvas_h: u32,
    db_floor: f32,
    db_ceiling: f32,
) -> Result<()> {
    let bar_w: i32 = 36;
    let bar_right_padding: i32 = 150;
    let bar_x: i32 = canvas_w as i32 - bar_right_padding - bar_w;
    let bar_top: i32 = 230;
    let bar_bottom: i32 = canvas_h as i32 - 220;
    let bar_h = bar_bottom - bar_top;
    if bar_h <= 0 {
        return Ok(());
    }

    for y in 0..bar_h {
        let t = 1.0 - y as f32 / (bar_h - 1).max(1) as f32;
        let (r, g, b) = colormap(t);
        let color = RGBColor(r, g, b);
        root.draw(&Rectangle::new(
            [(bar_x, bar_top + y), (bar_x + bar_w, bar_top + y + 1)],
            color.filled(),
        ))?;
    }

    root.draw(&Rectangle::new(
        [(bar_x, bar_top), (bar_x + bar_w, bar_bottom)],
        FAINT_AXIS.stroke_width(2),
    ))?;

    let tick_style = ("sans-serif", 26_i32).into_font().color(&SILVER);
    let title_style = ("sans-serif", 30_i32)
        .into_font()
        .style(FontStyle::Bold)
        .color(&GOLD);
    let n_ticks = 6;
    let label_anchor = Pos::new(HPos::Left, VPos::Center);

    for i in 0..=n_ticks {
        let t = i as f32 / n_ticks as f32;
        let db = db_floor + (db_ceiling - db_floor) * t;
        let y = bar_bottom - (bar_h as f32 * t).round() as i32;
        root.draw(&PathElement::new(
            vec![(bar_x + bar_w, y), (bar_x + bar_w + 12, y)],
            FAINT_AXIS.stroke_width(2),
        ))?;
        root.draw(&Text::new(
            format!("{:>+5.0} dB", db),
            (bar_x + bar_w + 22, y),
            tick_style.clone().pos(label_anchor),
        ))?;
    }

    root.draw(&Text::new(
        "Intensity".to_string(),
        (bar_x + bar_w / 2, bar_top - 36),
        title_style.pos(Pos::new(HPos::Center, VPos::Bottom)),
    ))?;

    Ok(())
}

fn format_freq(hz: &f64) -> String {
    let v = *hz;
    if v >= 1000.0 {
        format!("{:.1} kHz", v / 1000.0)
    } else {
        format!("{:.0} Hz", v)
    }
}

fn format_time(t: &f64) -> String {
    let v = *t;
    if v >= 60.0 {
        let m = (v / 60.0).floor();
        let s = v - m * 60.0;
        format!("{}:{:05.2}", m as i64, s)
    } else if v >= 10.0 {
        format!("{:.2} s", v)
    } else {
        format!("{:.3} s", v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colormap_endpoints_are_dark_and_bright() {
        let (r0, g0, b0) = colormap(0.0);
        assert_eq!((r0, g0, b0), (0, 0, 0));
        let (r1, g1, b1) = colormap(1.0);
        // Top stop is a bright cyan-white; all channels should be very high.
        assert!(r1 > 200 && g1 > 200 && b1 > 200);
    }

    #[test]
    fn colormap_is_monotonic_in_luminance() {
        let lum = |t: f32| {
            let (r, g, b) = colormap(t);
            0.2126 * r as f32 + 0.7152 * g as f32 + 0.0722 * b as f32
        };
        let mut prev = lum(0.0);
        for i in 1..=20 {
            let cur = lum(i as f32 / 20.0);
            // Allow tiny dips between stops, but the overall trend must be up.
            assert!(cur >= prev - 5.0, "luminance dropped at t={}", i);
            prev = cur;
        }
    }
}
