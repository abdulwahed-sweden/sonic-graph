# 🦇 sonic-graph

**A blazing-fast, mathematically precise CLI tool for generating high-resolution bioacoustics spectrograms.**

`sonic-graph` transforms raw audio signals (WAV files) into stunning 4K visual spectrograms. Engineered in Rust, it leverages Short-Time Fourier Transforms (STFT) to map frequencies and amplitudes over time, rendering the output with a luxurious, dark, and mysterious aesthetic.

## ✨ Features

- **Extreme Performance:** Built with heavily optimized Rust, utilizing `rayon` for multi-threaded FFT processing.
- **Mathematical Precision:** Uses robust STFT equations with Hamming/Hanning windowing to prevent spectral leakage, ensuring accurate mapping of animal vocalizations and sound waves.
- **Luxurious Aesthetics:** Outputs 4K (3840x2160) images. Features a custom dark colormap—mapping low decibels to deep voids and high intensities to glowing gold and neon cyan.
- **Pragmatic Architecture:** Zero-dependency overhead where possible, memory-efficient data pipelines, and rock-solid error handling.

## 🚀 Installation

Ensure you have [Rust and Cargo](https://rustup.rs/) installed.

```bash
git clone [https://github.com/yourusername/sonic-graph.git](https://github.com/yourusername/sonic-graph.git)
cd sonic-graph
cargo build --release
```
