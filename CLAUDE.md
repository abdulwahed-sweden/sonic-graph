# `sonic-graph` Development Guidelines

This file dictates the coding standards and architectural principles for the `sonic-graph` project.

## Architectural & Coding Standards

1. **Human-like & Pragmatic Code:** Write idiomatic, readable, and highly maintainable Rust. Avoid overly complex trait bounds or over-engineered abstractions unless they serve a clear, practical purpose.
2. **Performance First:** This tool deals with heavy mathematical transformations (STFT) over large datasets. Memory allocation must be minimized. Use zero-copy techniques where possible. Parallelize heavy compute workloads using `rayon`.
3. **Error Handling:** Use pragmatic, explicit error handling (e.g., `anyhow` for applications, or custom `thiserror` enums if building a library). Silently ignoring errors or panicking via unhandled `unwrap()`/`expect()` in production paths is forbidden.
4. **Modularity:** Keep files focused. Separate I/O logic (`hound`), mathematical processing (`rustfft`), and rendering logic (`plotters`) into distinct modules.
5. **Aesthetics:** Visual rendering logic must always default to the "dark, mysterious, and luxurious" theme (4K resolution, deep blacks, glowing gold/cyan highlights).

## Build Commands

- Run in dev: `cargo run -- <input.wav> <output.png>`
- Build highly optimized release: `cargo build --release`
- Run tests: `cargo test`
