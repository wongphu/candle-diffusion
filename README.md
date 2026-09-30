# candle-diffusion

Text-to-image with [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
using [candle](https://github.com/huggingface/candle) 0.11.0 on the Metal GPU.
Adapted from candle's `z_image` example into a standalone crate.

```bash
cargo run --release -- --prompt "A cute robot holding a candle" --width 1024 --height 1024 --seed 42
```

- The first run downloads ~33 GB of weights to `~/.cache/huggingface`.
- Width/height must be divisible by 16. Default steps: 9. Output: `z_image_output.png` (`--output` to change).
- `--model-path <dir>` uses a local copy of the weights; `--cpu` forces CPU.
- Metal is on by default; build with `--no-default-features` for CPU-only.
