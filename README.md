# candle-diffusion

Text-to-image with [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
using [candle](https://github.com/huggingface/candle) 0.11.0 on the Metal GPU.
Adapted from candle's `z_image` example into a standalone crate.

Install the `candy` command (to `~/.cargo/bin`):

```bash
cargo install --path .
```

```bash
candy --prompt "A cute robot holding a candle" --width 1024 --height 1024 --seed 42
```

Or without installing: `cargo run --release -- --prompt "..."`.

- The first run downloads ~33 GB of weights to `~/.cache/huggingface`.
- Without `--seed`, a random seed is used and added to the filename (e.g. `z_image_output-1234567.png`);
  rerun with `--seed 1234567` to get the same image.
- Width/height must be divisible by 16. Default steps: 9. Output: `z_image_output.png` (`--output` to change).
- `--model-path <dir>` uses a local copy of the weights; `--cpu` forces CPU.
- Metal is on by default; build with `--no-default-features` for CPU-only.

## Batch mode (JSONL)

Generate many images with one model load:

```bash
candy --input prompts.jsonl --output-dir out
```

Each line is a JSON object; only `prompt` is required. See [`batch5.jsonl`](batch5.jsonl)
for a full example (`candy -i batch5.jsonl --output-dir batch5`):

```json
{"prompt": "A red fox in fresh snow", "seed": 3}
{"prompt": "A bowl of ramen, top-down", "output": "food/ramen.png"}
{"prompt": "A watercolor sailboat at dawn", "width": 768, "height": 512, "num_steps": 6}
```

- Fields: `prompt`, `negative_prompt`, `width`, `height`, `num_steps`, `guidance_scale`, `seed`, `output`.
  Omitted fields fall back to the CLI flags (e.g. `--width 512` sets the default for every line).
- Without `output`, images are named by line number (`0003.png`); without `seed`, the random seed
  is appended (`0003-2747574158.png`). Outputs are relative to `--output-dir` (default `.`).
- Every line is validated before the model loads; unknown fields, bad sizes, invalid JSON and
  duplicate outputs are all reported at once.
- Lines whose image already exists are skipped, so an interrupted batch can be re-run.
  Use `--overwrite` to regenerate them.
- If a line fails during generation, the batch continues and exits non-zero at the end.

## Tests

```bash
cargo test --release                        # unit + CLI tests, no model needed (<1s)
cargo test --release -- --ignored           # end-to-end generation with the real model
```

The ignored test generates tiny 256×256 images and checks that the same seed gives
byte-identical output; it needs the downloaded weights.
