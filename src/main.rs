//! Z-Image Text-to-Image Generation with candle on Metal.
//!
//! Adapted from candle 0.11.0's `z_image` example as a standalone crate.
//!
//! ```bash
//! cargo run --release -- --prompt "A beautiful landscape with mountains" \
//!     --height 1024 --width 1024
//! ```
//!
//! # Model Files
//!
//! Models are automatically downloaded from HuggingFace, or you can download manually:
//! <https://huggingface.co/Tongyi-MAI/Z-Image-Turbo>

use anyhow::{Context, Error as E, Result};
use candle::{DType, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::z_image::{
    calculate_shift, postprocess_image, AutoEncoderKL, Config, FlowMatchEulerDiscreteScheduler,
    SchedulerConfig, TextEncoderConfig, VaeConfig, ZImageTextEncoder, ZImageTransformer2DModel,
};
use clap::Parser;
use hf_hub::api::sync::Api;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

/// Z-Image scheduler constants
const BASE_IMAGE_SEQ_LEN: usize = 256;
const MAX_IMAGE_SEQ_LEN: usize = 4096;
const BASE_SHIFT: f64 = 0.5;
const MAX_SHIFT: f64 = 1.15;

#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
enum Model {
    /// Z-Image-Turbo: optimized for fast inference (8-9 steps)
    Turbo,
}

impl Model {
    fn repo(&self) -> &'static str {
        match self {
            Self::Turbo => "Tongyi-MAI/Z-Image-Turbo",
        }
    }

    fn default_steps(&self) -> usize {
        match self {
            Self::Turbo => 9,
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The prompt to be used for image generation.
    #[arg(
        long,
        default_value = "A beautiful landscape with mountains and a lake",
        conflicts_with = "input"
    )]
    prompt: String,

    /// The negative prompt (for CFG).
    #[arg(long, default_value = "")]
    negative_prompt: String,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// The height in pixels of the generated image.
    #[arg(long, default_value_t = 1024)]
    height: usize,

    /// The width in pixels of the generated image.
    #[arg(long, default_value_t = 1024)]
    width: usize,

    /// Number of inference steps.
    #[arg(long)]
    num_steps: Option<usize>,

    /// Guidance scale for CFG.
    #[arg(long, default_value_t = 5.0)]
    guidance_scale: f64,

    /// The seed to use when generating random samples. If omitted, a random
    /// seed is used and appended to the output filename, e.g. out-1234.png.
    #[arg(long)]
    seed: Option<u64>,

    /// Which model variant to use.
    #[arg(long, value_enum, default_value = "turbo")]
    model: Model,

    /// Override path to the model weights directory (uses HuggingFace by default).
    #[arg(long)]
    model_path: Option<String>,

    /// Output image filename.
    #[arg(long, default_value = "z_image_output.png", conflicts_with = "input")]
    output: String,

    /// JSONL file with one image per line, e.g.
    /// {"prompt": "a cat", "seed": 1, "width": 768, "output": "cat.png"}.
    /// Fields: prompt (required), negative_prompt, width, height, num_steps,
    /// guidance_scale, seed, output. Omitted fields use the CLI values;
    /// omitted output defaults to the line number, e.g. 0003.png.
    #[arg(long, short)]
    input: Option<PathBuf>,

    /// Directory for images generated from --input; relative outputs are placed here.
    #[arg(long, default_value = ".", requires = "input")]
    output_dir: PathBuf,

    /// With --input, regenerate images whose output file already exists.
    #[arg(long, requires = "input")]
    overwrite: bool,
}

/// Format user prompt for Qwen3 chat template
/// Corresponds to add_generation_prompt=True, enable_thinking=True
///
/// Format:
/// <|im_start|>user
/// {prompt}<|im_end|>
/// <|im_start|>assistant
fn format_prompt_for_qwen3(prompt: &str) -> String {
    format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        prompt
    )
}

/// One image to generate, with all defaults resolved.
#[derive(Debug)]
struct Job {
    /// 1-based line number in the input file (0 for a single CLI job).
    line: usize,
    prompt: String,
    negative_prompt: String,
    width: usize,
    height: usize,
    num_steps: usize,
    guidance_scale: f64,
    seed: u64,
    /// True when the seed was picked at random; it is then part of `output`.
    random_seed: bool,
    /// The output path before any random seed is appended.
    base_output: PathBuf,
    output: PathBuf,
}

/// One line of the JSONL input. Missing fields fall back to the CLI values.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct JobSpec {
    prompt: String,
    negative_prompt: Option<String>,
    width: Option<usize>,
    height: Option<usize>,
    num_steps: Option<usize>,
    guidance_scale: Option<f64>,
    seed: Option<u64>,
    output: Option<String>,
}

impl Job {
    fn validate(&self) -> Result<()> {
        let vae_align = 16; // vae_scale_factor * 2 = 8 * 2 = 16
        anyhow::ensure!(!self.prompt.trim().is_empty(), "prompt is empty");
        anyhow::ensure!(self.num_steps > 0, "num_steps must be at least 1");
        if !self.height.is_multiple_of(vae_align) || !self.width.is_multiple_of(vae_align) {
            anyhow::bail!(
                "Image dimensions must be divisible by {}. Got {}x{}. \
                 Try {}x{} or {}x{} instead.",
                vae_align,
                self.width,
                self.height,
                (self.width / vae_align) * vae_align,
                (self.height / vae_align) * vae_align,
                ((self.width / vae_align) + 1) * vae_align,
                ((self.height / vae_align) + 1) * vae_align
            );
        }
        Ok(())
    }
}

impl Job {
    /// Fills in the seed, picking a random one and appending it to the
    /// output filename when none was given.
    fn with_seed(mut self, seed: Option<u64>) -> Self {
        match seed {
            Some(seed) => self.seed = seed,
            None => {
                self.seed = random_seed();
                self.random_seed = true;
                self.output = seeded_path(&self.base_output, self.seed);
            }
        }
        self
    }

    /// An existing image for this job, if any. For random seeds this matches
    /// any `<stem>-<seed>.<ext>` file, since the seed differs on every run.
    fn existing_output(&self) -> Option<PathBuf> {
        if !self.random_seed {
            return self.output.exists().then(|| self.output.clone());
        }
        let dir = match self.base_output.parent() {
            Some(p) if !p.as_os_str().is_empty() => p,
            _ => Path::new("."),
        };
        let stem = self.base_output.file_stem()?.to_str()?;
        let ext = self.base_output.extension().and_then(|e| e.to_str());
        std::fs::read_dir(dir)
            .ok()?
            .flatten()
            .map(|e| e.path())
            .find(|p| {
                let seed_part = p
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .and_then(|s| s.strip_prefix(stem))
                    .and_then(|s| s.strip_prefix('-'));
                let same_ext = p.extension().and_then(|e| e.to_str()) == ext;
                same_ext
                    && seed_part
                        .is_some_and(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()))
            })
    }
}

/// Standard normal noise from a seeded CPU RNG, so a seed always gives the same
/// image. (Seeding candle's Metal RNG via `Device::set_seed` is not reproducible.)
fn seeded_noise(
    seed: u64,
    shape: (usize, usize, usize, usize),
    device: &candle::Device,
) -> Result<Tensor> {
    use rand::SeedableRng;
    use rand_distr::Distribution;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let n = shape.0 * shape.1 * shape.2 * shape.3;
    let values: Vec<f32> = rand_distr::StandardNormal
        .sample_iter(&mut rng)
        .take(n)
        .collect();
    Ok(Tensor::from_vec(values, shape, device)?)
}

fn random_seed() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    // RandomState is seeded from OS randomness; keep seeds short for filenames.
    let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
    hasher.write_u128(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default(),
    );
    hasher.finish() % u32::MAX as u64
}

/// `dir/name.png` -> `dir/name-<seed>.png`.
fn seeded_path(path: &Path, seed: u64) -> PathBuf {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
    let name = match path.extension().and_then(|e| e.to_str()) {
        Some(ext) => format!("{stem}-{seed}.{ext}"),
        None => format!("{stem}-{seed}"),
    };
    path.with_file_name(name)
}

fn single_job(args: &Args) -> Job {
    Job {
        line: 0,
        prompt: args.prompt.clone(),
        negative_prompt: args.negative_prompt.clone(),
        width: args.width,
        height: args.height,
        num_steps: args.num_steps.unwrap_or_else(|| args.model.default_steps()),
        guidance_scale: args.guidance_scale,
        seed: 0,
        random_seed: false,
        base_output: PathBuf::from(&args.output),
        output: PathBuf::from(&args.output),
    }
    .with_seed(args.seed)
}

/// Parses and validates every line of a JSONL file, reporting all problems at once.
fn read_jobs(path: &Path, args: &Args) -> Result<Vec<Job>> {
    let content =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut jobs: Vec<Job> = Vec::new();
    let mut errors = Vec::new();
    let mut seen_outputs = std::collections::HashMap::new();

    for (idx, line) in content.lines().enumerate() {
        let line_no = idx + 1;
        if line.trim().is_empty() {
            continue;
        }
        let spec: JobSpec = match serde_json::from_str(line) {
            Ok(spec) => spec,
            Err(e) => {
                errors.push(format!("line {line_no}: {e}"));
                continue;
            }
        };
        let output = spec.output.unwrap_or_else(|| format!("{line_no:04}.png"));
        let seed = spec.seed.or(args.seed);
        let job = Job {
            line: line_no,
            prompt: spec.prompt,
            negative_prompt: spec
                .negative_prompt
                .unwrap_or_else(|| args.negative_prompt.clone()),
            width: spec.width.unwrap_or(args.width),
            height: spec.height.unwrap_or(args.height),
            num_steps: spec
                .num_steps
                .or(args.num_steps)
                .unwrap_or_else(|| args.model.default_steps()),
            guidance_scale: spec.guidance_scale.unwrap_or(args.guidance_scale),
            seed: 0,
            random_seed: false,
            base_output: args.output_dir.join(&output),
            output: args.output_dir.join(output),
        }
        .with_seed(seed);
        if let Err(e) = job.validate() {
            errors.push(format!("line {line_no}: {e}"));
            continue;
        }
        if let Some(prev) = seen_outputs.insert(job.base_output.clone(), line_no) {
            errors.push(format!(
                "line {line_no}: output {} is also used by line {prev}",
                job.base_output.display()
            ));
            continue;
        }
        jobs.push(job);
    }

    if !errors.is_empty() {
        anyhow::bail!(
            "{} invalid line(s) in {}:\n  {}",
            errors.len(),
            path.display(),
            errors.join("\n  ")
        );
    }
    anyhow::ensure!(!jobs.is_empty(), "no prompts found in {}", path.display());
    Ok(jobs)
}

/// The loaded models, reused across every image in a run.
struct Pipeline {
    device: candle::Device,
    dtype: DType,
    tokenizer: Tokenizer,
    text_encoder: ZImageTextEncoder,
    transformer: ZImageTransformer2DModel,
    transformer_cfg: Config,
    vae: AutoEncoderKL,
}

impl Pipeline {
    fn load(args: &Args) -> Result<Self> {
        let device = device(args.cpu)?;
        let dtype = device.bf16_default_to_f32();

        // Resolve model: use provided path or download from HuggingFace
        let api = Api::new()?;
        let repo = api.model(args.model.repo().to_string());
        let use_local = args.model_path.is_some();
        let model_path = args.model_path.clone().map(std::path::PathBuf::from);

        if use_local {
            println!(
                "\nLoading models from local path: {}",
                model_path.as_ref().unwrap().display()
            );
        } else {
            println!(
                "\nDownloading model from HuggingFace: {}",
                args.model.repo()
            );
        }

        // ==================== Load Tokenizer ====================
        println!("Loading tokenizer...");
        let tokenizer_path = if use_local {
            model_path
                .as_ref()
                .unwrap()
                .join("tokenizer")
                .join("tokenizer.json")
        } else {
            repo.get("tokenizer/tokenizer.json")?
        };
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(E::msg)?;

        // ==================== Load Text Encoder ====================
        println!("Loading text encoder...");
        let text_encoder_config_path = if use_local {
            model_path
                .as_ref()
                .unwrap()
                .join("text_encoder")
                .join("config.json")
        } else {
            repo.get("text_encoder/config.json")?
        };
        let text_encoder_cfg: TextEncoderConfig = if text_encoder_config_path.exists() {
            serde_json::from_reader(std::fs::File::open(&text_encoder_config_path)?)?
        } else {
            TextEncoderConfig::z_image()
        };

        let text_encoder_weights = {
            let files: Vec<std::path::PathBuf> = if use_local {
                (1..=3)
                    .map(|i| {
                        model_path
                            .as_ref()
                            .unwrap()
                            .join("text_encoder")
                            .join(format!("model-{:05}-of-00003.safetensors", i))
                    })
                    .filter(|p| p.exists())
                    .collect()
            } else {
                (1..=3)
                    .map(|i| repo.get(&format!("text_encoder/model-{:05}-of-00003.safetensors", i)))
                    .filter_map(|r| r.ok())
                    .collect()
            };

            if files.is_empty() {
                anyhow::bail!("Text encoder weights not found");
            }

            let files: Vec<&str> = files.iter().map(|p| p.to_str().unwrap()).collect();
            unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, &device)? }
        };

        let text_encoder = ZImageTextEncoder::new(&text_encoder_cfg, text_encoder_weights)?;

        // ==================== Load Transformer ====================
        println!("Loading transformer...");
        let transformer_config_path = if use_local {
            model_path
                .as_ref()
                .unwrap()
                .join("transformer")
                .join("config.json")
        } else {
            repo.get("transformer/config.json")?
        };
        let transformer_cfg: Config = if transformer_config_path.exists() {
            serde_json::from_reader(std::fs::File::open(&transformer_config_path)?)?
        } else {
            Config::z_image_turbo()
        };

        let transformer_weights = {
            let files: Vec<std::path::PathBuf> = if use_local {
                (1..=3)
                    .map(|i| {
                        model_path
                            .as_ref()
                            .unwrap()
                            .join("transformer")
                            .join(format!(
                                "diffusion_pytorch_model-{:05}-of-00003.safetensors",
                                i
                            ))
                    })
                    .filter(|p| p.exists())
                    .collect()
            } else {
                (1..=3)
                    .map(|i| {
                        repo.get(&format!(
                            "transformer/diffusion_pytorch_model-{:05}-of-00003.safetensors",
                            i
                        ))
                    })
                    .filter_map(|r| r.ok())
                    .collect()
            };

            if files.is_empty() {
                anyhow::bail!("Transformer weights not found");
            }

            let files: Vec<&str> = files.iter().map(|p| p.to_str().unwrap()).collect();
            unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, &device)? }
        };

        let transformer = ZImageTransformer2DModel::new(&transformer_cfg, transformer_weights)?;

        // ==================== Load VAE ====================
        println!("Loading VAE...");
        let vae_config_path = if use_local {
            model_path.as_ref().unwrap().join("vae").join("config.json")
        } else {
            repo.get("vae/config.json")?
        };
        let vae_cfg: VaeConfig = if vae_config_path.exists() {
            serde_json::from_reader(std::fs::File::open(&vae_config_path)?)?
        } else {
            VaeConfig::z_image()
        };

        let vae_path = if use_local {
            let path = model_path
                .as_ref()
                .unwrap()
                .join("vae")
                .join("diffusion_pytorch_model.safetensors");
            if !path.exists() {
                anyhow::bail!("VAE weights not found at {:?}", path);
            }
            path
        } else {
            repo.get("vae/diffusion_pytorch_model.safetensors")?
        };

        let vae_weights = unsafe {
            VarBuilder::from_mmaped_safetensors(&[vae_path.to_str().unwrap()], dtype, &device)?
        };
        let vae = AutoEncoderKL::new(&vae_cfg, vae_weights)?;

        Ok(Self {
            device,
            dtype,
            tokenizer,
            text_encoder,
            transformer,
            transformer_cfg,
            vae,
        })
    }

    fn generate(&self, job: &Job) -> Result<()> {
        let Self {
            device,
            dtype,
            tokenizer,
            text_encoder,
            transformer,
            transformer_cfg,
            vae,
        } = self;
        let dtype = *dtype;

        // ==================== Initialize Scheduler ====================
        let scheduler_cfg = SchedulerConfig::z_image_turbo();
        let mut scheduler = FlowMatchEulerDiscreteScheduler::new(scheduler_cfg);

        // ==================== Prepare Inputs ====================
        println!("\nTokenizing prompt...");
        let formatted_prompt = format_prompt_for_qwen3(&job.prompt);
        let tokens = tokenizer
            .encode(formatted_prompt.as_str(), true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        println!("Token count: {}", tokens.len());

        // Create input tensor
        let input_ids = Tensor::from_vec(tokens.clone(), (1, tokens.len()), &device)?;

        // Get text embeddings (from second-to-last layer)
        println!("Encoding text...");
        let cap_feats = text_encoder.forward(&input_ids)?;
        let cap_mask = Tensor::ones((1, tokens.len()), DType::U8, &device)?;

        // Process negative prompt for CFG
        let (neg_cap_feats, neg_cap_mask) =
            if !job.negative_prompt.is_empty() && job.guidance_scale > 1.0 {
                let formatted_neg = format_prompt_for_qwen3(&job.negative_prompt);
                let neg_tokens = tokenizer
                    .encode(formatted_neg.as_str(), true)
                    .map_err(E::msg)?
                    .get_ids()
                    .to_vec();
                let neg_input_ids =
                    Tensor::from_vec(neg_tokens.clone(), (1, neg_tokens.len()), &device)?;
                let neg_feats = text_encoder.forward(&neg_input_ids)?;
                let neg_mask = Tensor::ones((1, neg_tokens.len()), DType::U8, &device)?;
                (Some(neg_feats), Some(neg_mask))
            } else {
                (None, None)
            };

        // ==================== Calculate Latent Dimensions ====================
        // Formula from Python pipeline: latent = 2 * (image_size // 16)
        // This ensures: latent is divisible by patch_size=2, and VAE decode (8x) gives correct size
        let patch_size = transformer_cfg.all_patch_size[0];
        let vae_align = 16; // vae_scale_factor * 2 = 8 * 2 = 16

        // Correct latent size formula: 2 * (image_size // 16)
        let latent_h = 2 * (job.height / vae_align);
        let latent_w = 2 * (job.width / vae_align);
        println!("Latent size: {}x{}", latent_w, latent_h);

        // Calculate image sequence length for shift
        let image_seq_len = (latent_h / patch_size) * (latent_w / patch_size);
        let mu = calculate_shift(
            image_seq_len,
            BASE_IMAGE_SEQ_LEN,
            MAX_IMAGE_SEQ_LEN,
            BASE_SHIFT,
            MAX_SHIFT,
        );
        println!("Image sequence length: {}, mu: {:.4}", image_seq_len, mu);

        // Set timesteps
        scheduler.set_timesteps(job.num_steps, Some(mu));

        // ==================== Generate Initial Noise ====================
        println!("\nGenerating initial noise...");
        let mut latents =
            seeded_noise(job.seed, (1, 16, latent_h, latent_w), device)?.to_dtype(dtype)?;

        // Add frame dimension: (B, C, H, W) -> (B, C, 1, H, W)
        latents = latents.unsqueeze(2)?;

        // ==================== Denoising Loop ====================
        println!("\nStarting denoising loop ({} steps)...", job.num_steps);

        for step in 0..job.num_steps {
            let t = scheduler.current_timestep_normalized();
            let t_tensor = Tensor::from_vec(vec![t as f32], (1,), &device)?.to_dtype(dtype)?;

            // Model prediction
            let noise_pred = transformer.forward(&latents, &t_tensor, &cap_feats, &cap_mask)?;

            // Apply CFG if guidance_scale > 1.0
            let noise_pred = if job.guidance_scale > 1.0 {
                if let (Some(ref neg_feats), Some(ref neg_mask)) = (&neg_cap_feats, &neg_cap_mask) {
                    let neg_pred = transformer.forward(&latents, &t_tensor, neg_feats, neg_mask)?;
                    // CFG: pred = neg + scale * (pos - neg)
                    let diff = (&noise_pred - &neg_pred)?;
                    (&neg_pred + (diff * job.guidance_scale)?)?
                } else {
                    // No negative prompt, use unconditional with zeros
                    noise_pred
                }
            } else {
                noise_pred
            };

            // Negate the prediction (Z-Image specific)
            let noise_pred = noise_pred.neg()?;

            // Remove frame dimension for scheduler: (B, C, 1, H, W) -> (B, C, H, W)
            let noise_pred_4d = noise_pred.squeeze(2)?;
            let latents_4d = latents.squeeze(2)?;

            // Scheduler step
            let prev_latents = scheduler.step(&noise_pred_4d, &latents_4d)?;

            // Add back frame dimension
            latents = prev_latents.unsqueeze(2)?;

            println!(
                "Step {}/{}: t = {:.4}, sigma = {:.4}",
                step + 1,
                job.num_steps,
                t,
                scheduler.current_sigma()
            );
        }

        // ==================== VAE Decode ====================
        println!("\nDecoding latents with VAE...");
        // Remove frame dimension: (B, C, 1, H, W) -> (B, C, H, W)
        let latents = latents.squeeze(2)?;
        let image = vae.decode(&latents)?;

        // Post-process: [-1, 1] -> [0, 255]
        let image = postprocess_image(&image)?;

        // ==================== Save Image ====================
        println!("Saving image to {}...", job.output.display());
        let image = image.i(0)?; // Remove batch dimension
        save_image(&image, &job.output)?;
        Ok(())
    }
}

fn run(args: Args) -> Result<()> {
    let jobs = match &args.input {
        Some(path) => read_jobs(path, &args)?,
        None => {
            let job = single_job(&args);
            job.validate()?;
            vec![job]
        }
    };
    let batch = args.input.is_some();

    // In batch mode, skip images that already exist unless --overwrite is set.
    let mut todo = Vec::new();
    let mut skipped = Vec::new();
    for job in jobs {
        match job.existing_output() {
            Some(existing) if batch && !args.overwrite => {
                println!("Skipping line {}: {} exists", job.line, existing.display());
                skipped.push(job);
            }
            _ => todo.push(job),
        }
    }
    if todo.is_empty() {
        println!("Nothing to do (use --overwrite to regenerate).");
        return Ok(());
    }

    println!("Z-Image Text-to-Image Generation");
    println!("================================");
    println!("Model: {:?}", args.model);
    if let Some(path) = &args.input {
        println!(
            "Input: {} ({} to generate, {} skipped)",
            path.display(),
            todo.len(),
            skipped.len()
        );
    }

    let pipeline = Pipeline::load(&args)?;

    let mut failed = Vec::new();
    for (i, job) in todo.iter().enumerate() {
        println!("\n[{}/{}] {}", i + 1, todo.len(), job.output.display());
        println!("Prompt: {}", job.prompt);
        println!("Size: {}x{}", job.width, job.height);
        println!("Steps: {}", job.num_steps);
        println!("Guidance scale: {}", job.guidance_scale);
        let kind = if job.random_seed { " (random)" } else { "" };
        println!("Seed: {}{kind}", job.seed);
        if let Some(parent) = job.output.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let start = std::time::Instant::now();
        match pipeline.generate(job) {
            Ok(()) => println!(
                "Done! Image saved to {} ({:.1}s)",
                job.output.display(),
                start.elapsed().as_secs_f64()
            ),
            Err(e) if !batch => return Err(e),
            Err(e) => {
                eprintln!("Line {} failed: {e:#}", job.line);
                failed.push(job.line);
            }
        }
    }

    if batch {
        println!(
            "\nBatch finished: {} generated, {} skipped, {} failed",
            todo.len() - failed.len(),
            skipped.len(),
            failed.len()
        );
        if !failed.is_empty() {
            anyhow::bail!("failed lines: {:?}", failed);
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    run(args)
}

fn device(cpu: bool) -> Result<candle::Device> {
    if cpu || !candle::utils::metal_is_available() {
        if !cpu {
            println!("Metal not available, running on CPU");
        }
        Ok(candle::Device::Cpu)
    } else {
        Ok(candle::Device::new_metal(0)?)
    }
}

fn save_image<P: AsRef<std::path::Path>>(img: &Tensor, p: P) -> Result<()> {
    let (channel, height, width) = img.dims3()?;
    anyhow::ensure!(
        channel == 3,
        "save_image expects a (3, height, width) tensor"
    );
    let pixels = img.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
    let image: image::RgbImage = image::ImageBuffer::from_raw(width as u32, height as u32, pixels)
        .ok_or_else(|| anyhow::anyhow!("error saving image {:?}", p.as_ref()))?;
    image.save(p)?;
    Ok(())
}
