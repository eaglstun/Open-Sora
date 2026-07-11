# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Open-Sora is an open-source text/image-to-video diffusion model. This `main` branch is **Open-Sora 2.0** (11B), an MMDiT (multimodal diffusion transformer) with rectified-flow sampling, conditioned by a T5 + CLIP text stack and decoded through a video autoencoder. Training and multi-GPU parallelism run on [ColossalAI](https://github.com/hpcaitech/ColossalAI). The package is `opensora` (version 2.0.0), CUDA/NVIDIA-only.

Older versions (1.0–1.3) live on separate branches (`opensora/v1.x`), not here. Don't cross-reference their configs or model code.

## Environment & install

CUDA GPU is required — there is no CPU or Apple-Silicon path (inference falls back to `cpu` device but the models won't realistically run there). Target env is Python 3.10, `torch>=2.4.0`.

```bash
pip install -v -e .    # editable/dev install (drops the -e for a plain install)
pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121  # match your CUDA
pip install flash-attn --no-build-isolation
```

Model weights are **not** in the repo — download the 11B checkpoint into `./ckpts`:

```bash
huggingface-cli download hpcai-tech/Open-Sora-v2 --local-dir ./ckpts
```

## Running things

Everything is launched with `torchrun` (even single-GPU), pointing at a **script** + a **config file**, with optional dotted CLI overrides. There is no `argparse` flag list to memorize — see the config-override mechanism below.

**Inference (text→image→video, the recommended path):**

```bash
torchrun --nproc_per_node 1 --standalone scripts/diffusion/inference.py \
  configs/diffusion/inference/t2i2v_256px.py --save-dir samples --prompt "raining, sea"
```

- 768px config: `t2i2v_768px.py`; direct text→video (skip the flux T2I stage): `256px.py` / `768px.py`.
- Image→video: add `--cond_type i2v_head --ref assets/texts/i2v.png`.
- Batch from a CSV instead of `--prompt`: `--dataset.data-path assets/texts/example.csv`.
- Multi-GPU: bump `--nproc_per_node` (768px uses ColossalAI sequence parallelism; 256px `_tp` config uses tensor parallelism + `--offload True`).
- Memory: `--offload True`. Reproducibility: `--seed 42 --sampling_option.seed 42`.
- Prompt refine via ChatGPT: `export OPENAI_API_KEY=...` then `--refine-prompt True`. Motion score: `--motion-score 4` (or `dynamic`, needs the OpenAI key).

**Training / VAE:** `scripts/diffusion/train.py` (configs under `configs/diffusion/train/`, e.g. `stage1.py`, `stage2.py`, `*_i2v.py`). VAE training/inference under `scripts/vae/` + `configs/vae/`. Full walkthrough (dataset prep, columns, TensorNVMe for checkpointing) is in `docs/train.md`. Training requires `pip install git+https://github.com/hpcaitech/TensorNVMe.git`.

**Gradio demo:** `gradio/app.py`.

There is **no test suite and no lint make-target** in this repo. Code style is enforced only via pre-commit (black, isort, autoflake) — set it up with `pre-commit install`. Run `pre-commit run --all-files` before committing.

## Architecture

### Config-driven everything (read this first)

The system is built on **mmengine `Config` files + a registry**, not on function arguments. A config `.py` is a plain Python module of dict-like settings; `MODELS`/`DATASETS` registries (`opensora/registry.py`) turn a dict with a `type` key into an instantiated `nn.Module` via `build_module(...)`.

CLI overrides are parsed by hand in `opensora/utils/config.py::merge_args`: any `--a.b.c value` after the config path sets `cfg.a.b.c`, with the value coerced to the type already present in the config (so `--offload True` becomes a bool, `--num_frames 129` an int). `parse_alias` then maps a handful of convenience keys (`--resolution`, `--num_frames`, `--aspect_ratio`, `--ckpt_path`, `--guidance`, ...) onto their real nested homes under `cfg.sampling_option` / `cfg.model`. **When adding a tunable, add it to a config and let it flow through — don't add argparse flags.**

Inference configs compose via a base + `plugins/` (`sp.py` sequence-parallel, `tp.py` tensor-parallel, `t2i2v.py` the flux-T2I-then-video pipeline).

### Pipeline (scripts/diffusion/inference.py)

1. `parse_configs()` → `parse_alias()` build the config; `init_inference_environment()` sets up the ColossalAI distributed env.
2. `opensora/utils/sampling.py::prepare_models` builds the model stack; `prepare_api` returns the sampling closure. Core objects are `SamplingOption` and `sanitize_sampling_option`.
3. Text/image conditioning, denoising loop (rectified flow), VAE decode, then `process_and_save` writes videos.

### Package map (`opensora/`)

- `models/mmdit/` — the 11B diffusion transformer. `model.py` (the net), `layers.py`, `math.py` (attention/rope), `distributed.py` + `policy.py` (`MMDiTPolicy` — the ColossalAI shardformer policy for SP/TP).
- `models/text/conditioner.py` — text encoder stack (T5 + CLIP).
- `models/vae/`, `models/hunyuan_vae/`, `models/dc_ae/` — video autoencoders. 2.0 uses a DC-AE-style / HunyuanVideo VAE; `AE_SPATIAL_COMPRESSION` is set from `cfg.ae_spatial_compression` as an env var. VAE docs: `docs/ae.md`, `docs/hcae.md`.
- `acceleration/` — parallelism plumbing: `parallel_states.py` (process groups), `shardformer/`, `checkpoint.py` (activation/grad checkpointing).
- `datasets/` — dataloaders + `aspect.py` (aspect-ratio bucketing; `bucket_to_shapes`). Training/inference data is CSV/parquet with columns `path,text,num_frames,height,width,aspect_ratio,resolution,fps`.
- `utils/` — `config.py`, `sampling.py`, `inference.py`, `ckpt.py` (`CheckpointIO`, sharding), `cai.py` (ColossalAI booster helpers), `prompt_refine.py`, `misc.py`.
- `scripts/cnv/` — checkpoint conversion/sharding utilities.

### Constraints worth remembering

- `num_frames` must be `4k+1` and `< 129`. `aspect_ratio` ∈ `16:9, 9:16, 1:1, 2.39:1`.
- Default dtype is `bf16` (`to_torch_dtype`).
- Distributed correctness lives in the shardformer policies + `parallel_states` — changes to the transformer's forward must stay consistent with `MMDiTPolicy` or SP/TP runs will silently produce wrong shapes.

## Apple Silicon / MPS lane (experimental, on branch `feature/apple-silicon-mps`)

The upstream repo is CUDA-only. A single-device MPS lane exists for running **256px direct text-to-video inference** on Apple Silicon (needs ~40GB of the 64GB unified memory; 768px and the flux t2i2v pipeline are out of scope). Methodology mirrors the `finetrainers-mps` port: carve a `world_size=1` lane that bypasses the distributed machinery rather than porting it.

**The one rule: launch with plain `python`, NOT `torchrun`.** `torchrun` sets distributed env vars that route the device to CPU and trigger uninitialized-process-group crashes. Every README command must drop the `torchrun --nproc_per_node 1 --standalone` prefix on a Mac:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 OPENSORA_DEVICE=mps \
  python scripts/diffusion/inference.py configs/diffusion/inference/256px.py --prompt "raining, sea"
```

- **Device selection** is centralized in `opensora/utils/device.py::get_device()` — `OPENSORA_DEVICE` override → MPS → CUDA → CPU. `OPENSORA_DEVICE=cpu` runs a CPU oracle on the same machine. Import `DEVICE`/`get_device()` instead of writing `"cuda"`.
- **CUDA-only deps are wrapped, not required.** `flash_attn` → torch **SDPA** fallback and `float64`→`float32` rope (both in `mmdit/math.py`); `liger_kernel` rope/RMSNorm → pure-torch (the default `use_liger_rope=False` path never calls liger anyway); `tensornvme` stubbed in `utils/ckpt.py` (inference never saves). `xformers` in the VAE was already guarded upstream (SP-only). `dist.barrier()` in `inference.py` is wrapped by `_safe_barrier()`.
- **colossalai imports fine on Mac** once its pure-Python deps are present (`pip install colossalai psutil galore-torch bitsandbytes ...`) — the ws=1 lane imports its policies but never _calls_ the booster (`init_inference_environment()` is a no-op off-torchrun; `get_booster()` returns `None` unless `plugin=="hybrid"`).
- **"It ran" ≠ "it's correct."** MPS bugs are silent wrong numbers. `tests/mps/test_cpu_mps_parity.py` gates the MMDiT forward (SDPA + float32 rope) CPU-vs-MPS at fp32 atol 1e-4. Run it after touching any hot path. Extend it before trusting a new module on MPS.
- **Version drift risk:** the port runs on torch 2.10 / torchvision 0.25, not the pinned 2.4 / 0.19. Imports are clean but runtime API breakage is possible — parity-test, don't assume.

## Reports

Design/tech reports per version are in `docs/report_0{1..4}.md` (04 = v1.3); the 2.0 report is the arXiv paper (2503.09642).
