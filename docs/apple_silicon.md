# Running Open-Sora 2.0 on Apple Silicon (MPS)

Experimental single-device (MPS) lane for **256px inference**, on branch
`feature/apple-silicon-mps`. Upstream is CUDA-only; this carves a `world_size=1`
lane that _bypasses_ colossalai's distributed machinery rather than porting it
(same approach as the `finetrainers` MPS port). Validated on an M-series Mac,
64 GB unified memory, generating coherent rain-on-sea video.

Not in scope: 768px, the flux text-to-image-to-video pipeline, training, and the
full 129-frame (~5s) video — see **Limits** below.

---

## The one rule

**Launch with plain `python`, NOT `torchrun`.** `torchrun` sets distributed env
vars that route the device to CPU and trigger uninitialized-process-group
crashes. Drop the `torchrun --nproc_per_node 1 --standalone` prefix everywhere.

## Working command (this produces the rainy-sea clip)

```bash
HF_HUB_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 OPENSORA_DEVICE=mps \
  python scripts/diffusion/inference.py configs/diffusion/inference/256px.py \
  --prompt "raining, sea" --num_frames 13 --num_steps 30 --num-sample 1 --save-dir samples
```

Output lands in `samples/video_256px/` (or `samples/image_256px/` when
`--num_frames 1`).

### Image-to-video (`i2v_head`) — also works on MPS

Add `--cond_type i2v_head --ref assets/texts/i2v.png` to animate a still image
(frame 0 becomes the reference; the rest is generated). **Keep `--prompt`** — the
reference rides along with it via a temp CSV; `--ref` without `--prompt` crashes in
dataset build. Verified 2026-07-12: the hunyuan VAE _encoder_ runs on native Metal
and is CPU↔MPS parity-clean (`tests/mps/test_cpu_mps_parity_vae.py`). `i2v_tail`
and `i2v_loop` also verified 2026-07-12 — tail splices the ref at the **last** frame
(`--cond_type i2v_tail`); loop takes two `;`-separated refs for first+last
(`--ref "a.png;b.png"`; reuse one image for a seamless loop). Same encode/denoise,
different splice — no code change.

### Env vars

Two are load-bearing (`OPENSORA_DEVICE`, `HF_HUB_OFFLINE`);
`PYTORCH_ENABLE_MPS_FALLBACK` is optional insurance — see the note under the table.

| Var                             | Why                                                                           |
| ------------------------------- | ----------------------------------------------------------------------------- |
| `OPENSORA_DEVICE=mps`           | forces the device shim (`opensora/utils/device.py`); `=cpu` runs a CPU oracle |
| `HF_HUB_OFFLINE=1`              | weights are already local in `./ckpts`; skip HF network checks                |
| `PYTORCH_ENABLE_MPS_FALLBACK=1` | **optional, not load-bearing** — a net that routes any kernel-less op to CPU  |

> **`TORCHDYNAMO_DISABLE=1` is retired (P5, 2026-07-12).** It used to be required
> because a `@torch.compile(max-autotune)` on `timestep_embedding` stalled on MPS;
> that decorator is now CUDA-only (`is_cuda()`-gated), so the var is no longer needed
> — a default render without it is bit-identical (verified). Dropped from the command
> above. (Don't re-add it if you experiment with `--compile_mmdit True`: dynamo must
> be live for that flag to do anything.)

> **The fallback net is currently inert.** Verified 2026-07-11 on torch 2.10 by
> running both paths with the flag _unset_: nothing in the 256px t2v pipeline falls
> back — the denoise loop **and** the temporal 3D VAE decode run entirely on native
> Metal kernels, and image (`--num_frames 1`) and video (`--num_frames 13`) both
> complete cleanly. (Without the net, an unsupported op _raises_ rather than silently
> running on CPU; nothing raised.) Consequence for tuning: **CPU-fallback hunting is
> not a speed lever here** — the hot loop is already all-Metal. Keep the flag only as
> insurance for untested paths (i2v, other resolutions).

## ⚠️ The three things that will waste your time

1. **`num_frames` defaults to 129** (a ~5s video). That is the single biggest
   trap. A 129-frame denoise step took **~18 min/step** and looked "stuck." Use
   `--num_frames 1` (image) or `13` (short clip) for anything interactive.
   Valid values are `4k+1`.
2. **`--offload True` IS A NO-OP. The real flag is `--offload_model True`.**
   Nothing reads `cfg.offload` — the code reads `cfg.get("offload_model")`
   (`inference.py:141/189/227`), and `offload` is not in `parse_alias`. The
   upstream README (and, until 2026-07-12, these docs) say `--offload True`, which
   silently does **nothing** — you get no model offload and wonder why you're
   swapping. Verified: `--offload_model True` does land as `cfg.offload_model=True`.
3. **Memory is the wall on unified memory.** T5-XXL (~19 GB) + the 11B MMDiT
   (22 GB) + activations overflows 64 GB into swap → the process sits in
   uninterruptible sleep thrashing the disk at ~20% CPU, making ~zero progress.
   For small workloads (1 or 13 frames) everything-resident is fine and fastest.
   For ≥29 frames use **`--offload_text_encoders True`** (P4 — frees 9.5 GB,
   bit-identical, takes 29f from ~10 GB swap to ~2.5 GB). Note 49f completes even
   with nothing offloaded (it just swaps); whether real `--offload_model True`
   helps has **not** been measured.

## Observed timings (M-series, 64 GB, 256px)

| Frames | Steps | Time          | Notes                                                          |
| ------ | ----- | ------------- | -------------------------------------------------------------- |
| 1      | 4     | ~16 s denoise | soft blue blob; validates the pipeline                         |
| 13     | 8     | ~8 min        | coherent but abstract; first run pays Metal kernel compilation |
| 13     | 30    | ~4 min        | **sharp waves + rain + motion**; kernels already warm          |
| 129    | any   | hours / swaps | impractical here                                               |

First run compiles Metal kernels (slow); later runs reuse them. Step count is
the quality dial: 8 = broad color/composition, 30 = real detail.

## One-time setup on macOS arm64

The stack runs on **Python 3.13.12 / torch 2.10.0 / torchvision 0.25.0** (not the
pinned 2.4 / 0.19); imports are clean but it's the source of the version-drift bugs
the port guards. colossalai, flash-attn, liger, xformers, tensornvme have **no arm64
build** — the port makes them optional. colossalai itself _imports_ fine once its
pure-Python deps are present.

> **Interpreter trap (this will waste your time).** The working stack is installed in
> a **conda env** (here, miniconda3 `base`, Python 3.13). If a pyenv/system `python`
> sits earlier on `PATH`, bare `python` resolves to _that_ (e.g. pyenv 3.14) and dies
> at import with `ModuleNotFoundError: No module named 'colossalai'` before it ever
> reaches a model. Confirm `python -c "import colossalai"` succeeds before launching —
> or call the conda interpreter by full path.

```bash
# core
pip install mmengine omegaconf einops ftfy diffusers pandas pyarrow pandarallel av \
            tensorboard wandb openai pytest safetensors

# colossalai + its pure-python dep tree (no CUDA build)
pip install colossalai --no-deps
pip install psutil packaging peft accelerate galore-torch --no-deps
pip install bitsandbytes            # 0.49+ multi-backend imports on Mac (CPU fallback)

# the package itself
pip install -e . --no-deps
```

Do **not** try to install flash-attn / liger-kernel / xformers / tensornvme.

## What the port changed (and why)

All on `feature/apple-silicon-mps`; see `git diff main`.

- `opensora/utils/device.py` (new) — device shim: `OPENSORA_DEVICE` → MPS → CUDA → CPU.
- `mmdit/math.py` — flash-attn → torch **SDPA** fallback; **float32 rope** (MPS has no
  float64); pure-torch **liger rope** (`liger_rope_torch`, the `rotate_half`
  convention) since the 256px checkpoint uses `use_liger_rope=True`.
- `mmdit/layers.py` — liger RMSNorm → pure-torch fallback.
- `mmdit/distributed.py` — flash/liger guarded (sequence-parallel path, dead at ws=1).
- `text/conditioner.py` — skip colossalai shardformer on T5 off-CUDA (it does no
  tensor-parallelism at ws=1 and its forward monkeypatch breaks on transformers ≥5).
- `utils/ckpt.py` — `tensornvme` stubbed (inference never saves checkpoints).
- `utils/misc.py` — `log_cuda_*` no-op off-CUDA.
- `scripts/diffusion/inference.py` — device shim; `_safe_barrier()` (no-op without a
  process group); `pin_memory`/`num_workers` disabled off-CUDA (pin*memory calls
  `torch.cuda.current_device()`; macOS \_spawns* workers and can't pickle the
  `seed_worker` closure).
- `datasets/_video_io.py` (new) + `datasets/read_video.py` + `datasets/utils.py` +
  `scripts/cnv/meta.py` — video read/write ported off `torchvision.io.video` onto
  pyav (`av`). torchvision **removed** that API in 0.27 (and ships a broken fbcode
  stub in 0.28), so the old imports hard-crash on any torch ≥2.12. This decouples the
  data layer from the torchvision video API and is the prerequisite for running on
  newer torch. (av ≥15 gotcha: `frame.pict_type = "NONE"` — the string form — is
  rejected; omit it, NONE is the encoder default.)

## Upgrading torch — read this first

### Why video I/O is pyav-native (and why that unblocks you)

**torchvision deleted its video API.** `torchvision.io.video` — `read_video`,
`write_video`, `_check_av_available` — was **removed in torchvision 0.27**, and 0.28
ships it as a **broken fbcode stub** (`from pytorch.vision.fb.io.video import …`,
which raises `ModuleNotFoundError: No module named 'pytorch'`). The data layer used
to import those names at module load, so **any torch ≥ 2.12 hard-crashed on import**
— not just on MPS, on CUDA too, since torch pins torchvision.

**Fix:** video I/O is now **torchvision-independent**, backed directly by pyav:

- `opensora/datasets/_video_io.py` (new) — pyav `write_video` + `_check_av_available`.
- `opensora/datasets/read_video.py` — was already pyav-native; just dropped its
  `torchvision.io.video` / `get_video_backend` imports.
- `opensora/datasets/utils.py`, `scripts/cnv/meta.py` — redirected off `torchvision.io`.

torchvision is still used for **transforms**, `datasets.folder`, and `utils.save_image`
— those APIs are stable. Only the _video_ API is gone. **Do not reintroduce an import
from `torchvision.io.video`; it does not exist in any modern torchvision.**

### The upgrade playbook

Verified matrix (both pass the parity suite and render correctly):

| torch  | torchvision | status                                  |
| ------ | ----------- | --------------------------------------- |
| 2.10.0 | 0.25.0      | ✅ baseline / CPU-parity oracle         |
| 2.13.0 | 0.28.0      | ✅ **−17% warm on MPS**, throttles less |

Pairing rule of thumb: **torchvision minor = torch minor + 15** (2.10→0.25, 2.11→0.26,
2.12→0.27, 2.13→0.28). Let pip resolve the pair rather than pinning torchvision by hand.

When you bump torch:

1. **Use a dedicated env — never mutate the working one.** A
   `python -m venv --system-site-packages` over the conda base works well: it inherits
   every dependency and you only install the new `torch`/`torchvision` into it, which
   shadow the base copies. Keep the old env intact as the **CPU-parity oracle** and as
   a fallback if the bump goes badly.
2. **Bump torch and torchvision together.** A mismatched torchvision fails on its
   compiled `_C` extension (it's ABI-locked to a specific torch).
3. **Run the parity suite — this is the gate.** `python -m pytest -q tests/mps/`
   (12 tests: MMDiT, both VAEs, T5/CLIP, compiled-vs-eager). MPS bugs are _silent wrong
   numbers_, so a clean import proves nothing. This is what makes a torch bump a
   10-minute check instead of a re-derivation.
4. **Render something and look at it.** Seed-matched (`--seed 42
--sampling_option.seed 42`) against the old env; pixels should be near-identical
   (not bit-identical — kernel numerics differ across torch versions).
5. **Time it with cooldowns.** Thermal drift is ~+28% cold→warm and will forge a naive
   A/B. Use the `mps-bench` skill.

### Landmines seen on real upgrades

- **`av` ≥ 15 rejects `frame.pict_type = "NONE"`** (the legacy string form) with
  `TypeError: an integer is required`. Omit the assignment — `NONE` is the encoder
  default. (Already handled in `_video_io.py`.)
- **torchvision ≥ 0.27 has no video API** (above). If a new import of it appears
  anywhere, that's the bug.
- **A `pip install -U torch` in the shared env is how you lose an afternoon** —
  torchvision follows torch, and if anything in the tree still touched
  `torchvision.io.video` it would break the CUDA path too.

## Correctness discipline

MPS bugs are silent wrong numbers, not crashes. `tests/mps/test_cpu_mps_parity.py`
gates the MMDiT forward (both rope paths) CPU-vs-MPS at fp32 atol 1e-4:

```bash
python -m pytest -q tests/mps/test_cpu_mps_parity.py
```

Caveat: liger is CUDA/Triton-only, so there is **no local oracle** for the rope
fallback — the parity test only proves CPU==MPS, not ==liger. The real
acceptance test is the pixels: a wrong rope convention yields noise, not a
coherent seascape.

## Limits

- **768px / flux t2i2v**: not ported (flux adds ~24 GB; memory + time prohibitive).
- **Full 129-frame video**: impractical on MPS — that's a rented CUDA GPU job.
- **Speed**: ~1 min per 13-frame denoise step once warm; fine for short clips,
  not for batches.

Next phase (i2v, wider parity, frame-count frontier, compile): see
[apple_silicon_roadmap.md](apple_silicon_roadmap.md).
