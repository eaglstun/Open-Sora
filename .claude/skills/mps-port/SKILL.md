---
name: mps-port
description: The playbook for extending Open-Sora's Apple Silicon (MPS) world_size=1 lane to a new code path — i2v, other resolutions, flux t2i2v, new modules, or a torch upgrade. Use when porting/enabling anything on this repo's MPS branch, when a run crashes on a CUDA-only import or missing Metal kernel, when adding a tunable to the mmengine config system, or before trusting a newly-exercised module on MPS. Captures the guard-don't-port method, the ordered verification ritual, and the traps that have already cost hours.
---

# Extending the MPS lane

Read `docs/apple_silicon.md` first (ground truth for what works and the exact
run commands) and `docs/apple_silicon_roadmap.md` (what's next and why). This
skill is the _procedure_ for making a new path work without silently making it
wrong.

## The method: guard, don't port

The lane is `world_size=1`, launched with **plain `python`, never `torchrun`**.
Distributed machinery (colossalai boosters, shardformer, SP/TP) is bypassed,
not ported — it must still import cleanly for CUDA users, so:

- **CUDA-only imports** (`flash_attn`, `liger_kernel`, `xformers`,
  `tensornvme`, apex-likes): wrap in `try/except ImportError` with a pure-torch
  fallback, exactly like `opensora/models/mmdit/math.py` (flash-attn → SDPA,
  liger rope → `liger_rope_torch`). Never make them hard requirements.
- **Device strings**: route through `opensora/utils/device.py::get_device()` /
  `DEVICE` — never write `"cuda"` / `.cuda()` in a path the lane touches.
  Recon first: `grep -rn '"cuda"\|\.cuda()\|torch\.cuda\.' <files>` over the
  new path; also grep for `dist.` calls (guard with the `_safe_barrier()`
  pattern from `scripts/diffusion/inference.py`).
- **MPS dtype gaps**: no `float64` (cast to float32 like `math.py::rope`); bf16
  is the working dtype, **fp16 is dead** (the 11B overflows → black frames).
- **Tunables go in mmengine configs, not argparse.** CLI `--a.b.c` overrides
  are type-coerced from the value already in the config
  (`opensora/utils/config.py::merge_args`; convenience keys in `parse_alias`).
  A key that doesn't exist in the config can't be overridden — add it there.

## The verification ritual (in this order, no skipping)

1. **Smallest workload first.** `--num_frames 1`, `--num_steps 4`, one sample.
   Prove the path executes before spending minutes per attempt.
2. **Fallback-unset smoke run.** Run once with `PYTORCH_ENABLE_MPS_FALLBACK`
   _unset_: a missing Metal kernel then **raises** instead of silently running
   on CPU. (`mps-bench`'s `--no-fallback-net` does this.) If it raises, decide:
   accept a CPU fallback (slow, correct) or rewrite the op.
3. **Parity before pixels.** "It ran" proves nothing — MPS bugs are silent
   wrong numbers. Extend the CPU↔MPS parity tests for any newly-exercised
   module (tiny random-weight config, CPU-seeded inputs, fp32 atol 1e-4) —
   delegate to the `parity-runner` agent, which knows the authoring pattern
   and the coverage map.
4. **Pixel acceptance.** Generate the real thing and _look at it_ (and run
   `mps-bench`'s `inspect` for the black/flat-frame gates). For ops with no
   local oracle (liger is CUDA-only), pixels ARE the acceptance test.
5. **Memory probe before scale-up.** Watch RSS while stepping up frames/res.
   The swap-thrash signature: process in uninterruptible sleep, ~20% CPU, zero
   progress — kill it, use `--offload True` or a smaller workload.
6. **Timing last**, thermally controlled, via the `mps-bench` skill. Never
   conclude speed from a single hot run.

## Traps already paid for (do not relearn)

| Trap                           | Reality                                                                                                                                                                                  |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `torchrun`                     | routes device to CPU + uninit-process-group crashes; plain `python` only                                                                                                                 |
| `num_frames` default           | **129** (~18 min/step, looks hung). Always pass `1` or `13`; valid values `4k+1`                                                                                                         |
| `--offload` off + big workload | T5(19GB)+MMDiT(22GB)+activations > 64GB → swap thrash                                                                                                                                    |
| fp16                           | black frames (overflow), and not faster — bf16 stays                                                                                                                                     |
| `TORCHDYNAMO_DISABLE=1`        | **retired (P5)** — `timestep_embedding`'s compile is now CUDA-only (`is_cuda()`-gated); the var is inert on MPS, drop it. Don't set it if using `--compile_mmdit` (dynamo must be live). |
| float64 on MPS                 | unsupported — cast rope-style math to float32                                                                                                                                            |
| DataLoader workers on macOS    | spawn (not fork) can't pickle local closures; `num_workers=0`, `pin_memory` only on CUDA                                                                                                 |
| `av >= 15`                     | rejects `frame.pict_type = "NONE"` string form — omit it (see `opensora/datasets/_video_io.py`)                                                                                          |
| wrong interpreter              | pyenv `python` has no colossalai; use `$OPENSORA_MPS_PY` (torch 2.13) or `~/miniconda3/bin/python` (torch 2.10 oracle); check `python -c "import colossalai"` first                      |
| version drift                  | stack runs torch 2.10/2.13, repo pins 2.4 — imports clean ≠ runtime-safe; parity-test after any bump                                                                                     |

## Where the port lives

`opensora/utils/device.py` (shim) · `opensora/models/mmdit/math.py` +
`layers.py` (SDPA/rope/RMSNorm fallbacks) · `opensora/models/text/conditioner.py`
(shardformer skip off-CUDA) · `opensora/utils/ckpt.py` (tensornvme stub) ·
`opensora/datasets/_video_io.py` (pyav I/O, the torch≥2.12 prerequisite) ·
`scripts/diffusion/inference.py` (`_safe_barrier`, dataloader guards) ·
`tests/mps/` (parity). Diff it all with `git diff main`.
