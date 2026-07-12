---
name: mps-bench
description: Thermal-aware benchmarking of Open-Sora inference on Apple Silicon — timed 256px renders with enforced cooldowns, cold/warm separation, model-load vs render split, output-pixel sanity gates (black/flat-frame detection), and baseline regression comparison. Use when asked to benchmark, time, profile, or A/B Open-Sora on MPS ("is X faster", "did my change regress render time", "compare torch versions", "measure N frames"), or to check whether an output video is degenerate. Not for correctness/parity — that's the parity-runner agent.
---

# Benchmarking Open-Sora on MPS

The harness is `scripts/osora_bench.py`. It wraps `scripts/diffusion/inference.py`
as a subprocess (plain `python`, never `torchrun`), timestamps the script's own
log markers (`Building models...` → `Generating video...` → `Inference finished.`)
to split **model load** from **render** (denoise + VAE decode + save), and
sanity-checks the output pixels. Results are JSON; `compare` exits non-zero on
regression, so it drops into pass/fail logic.

**Run it with the inference interpreter** (it needs `av` for the output check):

```bash
cd /Users/eeaglstun/Documents/dev/Open-Sora
"$OPENSORA_MPS_PY" .claude/skills/mps-bench/scripts/osora_bench.py run \
    --label "13f/20step, torch 2.13" \
    --out .claude/skills/mps-bench/baselines/t2v_13f20s.torch213.mps.json
```

- Default interpreter: `$OPENSORA_MPS_PY` (`~/venvs/opensora-torch213/bin/python`,
  torch 2.13) or self. The conda-base oracle is `~/miniconda3/bin/python`
  (torch 2.10) — pass `--python` to A/B envs.
- Default workload = the documented probe: `--prompt "raining, sea"
--num_frames 13 --num_steps 20 --seed 42 --sampling_option.seed 42` on
  `configs/diffusion/inference/256px.py`. Override by appending the full
  inference workload after a literal `--` separator, passed through verbatim:
  `run --label "29 frames" -- --prompt "raining, sea" --num_frames 29 ...`
  (an override replaces the whole default workload, so re-pin the seeds).
- `--no-fallback-net` unsets `PYTORCH_ENABLE_MPS_FALLBACK` so any missing Metal
  kernel **raises** instead of silently running on CPU — the smoke test for a
  newly ported path (as of 2026-07-11 the 256px t2v path needs no fallback).

## Thermal methodology (the whole point)

Sustained MPS load throttles this machine **+13–28% cold→warm** — bigger than
most optimizations you'll measure. Non-negotiables:

1. **Repeat ≥2 with cooldowns.** Repeat 1 = cold, later repeats = warm. Default
   `--cooldown 180` between repeats; don't shrink it for a "quick" number.
2. **Compare like positions only**: cold-vs-cold, warm-median-vs-warm-median.
   The harness reports both plus `cold_to_last_drift_pct` (a drift far above
   ~30% means the machine was already hot — rerun after a rest).
3. **A/B order bias**: whichever variant runs second runs hotter. For a fair
   A/B either alternate (A,B,A,B) with cooldowns or run on different rests.
   Note: torch 2.13 beat 2.10 _despite_ running second — that's a robust win.
4. **First-ever run compiles Metal kernels** — treat it as a throwaway, not
   even a "cold" number.

## Reading results

- `render_s` is the number that matters (`load_s` is disk/HF ceremony).
- `--threshold` default 10% — thermal noise makes 5% uncallable here.
- **Output flags are correctness, not noise**: `BLACK_FRAMES` is the fp16
  overflow signature, `FLAT_FRAMES` means the model produced nothing. Any flag
  = exit 2 = stop and debug, regardless of how fast it ran.
- Cross-`machine`/`device` comparisons are refused as regressions; cross-torch
  comparisons are labeled env A/Bs. Baselines are per (workload × env × machine).
- A speed win with different pixels isn't a win yet — seed-matched output plus
  the parity test (parity-runner agent) gate any hot-path change.

## Reference numbers (M4 Max 64 GB, 256px, 13f/20-step, seed 42)

| env                  | cold render | warm render | drift |
| -------------------- | ----------- | ----------- | ----- |
| torch 2.10 (oracle)  | 137.6 s     | 176.5 s     | +28%  |
| torch 2.13 (default) | 129.0 s     | 146.5 s     | +13%  |

Model load adds ~1–2 min per run. A 2-repeat run with cooldown takes ~15 min —
**longer than a foreground Bash timeout**, so launch benchmarks in the
background and poll, or run repeats as separate invocations.

Baselines live in `baselines/` (committable — this is a single-machine project,
and the harness stamps each with `git_sha`/`dirty`/torch/machine so a stale one
is self-identifying). Per-run logs land in `scripts/logs/`, which the repo's
`logs/` gitignore rule already covers. Regenerate baselines after switching
machines or envs rather than comparing across them.

## The agent

Delegate whole benchmarking jobs ("did X regress", "A/B these envs", "time 29
frames") to the `bench-runner` agent (`.claude/agents/bench-runner.md`) — it
runs this harness end-to-end with the cooldown discipline and reports a verdict.
