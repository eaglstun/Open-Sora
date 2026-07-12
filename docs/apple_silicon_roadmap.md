# Apple Silicon (MPS) lane — next-phase roadmap

Continuation of [apple_silicon.md](apple_silicon.md) (read that first — it's the
ground truth for what already works). Status as of 2026-07-12: 256px t2v runs
all-Metal on an M4 Max / 64 GB, torch 2.13 venv (`~/venvs/opensora-torch213`,
`osora-mps` in `~/.zshrc`) with conda `base` (torch 2.10) as the parity oracle.

Items are ordered by value ÷ (effort × risk) on **this** machine. Honest budget
framing: a warm 13f/20-step render is ~2.5 min, model load ~1–2 min, and thermal
drift is ~+13–28% cold→warm — so every timed A/B costs ~15–30 min with cooldowns,
and anything that scales attention quadratically gets expensive fast.

## Ground rules (every item below)

1. **Smallest workload first** — `--num_frames 1`, few steps, before any scale-up.
2. **Fallback-unset smoke run** — once per new code path, run with
   `PYTORCH_ENABLE_MPS_FALLBACK` _unset_ so a missing Metal kernel raises instead
   of silently running on CPU.
3. **Parity before pixels** — extend `tests/mps/test_cpu_mps_parity.py` (or a
   sibling) to cover any newly-exercised module before trusting its output.
   Delegate to the `parity-runner` agent.
4. **Memory probe before scale-up** — watch RSS; the swap-thrash failure mode is
   a process in uninterruptible sleep at ~20% CPU making zero progress.
5. **Thermally controlled timing** — use the `mps-bench` skill
   (`.claude/skills/mps-bench/`); never compare a cold number to a warm one.

---

## P1 — Image-to-video (`i2v_head`) at 256px ✅ DONE (2026-07-12)

**Result:** works on MPS at 256px, torch 2.13, first try — no code change needed.
Ran the full ritual: fallback-unset smoke passed (the hunyuan **VAE encode**, never
before run on MPS, executes on native Metal — no kernel gaps); CPU↔MPS parity added
and green (`tests/mps/test_cpu_mps_parity_vae.py` — encode ~1e-7, decode ~1e-5,
roundtrip ~4e-5, both torch lanes); pixel acceptance passed (frame 0 _is_ the
reference image; 13f/30-step render animates it coherently — ~193 s warm, on par with
t2v since the 3× CFG batch was already present). **Usage note:** i2v still needs
`--prompt` present — the `--ref` image rides along via `create_tmp_csv`
(`inference.py:85`); `--ref` alone crashes in dataset build. **`i2v_tail`/`i2v_loop`
also done 2026-07-12** — both render coherently on MPS (fallback-unset, no kernel
gaps), same encode/denoise as head; tail splices the ref at frame −1, loop takes two
`;`-joined refs for frames 0 and −1 (reuse one image for a seamless loop). The full
i2v family works — no model-code change for any of the three.

**What:** `--cond_type i2v_head --ref <image>` on the existing 256px config.

**Why it's first:** it's the cheapest real capability gain. The 256px config
already uses `method="i2v"` — plain t2v _already_ runs `I2VDenoiser` with zero
masks and 3×-batched CFG (cond/uncond/uncond₂), so the denoise loop needs
nothing new. The only untested surface is the reference path:
`collect_references_batch` → image read/transform → **`model_ae.encode`** (the
hunyuan VAE encoder — only `decode` has run on MPS so far), plus the
mask/`masked_ref` conditioning tensors and the head-frame splice + pad-trim in
`utils/sampling.py::api_fn`.

- **Effort:** low (hours). Likely works or fails in one obvious place.
- **Risk:** VAE-encode kernel gaps (3D convs going the other direction) or
  silently wrong encode statistics → mushy/drifting first frame.
- **Gate:** fallback-unset smoke run; VAE encode↔decode round-trip parity test
  (tiny random-weight config, CPU oracle); pixel acceptance = frame 0 of the
  output visibly _is_ the reference image (the pipeline literally splices the
  reference latent in before decode, so a wrong encode is unmissable).
- **Bonus for ~free once head works:** `i2v_tail` and `i2v_loop` share the same
  machinery (different splice + trim only).

## P2 — Widen the parity net: VAE + text encoders

**What:** CPU↔MPS parity tests for the hunyuan VAE (encode + decode, tiny
random-weight config, small `(B,C,T,H,W)`) and the `HFEmbedder` T5/CLIP stack.

**Why:** the current test gates only the MMDiT forward. The VAE decode is
trusted on pixels alone; T5/CLIP have never been compared against CPU. Every
other roadmap item (torch upgrades, compile, i2v) leans on these being right,
and MPS regressions are silent wrong numbers — the test is what makes a future
torch bump a 10-minute check instead of a re-derivation.

- **Effort:** medium. VAE: instantiate a small config without `from_pretrained`
  (mirror the tiny-MMDiT pattern). T5/CLIP: either a tiny random HF config built
  offline, or the real checkpoints at fp32 on a short prompt (fits in 64 GB;
  slow but a one-time cost) — prefer whichever gets a deterministic oracle
  without network access (`HF_HUB_OFFLINE=1` is the house style).
- **Risk:** low. Worst case is discovering an existing silent divergence —
  which is the point.
- **Gate:** this _is_ the gate. fp32 atol/rtol 1e-4, same finite-output checks
  as the MMDiT test.

## P3 — Frame-count frontier: 29 → 49 frames ✅ DONE (2026-07-12)

**Result (torch 2.13, 20 steps, seed 42, cold, "raining, sea"):**

> ⚠️ **CORRECTED 2026-07-12.** These runs were originally described as using
> `--offload True`. **That flag is INERT** — nothing reads `cfg.offload` (the code
> reads `cfg.get("offload_model")`, and `offload` is not in `parse_alias`). So **no
> model offload ever ran** in these measurements; every row below is a
> _no-model-offload_ run that simply swapped through. See the footgun note in
> `apple_silicon.md`. The real flag is `--offload_model True` — **still untested**.

| frames | latent | render | memory (no model offload)                 | ~video @24fps | coherent |
| ------ | ------ | ------ | ----------------------------------------- | ------------- | -------- |
| 13     | 4      | 137 s  | comfortable                               | ~0.5 s        | ✅       |
| 29     | 8      | 237 s  | **~10 GB swap, U-state** — the cliff      | ~1.2 s        | ✅       |
| 49     | 13     | 399 s  | heavy swap, free→0 — but **it completes** | ~2.0 s        | ✅       |

**Finding:** render time scales **~linearly** with latent frames (≈30–34 s each), not
quadratically — attention is _not_ the bottleneck (linear/MLP layers dominate), so the
wall is **memory, not compute**. **Verdict: 29 frames is the comfortable ceiling** on
64 GB (~4 min, rides the swap edge); **49 frames still completes** (~6.7 min) purely by
swapping — it does _not_ require model offload (that was the inert-flag error). Beyond
~49f toward 129f is memory-prohibitive. So the lane is for **short clips, ~0.5–2 s**.
All numbers cold; add ~+28% warm. **Open question:** whether real `--offload_model True`
actually helps here has never been measured.

---

**Original plan (for reference):**

**What:** map time + memory between the known-good 13 frames and the impossible 129. With the causal VAE (`temporal_reduction=4`), latent frames =
`(n−1)/4 + 1`: 13f→4, 29f→8, 49f→13, 129f→33. Tokens scale linearly with latent
frames; attention cost quadratically; and remember the CFG batch is 3×.

**Why:** 13 frames is a proof, ~2–3 s at a usable fps is a product. 49f is
~3.25× the tokens of 13f (~10× the attention FLOPs) — plausibly tolerable warm,
possibly memory-fine with `--offload True`. Nobody knows until it's measured,
and the answer defines what this lane is actually _for_.

- **Effort:** low — no code, just runs and patience. Do 29 first, then 49.
- **Risk:** time (a render could be 20–40 min) and swap-thrash at 49f without
  offload. Know the symptom (ground rule 4) and kill early.
- **Gate:** memory watch during the run; coherent pixels; record per-frame-count
  baselines with `mps-bench` so the scaling curve is written down once.

## P4 — Memory lever: release T5 after encode ✅ DONE (2026-07-12)

**Result:** shipped as the config-gated flag `--offload_text_encoders True` (default
off; base config key, inherited everywhere; CUDA path untouched). After the text
embeddings are copied into `inp`, T5+CLIP are moved to CPU and `torch.mps.empty_cache()`
returns the memory (the `empty_cache` is load-bearing — `.to("cpu")` alone leaves the
allocation wired). Measured (torch 2.13, fresh machine, seed 42, 20 steps):

| run                | device mem into denoise           | peak swap  | vs P3 baseline             |
| ------------------ | --------------------------------- | ---------- | -------------------------- |
| device probe (13f) | 35.4 → **25.8 GB** (**−9.54 GB**) | —          | —                          |
| 29f + text-offload | 25.8 GB                           | **2.5 GB** | was ~10 GB (off the cliff) |
| 49f + text-offload | 25.8 GB                           | 9.55 GB    | completes in 421 s         |

**Findings:** frees **9.54 GB** device memory (exactly the bf16 T5+CLIP footprint);
output is **bit-identical** (flag on==off, decoded-frame MD5 match). It's a **memory
lever, not a speed lever** — render times are ~neutral (the T5→CPU move ≈ the swap
saved). **Headline: 29f peak swap drops ~10 GB → 2.5 GB**, i.e. it takes 29f off the
swap cliff. Use it for any frame count ≥29.

> ⚠️ **CORRECTED 2026-07-12.** This section originally claimed "49f now runs without
> the blunt full `--offload` — was impossible before." **That was wrong.** The P3 49f
> baseline passed `--offload True`, which is an **inert flag** (nothing reads
> `cfg.offload`), so it had no model offload either — 49f completed _both_ times
> purely by swapping (399 s then 421 s). Text-offload did **not** "enable" 49f. The
> measured wins above (−9.54 GB device memory, 29f swap 10→2.5 GB, bit-identical) are
> unaffected — they were measured directly, not inferred from the flag.

Re-materialization is per-`api_fn`-call, so
multi-prompt CSVs / `num_sample>1` still work.

**Original plan (for reference):**

**What:** the text embeddings are computed once per prompt at the top of
`api_fn` (`prepare()`); T5-XXL then sits resident through the entire denoise +
decode. Move T5 (and CLIP) to CPU — or drop them — right after encoding, and
re-materialize per batch only if needed.

**Why:** frees ~10–19 GB of unified memory (dtype-dependent) during exactly the
phase where P3 wants it. This is the cheap, targeted alternative to the blunt
`--offload True` model ping-pong, and it funds larger frame counts without
touching the denoise loop.

- **Effort:** low–medium. One well-placed `.to("cpu")` + a re-load path;
  keep it config-gated (mmengine config key, not an argparse flag).
- **Risk:** low. The embeddings are already computed — output must be
  bit-identical.
- **Gate:** seed-matched pixels vs. before (identical, not just similar) +
  before/after memory measurement at 29f/49f.

## P5 — torch.compile / Inductor-on-MPS experiment ✅ DONE (2026-07-12) — negative, but with a bonus

**Result: compile is a measured net SLOWDOWN on MPS torch 2.13 — keep it off.**
Shipped as opt-in `--compile_mmdit True` (default off) purely to reproduce the
negative result. Inductor-Metal compiles the MMDiT blocks cleanly (no stall, no
error), but the generated kernels **lose to eager MPS**: micro-bench at 13f/256px
shapes = DoubleStreamBlock **−27%** (98→125 ms), SingleStreamBlock −4%, weighted
**≈ −12% per forward**. `max-autotune` gives the identical time — Inductor refuses
GEMM autotuning off-CUDA (`"Not enough SMs"`), so there's no better mode. The
roadmap's hoped −10–25% is a **+12% regression**. Numerics are fine: compiled-MPS
vs eager-MPS is **~3e-7 at fp32** (`tests/mps/test_compile_mps_parity.py`, suite now
9 passed) — Fable's ~3e-2 was a bf16 artifact, not a correctness defect. So "keep
off" rests purely on speed.

**Bonus (the real P5 win): `TORCHDYNAMO_DISABLE=1` is retired.** The historic
`timestep_embedding` compile-stall does not reproduce; that decorator is now
`is_cuda()`-gated (CUDA-only), so a default MPS render runs fine without the kill-switch
and is **bit-identical** to running with it (verified, MD5-matched). Dropped from all
launch commands, `apple_silicon.md`, the `mps-port` skill, and the `mps-bench` harness.
Code: `mmdit/layers.py` (CUDA-gated timestep_embedding compile), `utils/sampling.py`
(`compile_mmdit_blocks`, config-gated), `configs/…/256px.py` (`compile_mmdit=False`).

**Original plan (for reference):**

**What:** `TORCHDYNAMO_DISABLE=1` is currently load-bearing because the
`@torch.compile(max-autotune)` on `timestep_embedding` stalls on MPS. torch 2.13
Inductor has a Metal backend; try selective compilation (the MMDiT forward, or
just its hot blocks) with sane options instead of a global kill-switch.

**Why:** the only remaining single-digit-× speed lever that doesn't require new
hardware. But calibrated expectations: the hot path is SDPA + GEMMs that already
run on Metal kernels — compile mostly wins on fusion of the surrounding
elementwise work, so think −10–25%, not −2×.

- **Effort:** medium. **Risk:** medium-high — compile stalls, silent numeric
  drift, long first-run compile times confounding thermal measurement.
- **Gate:** parity test with the compiled module (compiled-MPS vs eager-MPS vs
  CPU oracle); seed-matched pixels; thermally controlled A/B via `mps-bench`
  (compile time excluded from the timed region — warm renders only).
- **Timebox it:** if it isn't cleanly winning within a day, write down the
  failure mode and stop.

## P6 (stretch) — 768px images and a flux-t2i2v probe

**What:** two bounded probes, not commitments.

- **768px, `--num_frames 1`** ✅ **DONE (2026-07-12) — works.** Note the `768px.py`
  config inherits `plugins/sp.py` (sequence-parallel = distributed, out of scope), so
  drive 768px through the ws=1 **256px config** with `--resolution 768px` instead.
  Result: fallback-unset smoke passed (**no kernel gaps at the new shapes** — 768px
  runs on native Metal), **237 s** for a 768px still at 20 steps (~4 min, matching the
  "low minutes" guess), **peak swap <1 GB** (num*frames 1 keeps activations small,
  so memory is a non-issue for stills). Pixels coherent (a red barn at golden hour —
  a touch soft, since the 256px-trained checkpoint is stretching resolution, but
  correct subject/composition). \*\*768px \_video* stays out of scope\*\* (the temporal
  dim would reintroduce the memory wall). Command:
  `osora-mps --prompt "..." --resolution 768px --num_frames 1 --num_steps 20`.
- **flux t2i2v at 256px** ⬜ not yet probed: `scripts/diffusion/inference.py` already ping-pongs
  the flux and video models between CPU and device under `--offload_model True`; flux
  adds ~24 GB moving through unified memory. Probe only after P4 lands (it has).
  It's an entirely untested model on MPS — needs its own parity gate before its
  output is trusted; the heavy, lower-value half of P6.
- **flux t2i2v at 256px**: `scripts/diffusion/inference.py` already ping-pongs
  the flux and video models between CPU and device under `--offload_model True`; flux
  adds ~24 GB moving through unified memory. Probe only after P4 lands.

- **Effort:** low to probe, high to make pleasant. **Risk:** memory (flux),
  patience (768px).
- **Gate:** fallback-unset smoke (both exercise new shapes/modules — flux is an
  entirely untested model on MPS and needs its own parity test before trusting
  output); memory watch; pixels.

---

## Explicit non-goals (unchanged)

Training, the full 129-frame video, 768px video, and any SP/TP/distributed path
remain rented-CUDA territory. The lane stays `world_size=1`, plain `python`,
launched per the commands in [apple_silicon.md](apple_silicon.md).
