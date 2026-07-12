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

| frames | latent | render | memory                                              | ~video @24fps | coherent |
| ------ | ------ | ------ | --------------------------------------------------- | ------------- | -------- |
| 13     | 4      | 137 s  | comfortable (no-offload)                            | ~0.5 s        | ✅       |
| 29     | 8      | 237 s  | no-offload but **~10 GB swap, U-state** — the cliff | ~1.2 s        | ✅       |
| 49     | 13     | 399 s  | **requires `--offload True`** (free→0 even so)      | ~2.0 s        | ✅       |

**Finding:** render time scales **~linearly** with latent frames (≈30–34 s each), not
quadratically — attention is _not_ the bottleneck (linear/MLP layers dominate), so the
wall is **memory, not compute**. **Verdict: 29 frames is the practical no-offload
ceiling** on 64 GB (~4 min, rides the swap edge); **49 frames is the practical max**
with `--offload True` (~6.7 min, at free→0). Beyond ~49f toward 129f is
memory-prohibitive. So the lane is for **short clips, ~0.5–2 s**: 29f no-offload for
quick iteration, 49f+offload when you need the length. All numbers cold; add ~+28% warm.
_Caveats: the 49f timing carries offload + some residual-swap overhead (treat as an
upper estimate); baselines recorded via `mps-bench`._

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

## P4 — Memory lever: release T5 after encode

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

## P5 — torch.compile / Inductor-on-MPS experiment

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

- **768px, `--num_frames 1`**: ~9× the tokens of a 256px image (~81× attention),
  but a 256px image denoises in ~16 s — so a 768px still image may land in
  low minutes. Video at 768px stays out of scope.
- **flux t2i2v at 256px**: `scripts/diffusion/inference.py` already ping-pongs
  the flux and video models between CPU and device under `--offload True`; flux
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
