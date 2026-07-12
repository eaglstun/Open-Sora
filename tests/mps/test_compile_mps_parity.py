"""torch.compile parity for the MMDiT transformer blocks on Apple Silicon (P5).

The P5 lane adds an opt-in ``compile_mmdit`` flag that ``torch.compile``s the
Double/SingleStreamBlocks on MPS (via ``compile_mmdit_blocks`` in
``opensora/utils/sampling.py``). It is OFF by default because it is a measured
~-12% speed regression -- but ``torch.compile`` is a newly-exercised code path,
and on MPS a wrong number is silent. So we gate it the same way we gate every
other hot path: build one seeded model, run it EAGER on MPS, compile the same
weights, run the SAME seeded inputs, and compare max-abs delta at fp32.

This mirrors ``tests/mps/test_cpu_mps_parity.py`` (tiny random-weight MMDiT, no
checkpoint, seconds-fast). Two comparisons:
  - compiled-MPS vs eager-MPS  (the P5 question: is Inductor's Metal lossy?)
  - compiled-MPS vs CPU oracle (belt-and-suspenders vs the ground truth)

Tolerance (fp32): atol/rtol 1e-4 -- compile reorders/fuses ops, but at fp32 the
only legitimate divergence is GEMM accumulation-order reassociation, same budget
as the CPU<->MPS test. A drift >1e-4 here is a real finding (the compiled path
is genuinely lossy), NOT a reason to loosen the tolerance.

NOTE: the ``compile_mmdit`` flag is a silent no-op if TORCHDYNAMO_DISABLE is set
in the environment; run this WITHOUT that var or the test proves nothing (it
would compare eager-vs-eager). We assert the compile actually happened below.

Run (torch 2.13 venv, where compile matters most):
    /Users/eeaglstun/venvs/opensora-torch213/bin/python -m pytest -s \
        tests/mps/test_compile_mps_parity.py
"""
import pytest
import torch

from opensora.models.mmdit.model import MMDiTConfig, MMDiTModel

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available (non-Apple-Silicon)"
)

B, L_IMG, L_TXT = 2, 32, 8


def _tiny_config(use_liger_rope: bool = False) -> MMDiTConfig:
    # hidden=64, heads=4 -> pe_dim=16; axes_dim even and summing to 16.
    return MMDiTConfig(
        from_pretrained=None, cache_dir=None,
        in_channels=8, vec_in_dim=16, context_in_dim=16,
        hidden_size=64, mlp_ratio=2.0, num_heads=4,
        depth=2, depth_single_blocks=2,
        axes_dim=[4, 4, 8], theta=10000, qkv_bias=True,
        guidance_embed=False, cond_embed=False, fused_qkv=True,
        grad_ckpt_settings=None, use_liger_rope=use_liger_rope, patch_size=2,
    )


def _make_inputs(cfg: MMDiTConfig, device: str) -> dict:
    g = torch.Generator().manual_seed(1234)
    img = torch.randn(B, L_IMG, cfg.in_channels, generator=g)
    txt = torch.randn(B, L_TXT, cfg.context_in_dim, generator=g)
    y_vec = torch.randn(B, cfg.vec_in_dim, generator=g)
    timesteps = torch.rand(B, generator=g)
    img_ids = torch.zeros(B, L_IMG, 3)
    img_ids[..., 0] = torch.arange(L_IMG).float()
    txt_ids = torch.zeros(B, L_TXT, 3)
    to = lambda t: t.to(device)
    return dict(img=to(img), img_ids=to(img_ids), txt=to(txt), txt_ids=to(txt_ids),
                timesteps=to(timesteps), y_vec=to(y_vec))


def _compile_blocks(model: MMDiTModel) -> None:
    """Mirror of opensora/utils/sampling.py::compile_mmdit_blocks -- inlined so
    this test does not drag in sampling.py's heavy import chain (peft, HFEmbedder,
    datasets). Default Inductor mode, in place, per block."""
    for block in list(model.double_blocks) + list(model.single_blocks):
        block.compile()


@pytest.mark.parametrize(
    "use_liger_rope",
    [
        False,  # apply_rope (interleaved) path
        True,   # liger_rope_torch fallback -- the path the real 256px ckpt uses
    ],
    ids=["apply_rope", "liger_rope"],
)
def test_mmdit_compile_mps_parity_fp32(use_liger_rope):
    # Guard: TORCHDYNAMO_DISABLE would make .compile() a silent no-op, turning
    # this into an eager-vs-eager tautology. Fail loudly instead of pretending.
    import os
    assert not os.environ.get("TORCHDYNAMO_DISABLE"), (
        "TORCHDYNAMO_DISABLE is set -- torch.compile is a no-op, this test would "
        "compare eager-vs-eager. Unset it to actually exercise the compiled path."
    )

    torch.manual_seed(0)
    cfg = _tiny_config(use_liger_rope=use_liger_rope)
    model = MMDiTModel(cfg).eval().float()

    with torch.no_grad():
        # CPU oracle (ground truth) with these exact seeded weights.
        out_cpu = model(**_make_inputs(cfg, "cpu")).float().cpu()

        # Eager MPS -- baseline the compiled path must match.
        model = model.to("mps")
        out_mps_eager = model(**_make_inputs(cfg, "mps")).float().cpu()

        # Compile the SAME weights in place, run the SAME inputs.
        _compile_blocks(model)
        out_mps_compiled = model(**_make_inputs(cfg, "mps")).float().cpu()

    # Non-finite on MPS is a hard stop (confident garbage).
    assert torch.isfinite(out_cpu).all(), "CPU output has non-finite values"
    assert torch.isfinite(out_mps_eager).all(), "eager-MPS output has non-finite values"
    assert torch.isfinite(out_mps_compiled).all(), "compiled-MPS output has non-finite values"
    assert out_mps_compiled.shape == out_mps_eager.shape == out_cpu.shape

    # Primary comparison: compiled-MPS vs eager-MPS (the P5 question).
    d_ce = (out_mps_compiled - out_mps_eager).abs()
    r_ce = d_ce / out_mps_eager.abs().clamp_min(1e-12)
    print(
        f"\n[compile_mps_parity/{'liger_rope' if use_liger_rope else 'apply_rope'}] "
        f"compiled-vs-eager  max|abs|={d_ce.max().item():.3e}  max|rel|={r_ce.max().item():.3e}"
    )
    # Secondary: compiled-MPS vs CPU oracle.
    d_cc = (out_mps_compiled - out_cpu).abs()
    print(
        f"[compile_mps_parity/{'liger_rope' if use_liger_rope else 'apply_rope'}] "
        f"compiled-vs-cpu    max|abs|={d_cc.max().item():.3e}"
    )

    torch.testing.assert_close(out_mps_compiled, out_mps_eager, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(out_mps_compiled, out_cpu, atol=1e-4, rtol=1e-4)
