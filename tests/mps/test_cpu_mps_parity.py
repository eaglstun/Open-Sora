"""CPU (oracle) <-> MPS forward-parity for the MMDiT transformer on Apple Silicon.

MPS failures are usually *silent wrong numbers*, not crashes -- "it ran" proves
nothing. CPU is the oracle: build one seeded model, run it on CPU, move the same
weights to MPS, run the same seeded inputs, compare within a per-dtype tolerance.

This exercises the Apple Silicon port's rewritten hot paths:
  - the torch-SDPA fallback for flash_attn (opensora/models/mmdit/math.py)
  - the float32 rope path (float64 is unsupported on MPS)

Tolerance (fp32): atol/rtol 1e-4 -- only GEMM accumulation-order reassociation
differs between the CPU and Metal backends. Also asserts finite output (the
fp16/bf16 "confident garbage" check).

Run:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python -m pytest -s tests/mps/test_cpu_mps_parity.py
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


@pytest.mark.parametrize(
    "use_liger_rope",
    [
        # apply_rope (interleaved) path
        False,
        # liger path: pe is a (cos, sin) tuple -> liger_rope_torch fallback on
        # MPS. This is the path the real 256px checkpoint uses (use_liger_rope=True).
        True,
    ],
    ids=["apply_rope", "liger_rope"],
)
def test_mmdit_forward_cpu_mps_parity_fp32(use_liger_rope):
    torch.manual_seed(0)
    cfg = _tiny_config(use_liger_rope=use_liger_rope)
    model = MMDiTModel(cfg).eval().float()

    with torch.no_grad():
        out_cpu = model(**_make_inputs(cfg, "cpu")).float().cpu()
        model = model.to("mps")
        out_mps = model(**_make_inputs(cfg, "mps")).float().cpu()

    assert torch.isfinite(out_cpu).all(), "CPU output has non-finite values"
    assert torch.isfinite(out_mps).all(), "MPS output has non-finite values (confident garbage)"
    assert out_cpu.shape == out_mps.shape
    torch.testing.assert_close(out_mps, out_cpu, atol=1e-4, rtol=1e-4)
