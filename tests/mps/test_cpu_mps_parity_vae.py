"""CPU (oracle) <-> MPS parity for the Hunyuan video VAE on Apple Silicon.

MPS failures are usually *silent wrong numbers*, not crashes -- "it ran" proves
nothing. CPU is the oracle: build one seeded VAE, run it on CPU, move the same
weights to MPS, run the same seeded inputs, compare within a per-dtype tolerance.

This is the P1 correctness gate for image-to-video (`i2v_head`): the reference
image now flows through `AutoencoderKLCausal3D.encode` on native Metal for the
first time (opensora/utils/inference.py:248). Encode was previously never
parity-checked; decode was trusted on rendered pixels alone. A silently-wrong
encode -> mushy / drifting first frame.

Covered here (direct, non-tiled path -- tiling is CPU-side blending of the same
encoder/decoder tiles, no new Metal kernels):
  - EncoderCausal3D  (causal 3D convs w/ replicate pad, resnets, GroupNorm,
    SDPA mid-block attention, quant_conv)
  - DecoderCausal3D  (post_quant_conv, mid-block attention, nearest-interp
    causal upsample, resnets, conv_out)
  - encode -> decode round-trip

`sample_posterior=False` (posterior mode / mean) is used for the encode + the
round-trip so the comparison is deterministic -- `DiagonalGaussianDistribution.
sample()` draws from an RNG stream that differs per device and would break
parity for reasons unrelated to op correctness.

Tolerance (fp32): atol/rtol 1e-4 -- only GEMM/conv accumulation-order
reassociation differs between the CPU and Metal backends. Also asserts finite
output (the fp16/bf16 "confident garbage" check).

Run:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python -m pytest -s tests/mps/test_cpu_mps_parity_vae.py
"""
import pytest
import torch

from opensora.models.hunyuan_vae.autoencoder_kl_causal_3d import (
    AutoEncoder3DConfig,
    AutoencoderKLCausal3D,
)

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available (non-Apple-Silicon)"
)

# Small (B, C, T, H, W). T=9 -> latent T = (9-1)//4 + 1 = 3 latent frames, which
# exercises the causal-in-time 3D convs across multiple latent timesteps.
# H=W=16 -> 3 spatial downsamples (/8) -> 2 latent px. Tiny but architecturally
# complete: 4 down/up stages, time + spatial compression, mid-block attention.
B, C_IN, T, H, W = 1, 3, 9, 16, 16
LATENT_C = 4


def _tiny_config() -> AutoEncoder3DConfig:
    # norm_num_groups=4 divides every block_out_channels entry (8); keep the
    # 4-stage block layout so the spatial(/8)+time(/4) compression logic in
    # EncoderCausal3D/DecoderCausal3D is fully exercised.
    return AutoEncoder3DConfig(
        from_pretrained=None,
        in_channels=C_IN,
        out_channels=C_IN,
        latent_channels=LATENT_C,
        layers_per_block=1,
        norm_num_groups=4,
        block_out_channels=(8, 8, 8, 8),
        time_compression_ratio=4,
        spatial_compression_ratio=8,
        mid_block_add_attention=True,
        # keep the direct (non-tiled) path -- the oracle we care about here
        use_slicing=False,
        use_spatial_tiling=False,
        use_temporal_tiling=False,
    )


def _make_model() -> AutoencoderKLCausal3D:
    torch.manual_seed(0)
    return AutoencoderKLCausal3D(_tiny_config()).eval().float()


def _pixels(device: str) -> torch.Tensor:
    g = torch.Generator().manual_seed(1234)
    x = torch.randn(B, C_IN, T, H, W, generator=g)
    return x.to(device)


def _latent(model: AutoencoderKLCausal3D, device: str) -> torch.Tensor:
    lat_t, lat_h, lat_w = model.get_latent_size([T, H, W])
    g = torch.Generator().manual_seed(4321)
    z = torch.randn(B, LATENT_C, lat_t, lat_h, lat_w, generator=g)
    return z.to(device)


def test_hunyuan_vae_encode_cpu_mps_parity_fp32():
    model = _make_model()
    with torch.no_grad():
        z_cpu = model.encode(_pixels("cpu"), sample_posterior=False).float().cpu()
        model = model.to("mps")
        z_mps = model.encode(_pixels("mps"), sample_posterior=False).float().cpu()

    assert torch.isfinite(z_cpu).all(), "CPU encode has non-finite values"
    assert torch.isfinite(z_mps).all(), "MPS encode has non-finite values (confident garbage)"
    assert z_cpu.shape == z_mps.shape
    max_abs = (z_mps - z_cpu).abs().max().item()
    print(f"\n[hunyuan_vae encode] shape={tuple(z_cpu.shape)} max_abs_delta={max_abs:.3e}")
    torch.testing.assert_close(z_mps, z_cpu, atol=1e-4, rtol=1e-4)


def test_hunyuan_vae_decode_cpu_mps_parity_fp32():
    model = _make_model()
    with torch.no_grad():
        dec_cpu = model.decode(_latent(model, "cpu")).float().cpu()
        model = model.to("mps")
        dec_mps = model.decode(_latent(model, "mps")).float().cpu()

    assert torch.isfinite(dec_cpu).all(), "CPU decode has non-finite values"
    assert torch.isfinite(dec_mps).all(), "MPS decode has non-finite values (confident garbage)"
    assert dec_cpu.shape == dec_mps.shape
    max_abs = (dec_mps - dec_cpu).abs().max().item()
    print(f"\n[hunyuan_vae decode] shape={tuple(dec_cpu.shape)} max_abs_delta={max_abs:.3e}")
    torch.testing.assert_close(dec_mps, dec_cpu, atol=1e-4, rtol=1e-4)


def test_hunyuan_vae_roundtrip_cpu_mps_parity_fp32():
    """encode -> decode on each device; the full path an i2v ref image takes."""
    model = _make_model()
    with torch.no_grad():
        x_cpu = _pixels("cpu")
        rt_cpu = model.decode(model.encode(x_cpu, sample_posterior=False)).float().cpu()
        model = model.to("mps")
        x_mps = _pixels("mps")
        rt_mps = model.decode(model.encode(x_mps, sample_posterior=False)).float().cpu()

    assert torch.isfinite(rt_cpu).all(), "CPU round-trip has non-finite values"
    assert torch.isfinite(rt_mps).all(), "MPS round-trip has non-finite values (confident garbage)"
    assert rt_cpu.shape == rt_mps.shape
    max_abs = (rt_mps - rt_cpu).abs().max().item()
    print(f"\n[hunyuan_vae roundtrip] shape={tuple(rt_cpu.shape)} max_abs_delta={max_abs:.3e}")
    torch.testing.assert_close(rt_mps, rt_cpu, atol=1e-4, rtol=1e-4)
