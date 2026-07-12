"""CPU (oracle) <-> MPS parity for the flux 2D image VAE (`autoencoder_2d`).

MPS failures are usually *silent wrong numbers*, not crashes -- "it ran" proves
nothing. CPU is the oracle: build one seeded VAE, run it on CPU, move the same
weights to MPS, run the same seeded inputs, compare within a per-dtype tolerance.

This is the correctness gate for the **t2i2v** pipeline's image stage: the flux
T2I latent is decoded to pixels through `AutoEncoder.decode`
(opensora/models/vae/autoencoder_2d.py, registered `type="autoencoder_2d"`, wired
as `img_flux_ae` in configs/diffusion/inference/plugins/t2i2v.py), and that image
is then re-encoded as the i2v reference. The module first ran on Metal in the P6
t2i2v probe; "parity before pixels" -- it gets a permanent gate here.

Unlike the hunyuan VAE (3D, causal-in-time convs), this one is a plain **2D**
image autoencoder: the public encode/decode take (B, C, T, H, W) but immediately
fold T into the batch and run 2D convs, so the flux path uses T=1.

Covered (direct path, no tiling exists on this module):
  - Encoder  (conv_in, ResnetBlock stack, strided Downsample w/ asymmetric
    zero-pad, SDPA mid-block AttnBlock, GroupNorm, conv_out)
  - Decoder  (conv_in, mid-block attention, nearest-interp Upsample, resnets,
    conv_out)
  - encode -> decode round-trip

`sample=False` (posterior mode / mean) so the comparison is deterministic --
`DiagonalGaussianDistribution.sample()` draws from an RNG stream that differs per
device and would break parity for reasons unrelated to op correctness.

Tolerance (fp32): atol/rtol 1e-4 -- only GEMM/conv accumulation-order
reassociation differs between the CPU and Metal backends. Also asserts finite
output (the fp16/bf16 "confident garbage" check).

Run:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python -m pytest -s tests/mps/test_cpu_mps_parity_ae2d.py
"""
import pytest
import torch

from opensora.models.vae.autoencoder_2d import AutoEncoder, AutoEncoderConfig

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available (non-Apple-Silicon)"
)

# (B, C, T, H, W). T=1 is what the flux image stage actually uses.
# H=W=32 with the real ch_mult=[1,2,4,4] -> 3 strided Downsamples (/8) -> 4x4
# latent, so the full downsample stack and the 16-token mid-block attention are
# both exercised. Tiny but architecturally complete.
B, C_IN, T, H, W = 1, 3, 1, 32, 32
Z_CHANNELS = 4
CH = 32  # AttnBlock/ResnetBlock hardcode GroupNorm(num_groups=32) -> every
# channel count (ch * ch_mult[i]) must be a multiple of 32.
CH_MULT = [1, 2, 4, 4]  # the real config's stage layout


def _tiny_config() -> AutoEncoderConfig:
    return AutoEncoderConfig(
        from_pretrained=None,
        cache_dir=None,
        resolution=H,
        in_channels=C_IN,
        ch=CH,
        out_ch=C_IN,
        ch_mult=CH_MULT,
        num_res_blocks=1,
        z_channels=Z_CHANNELS,
        # real flux-ae values -- they scale/shift the latent, so keeping them
        # means the parity check sees the same magnitudes production does.
        scale_factor=0.3611,
        shift_factor=0.1159,
        # deterministic: posterior.mode() instead of posterior.sample()
        sample=False,
    )


def _make_model() -> AutoEncoder:
    torch.manual_seed(0)
    return AutoEncoder(_tiny_config()).eval().float()


def _pixels(device: str) -> torch.Tensor:
    g = torch.Generator().manual_seed(1234)
    x = torch.randn(B, C_IN, T, H, W, generator=g)
    return x.to(device)


def _latent(device: str) -> torch.Tensor:
    # 3 spatial downsamples -> H/8, W/8
    ds = 2 ** (len(CH_MULT) - 1)
    g = torch.Generator().manual_seed(4321)
    z = torch.randn(B, Z_CHANNELS, T, H // ds, W // ds, generator=g)
    return z.to(device)


def test_ae2d_encode_cpu_mps_parity_fp32():
    model = _make_model()
    with torch.no_grad():
        z_cpu = model.encode(_pixels("cpu")).float().cpu()
        model = model.to("mps")
        z_mps = model.encode(_pixels("mps")).float().cpu()

    assert torch.isfinite(z_cpu).all(), "CPU encode has non-finite values"
    assert torch.isfinite(z_mps).all(), "MPS encode has non-finite values (confident garbage)"
    assert z_cpu.shape == z_mps.shape
    max_abs = (z_mps - z_cpu).abs().max().item()
    print(f"\n[ae2d encode] shape={tuple(z_cpu.shape)} max_abs_delta={max_abs:.3e}")
    torch.testing.assert_close(z_mps, z_cpu, atol=1e-4, rtol=1e-4)


def test_ae2d_decode_cpu_mps_parity_fp32():
    model = _make_model()
    with torch.no_grad():
        dec_cpu = model.decode(_latent("cpu")).float().cpu()
        model = model.to("mps")
        dec_mps = model.decode(_latent("mps")).float().cpu()

    assert torch.isfinite(dec_cpu).all(), "CPU decode has non-finite values"
    assert torch.isfinite(dec_mps).all(), "MPS decode has non-finite values (confident garbage)"
    assert dec_cpu.shape == dec_mps.shape
    max_abs = (dec_mps - dec_cpu).abs().max().item()
    print(f"\n[ae2d decode] shape={tuple(dec_cpu.shape)} max_abs_delta={max_abs:.3e}")
    torch.testing.assert_close(dec_mps, dec_cpu, atol=1e-4, rtol=1e-4)


def test_ae2d_roundtrip_cpu_mps_parity_fp32():
    """encode -> decode on each device; the full path a t2i2v image takes."""
    model = _make_model()
    with torch.no_grad():
        rt_cpu = model.decode(model.encode(_pixels("cpu"))).float().cpu()
        model = model.to("mps")
        rt_mps = model.decode(model.encode(_pixels("mps"))).float().cpu()

    assert torch.isfinite(rt_cpu).all(), "CPU round-trip has non-finite values"
    assert torch.isfinite(rt_mps).all(), "MPS round-trip has non-finite values (confident garbage)"
    assert rt_cpu.shape == rt_mps.shape
    max_abs = (rt_mps - rt_cpu).abs().max().item()
    print(f"\n[ae2d roundtrip] shape={tuple(rt_cpu.shape)} max_abs_delta={max_abs:.3e}")
    torch.testing.assert_close(rt_mps, rt_cpu, atol=1e-4, rtol=1e-4)
