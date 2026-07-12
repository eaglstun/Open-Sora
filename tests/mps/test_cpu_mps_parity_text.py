"""CPU (oracle) <-> MPS parity for the T5 + CLIP text encoders on Apple Silicon.

MPS failures are usually *silent wrong numbers*, not crashes -- "it ran" proves
nothing. CPU is the oracle: build one seeded encoder, run it on CPU, move the
same weights to MPS, run the same seeded inputs, compare within a per-dtype
tolerance.

This is the P2 correctness gate for text conditioning. Every denoise step is
steered by these embeddings; a silently-wrong text forward -> the model renders
a coherent-but-wrong video and nothing crashes. The stack was NEVER compared
CPU-vs-MPS before this.

What runs in production (opensora/models/text/conditioner.py::HFEmbedder.forward):
  - T5:   T5EncoderModel(...)["last_hidden_state"]   (config `t5`)
  - CLIP: CLIPTextModel(...)["pooler_output"]         (config `clip`)
both called with attention_mask=None, output_hidden_states=False. The tokenizer
step is device-independent (CPU int64 ids), so the Metal surface under test is
exactly the hf_module forward -- exercised here directly with the same call
shape HFEmbedder uses. Off-CUDA the shardformer path is a no-op
(conditioner.py:27 gates it on torch.cuda.is_available()), so vanilla HF T5/CLIP
IS the MPS production forward -- that is what we compare.

Oracle: tiny random-weight HF configs built offline (no ./ckpts, no 19GB T5-XXL,
no network). Mirrors the tiny-MMDiT / tiny-VAE pattern -- small hidden size, 2
layers, laptop-runnable in seconds. The Metal kernels reached (embeddings, the
T5 relative-position attention + RMSNorm + gated-GeLU FF, the CLIP causal-mask
attention + LayerNorm + GeLU FF + argmax pooling) are the same ops the real
checkpoints hit; only the matmul dimensions shrink.

Tolerance (fp32): atol/rtol 1e-4 -- only GEMM accumulation-order reassociation
differs between the CPU and Metal backends. Also asserts finite output (the
fp16/bf16 "confident garbage" check).

Run:
    PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1 \
        python -m pytest -s tests/mps/test_cpu_mps_parity_text.py
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available (non-Apple-Silicon)"
)

B, L, VOCAB = 2, 16, 128


def _make_ids(device: str) -> torch.Tensor:
    # Seeded on CPU, then moved -- generating on MPS draws a different RNG stream
    # and would silently break the comparison. int64 token ids, device-agnostic.
    g = torch.Generator().manual_seed(1234)
    ids = torch.randint(0, VOCAB, (B, L), generator=g)
    return ids.to(device)


def _t5_encoder():
    from transformers import T5Config, T5EncoderModel

    torch.manual_seed(0)
    cfg = T5Config(
        vocab_size=VOCAB, d_model=64, d_kv=16, d_ff=128,
        num_layers=2, num_heads=4, relative_attention_num_buckets=32,
    )
    return T5EncoderModel(cfg).eval().float()


def _clip_text_model():
    from transformers import CLIPTextConfig, CLIPTextModel

    torch.manual_seed(0)
    cfg = CLIPTextConfig(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=L,
    )
    return CLIPTextModel(cfg).eval().float()


def _forward(model, ids, output_key: str) -> torch.Tensor:
    # Exactly the call HFEmbedder.forward makes on hf_module.
    out = model(input_ids=ids, attention_mask=None, output_hidden_states=False)
    return out[output_key]


def test_t5_encoder_cpu_mps_parity_fp32():
    """T5EncoderModel last_hidden_state -- the `t5` text_embedder forward."""
    model = _t5_encoder()
    with torch.no_grad():
        h_cpu = _forward(model, _make_ids("cpu"), "last_hidden_state").float().cpu()
        model = model.to("mps")
        h_mps = _forward(model, _make_ids("mps"), "last_hidden_state").float().cpu()

    assert torch.isfinite(h_cpu).all(), "CPU T5 output has non-finite values"
    assert torch.isfinite(h_mps).all(), "MPS T5 output has non-finite values (confident garbage)"
    assert h_cpu.shape == h_mps.shape
    max_abs = (h_mps - h_cpu).abs().max().item()
    print(f"\n[t5 encoder] shape={tuple(h_cpu.shape)} max_abs_delta={max_abs:.3e}")
    torch.testing.assert_close(h_mps, h_cpu, atol=1e-4, rtol=1e-4)


def test_clip_text_cpu_mps_parity_fp32():
    """CLIPTextModel pooler_output -- the `clip` text_embedder forward."""
    model = _clip_text_model()
    with torch.no_grad():
        p_cpu = _forward(model, _make_ids("cpu"), "pooler_output").float().cpu()
        model = model.to("mps")
        p_mps = _forward(model, _make_ids("mps"), "pooler_output").float().cpu()

    assert torch.isfinite(p_cpu).all(), "CPU CLIP output has non-finite values"
    assert torch.isfinite(p_mps).all(), "MPS CLIP output has non-finite values (confident garbage)"
    assert p_cpu.shape == p_mps.shape
    max_abs = (p_mps - p_cpu).abs().max().item()
    print(f"\n[clip text] shape={tuple(p_cpu.shape)} max_abs_delta={max_abs:.3e}")
    torch.testing.assert_close(p_mps, p_cpu, atol=1e-4, rtol=1e-4)
