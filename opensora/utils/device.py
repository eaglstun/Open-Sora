"""Device resolution for the single-device (incl. Apple Silicon / MPS) lane.

CUDA-first code hardcodes ``"cuda"`` / ``.cuda()`` everywhere; on a Mac that
either crashes or silently lands on CPU. This is the one chokepoint: import
``DEVICE`` (or call ``get_device()``) instead of writing ``"cuda"``.

Priority: ``OPENSORA_DEVICE`` env override -> MPS -> CUDA -> CPU.
Set ``OPENSORA_DEVICE=cpu`` to run a CPU oracle on the same machine (used by the
CPU<->MPS parity check).
"""

import os

import torch

_VALID = ("mps", "cuda", "cpu")


def get_device() -> str:
    forced = os.environ.get("OPENSORA_DEVICE")
    if forced:
        forced = forced.lower()
        assert forced in _VALID, f"OPENSORA_DEVICE must be one of {_VALID}, got {forced!r}"
        return forced
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = get_device()


def is_cuda() -> bool:
    return DEVICE == "cuda"


def is_mps() -> bool:
    return DEVICE == "mps"
