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


def empty_cache() -> None:
    """Release the caching allocator's unused blocks back to the OS.

    On MPS this is what actually returns unified memory after tensors are
    freed/moved off-device — dropping the last reference only hands the
    buffers back to the MPS caching allocator, which keeps them wired until
    ``torch.mps.empty_cache()``. No-op on CPU (and on a forced-CPU oracle run).
    """
    if DEVICE == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif DEVICE == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
