"""Self-contained pyav video I/O for the datasets layer.

torchvision removed its `torchvision.io.read_video` / `write_video` API in 0.27
(and shipped a broken fbcode re-export stub in 0.28), so the data layer can no
longer lean on it. This module provides the small subset Open-Sora actually uses,
backed directly by pyav (`av`) — decoupling the pipeline from the torchvision
video API, which is a prerequisite for running on torch>2.10 (whose torchvision
pair lacks that API).

`write_video` mirrors the working torchvision<=0.25 implementation (video-only;
Open-Sora never writes an audio track). Reads are already pyav-native in
``read_video.py::read_video_av`` — that function is the canonical reader; use it
directly instead of a wrapper here.
"""

import av
import numpy as np
import torch


def _check_av_available() -> None:
    """No-op availability check, kept so call sites read as before.

    ``av`` is imported at module load; if it were missing this module would fail
    to import, so reaching here means pyav is present.
    """
    if not hasattr(av, "open"):
        raise ImportError("pyav (`av`) is required for video I/O but is unavailable.")


def write_video(
    filename: str,
    video_array,
    fps,
    video_codec: str = "libx264",
    options: dict | None = None,
) -> None:
    """Encode a uint8 ``(T, H, W, C)`` RGB tensor/array to a video file via pyav.

    Drop-in for the subset of ``torchvision.io.write_video`` Open-Sora relies on.
    ``fps`` may be int/float; pyav rejects floating-point rates, so floats are
    rounded. ``options`` is passed through to the codec (e.g. ``{"crf": "17"}``).
    """
    _check_av_available()
    video_array = torch.as_tensor(video_array, dtype=torch.uint8).numpy(force=True)

    # PyAV rejects fractional frame rates.
    if isinstance(fps, float):
        fps = int(np.round(fps))

    with av.open(filename, mode="w") as container:
        stream = container.add_stream(video_codec, rate=fps)
        stream.width = video_array.shape[2]
        stream.height = video_array.shape[1]
        stream.pix_fmt = "yuv420p" if video_codec != "libx264rgb" else "rgb24"
        stream.options = options or {}

        for img in video_array:
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            # Leave pict_type unset: it defaults to NONE (encoder decides). Newer
            # pyav (av>=15) rejects the legacy string form `frame.pict_type = "NONE"`.
            for packet in stream.encode(frame):
                container.mux(packet)

        # Flush the encoder.
        for packet in stream.encode():
            container.mux(packet)
