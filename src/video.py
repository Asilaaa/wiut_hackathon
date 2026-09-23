"""Video reading with frame skipping and downscaling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np

from .config import REF_H, REF_W


@dataclass
class VideoInfo:
    path: str
    fps: float
    n_frames: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0


def probe(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    info = VideoInfo(
        path=path,
        fps=cap.get(cv2.CAP_PROP_FPS) or 25.0,
        n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return info


def to_ref(frame: np.ndarray) -> np.ndarray:
    """Resize a frame to the reference resolution used for all geometry."""
    if frame.shape[1] == REF_W and frame.shape[0] == REF_H:
        return frame
    return cv2.resize(frame, (REF_W, REF_H), interpolation=cv2.INTER_AREA)


def iter_frames(path: str, stride: int = 1) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_idx, t_sec, frame_in_ref_resolution) for every `stride`-th frame.

    Skipped frames are only grabbed (not converted), which is ~30% cheaper
    than reading every frame of a 4K stream.
    """
    info = probe(path)
    cap = cv2.VideoCapture(path)
    idx = 0
    try:
        while True:
            if idx % stride == 0:
                ok, frame = cap.read()
                if not ok:
                    break
                yield idx, idx / info.fps, to_ref(frame)
            elif not cap.grab():
                break
            idx += 1
    finally:
        cap.release()
