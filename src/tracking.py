"""Road-user detection (YOLO) + multi-object tracking (ByteTrack)."""
from __future__ import annotations

import csv
from dataclasses import astuple, dataclass, fields
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from ultralytics import YOLO

from . import config


@dataclass
class Detection:
    frame: int
    t: float
    track_id: int
    type: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float


def _device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


class Tracker:
    """Stateful detector + tracker. Feed frames in temporal order, one video per instance."""

    def __init__(self, weights: Path = config.DETECTOR_WEIGHTS, imgsz: int = config.DETECT_IMGSZ):
        torch.manual_seed(config.SEED)
        self.model = YOLO(str(weights))
        self.imgsz = imgsz
        self.device = _device()
        self.classes = list(config.COCO_TO_TYPE)

    def update(self, frame: np.ndarray, frame_idx: int, t: float) -> list[Detection]:
        """Detect and track road users in one frame (already at reference resolution)."""
        res = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            imgsz=self.imgsz,
            conf=config.DETECT_CONF,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )[0]
        boxes = res.boxes
        if boxes.id is None:
            return []
        out = []
        for tid, cls, conf, (x1, y1, x2, y2) in zip(
            boxes.id.int().tolist(), boxes.cls.int().tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()
        ):
            out.append(Detection(frame_idx, round(t, 3), tid, config.COCO_TO_TYPE[cls], round(conf, 3),
                                 round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)))
        return out


def save_tracks(dets: Iterable[Detection], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([fl.name for fl in fields(Detection)])
        w.writerows(astuple(d) for d in dets)


def load_tracks(path: Path) -> list[Detection]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [
        Detection(int(r["frame"]), float(r["t"]), int(r["track_id"]), r["type"], float(r["conf"]),
                  float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]))
        for r in rows
    ]
