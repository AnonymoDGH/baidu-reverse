"""Rotation-angle predictor for Baidu's "spin" captcha (拖动左侧滑块使图片为正).

The captcha shows a circular picture that has been rotated by an unknown angle;
the user has to drag a slider until the picture is upright. The page rotates
the ``<img>`` with ``transform: rotate(start + 360 * percent deg)`` (clockwise)
and only the ``percent`` (plus a mouse trace) is sent to the server.

So solving it = predicting how far the *currently displayed* picture is rotated
counter-clockwise (``d``) and dragging the slider to ``percent = d / 360``.

The predictor is a RotNet-style classifier (RegNet-Y-3.2GF backbone, 128 angle
bins) from https://github.com/lumina37/rotate-captcha-crack (MIT). Weights are
downloaded once into ``~/.cache/baidu_reverse`` (or ``$BAIDU_REVERSE_HOME``).

Optional dependencies (``pip install baidu-reverse[captcha]``): torch, torchvision,
pillow, numpy, py7zr.
"""

from __future__ import annotations

import io
import math
import shutil
import urllib.request
from pathlib import Path
from typing import Optional

from ..paths import cache_dir

CLS_NUM = 128
TARGET_SIZE = 224
MODEL_URL = (
    "https://github.com/lumina37/rotate-captcha-crack/releases/download/"
    "v0.5.1/RotNetR.7z"
)
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def _download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "baidu-reverse"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as fh:
        shutil.copyfileobj(resp, fh, 1 << 20)


def ensure_weights(path: Optional[str] = None) -> Path:
    """Return a path to ``best.pth``; download + extract it on first use."""
    if path:
        return Path(path)
    root = cache_dir()
    existing = sorted(root.glob("RotNetR/*/best.pth"))
    if existing:
        return existing[-1]
    archive = root / "RotNetR.7z"
    if not archive.exists() or archive.stat().st_size < 1_000_000:
        _download(MODEL_URL, archive)
    try:
        import py7zr  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "py7zr is required to unpack the model: pip install py7zr "
            f"(or extract {archive} manually into {root})"
        ) from exc
    with py7zr.SevenZipFile(archive) as zf:
        zf.extractall(root)
    existing = sorted(root.glob("RotNetR/*/best.pth"))
    if not existing:
        raise RuntimeError(f"best.pth not found after extracting {archive}")
    return existing[-1]


class AnglePredictor:
    """Predict the counter-clockwise rotation (degrees, [0, 360)) of a captcha image."""

    def __init__(self, weights: Optional[str] = None, device: str = "cpu"):
        import torch
        from torch import nn
        from torchvision import models

        self._torch = torch
        self.device = torch.device(device)
        backbone = models.regnet_y_3_2gf(weights=None)
        backbone.fc = nn.Linear(backbone.fc.in_features, CLS_NUM)

        class _Net(nn.Module):
            def __init__(self, bb):
                super().__init__()
                self.backbone = bb

            def forward(self, x):
                return self.backbone(x)

        self.model = _Net(backbone)
        state = torch.load(ensure_weights(weights), map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

    # -- preprocessing ------------------------------------------------------
    def _prep(self, img, extra_ccw: float = 0.0):
        """Crop the circle's inscribed square -> 224x224 normalized tensor.

        ``extra_ccw`` rotates the picture counter-clockwise first (for TTA).
        """
        import numpy as np
        from PIL import Image

        torch = self._torch
        img = img.convert("RGB")
        w, h = img.size
        side = min(w, h)
        if extra_ccw:
            img = img.rotate(extra_ccw, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
        # strip the border: keep the inscribed square of the circle
        crop = int(side / math.sqrt(2))
        left, top = (w - crop) // 2, (h - crop) // 2
        img = img.crop((left, top, left + crop, top + crop)).resize(
            (TARGET_SIZE, TARGET_SIZE), Image.Resampling.BILINEAR
        )
        arr = np.asarray(img, dtype="float32") / 255.0
        arr = (arr - np.array(_MEAN, dtype="float32")) / np.array(_STD, dtype="float32")
        return torch.from_numpy(arr.transpose(2, 0, 1)).contiguous()

    # -- inference ----------------------------------------------------------
    def probs(self, img, tta: bool = True):
        """Angle distribution over CLS_NUM bins (counter-clockwise, bin = 360/CLS_NUM deg)."""
        import numpy as np

        torch = self._torch
        shifts = (0, 90, 180, 270) if tta else (0,)
        batch = torch.stack([self._prep(img, s) for s in shifts]).to(self.device)
        with torch.no_grad():
            p = torch.softmax(self.model(batch), dim=1).cpu().numpy()
        acc = np.zeros(CLS_NUM, dtype="float64")
        step = CLS_NUM // 4
        for i, s in enumerate(shifts):
            # the TTA copy was pre-rotated CCW by ``s`` so its predicted angle is
            # (true + s); shift the distribution back by s degrees.
            acc += np.roll(p[i], -(s * CLS_NUM // 360))
        return acc / len(shifts)

    def predict(self, img, tta: bool = True) -> tuple[float, float]:
        """Return ``(angle_ccw_deg, confidence)``.

        Confidence is the probability mass within +-5 bins of the argmax.
        """
        import numpy as np

        pr = self.probs(img, tta)
        k = int(pr.argmax())
        idx = [(k + o) % CLS_NUM for o in range(-5, 6)]
        # circular weighted mean around the peak for sub-bin precision
        bins = np.array([o for o in range(-5, 6)], dtype="float64")
        w = pr[idx]
        center = k + float((bins * w).sum() / max(w.sum(), 1e-9))
        angle = (center * 360.0 / CLS_NUM) % 360.0
        return angle, float(w.sum())

    def predict_bytes(self, data: bytes, tta: bool = True) -> tuple[float, float]:
        from PIL import Image

        return self.predict(Image.open(io.BytesIO(data)), tta)
