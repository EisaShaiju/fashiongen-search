"""
structural_fidelity.py
======================
Structural Preservation Fidelity (SPF) for garment generation.

When we fine-tune SDXL + ControlNet to regenerate/edit a garment while keeping
its *structure* (silhouette, seams, collar/sleeve geometry, print layout), we
need a number that says "how faithfully was the original structure preserved?"

SPF combines three complementary structural signals between a reference garment
image and the generated one (both assumed roughly aligned / same canvas):

    1. Edge agreement   - Canny edge maps compared with a tolerant (dilated)
                          F1 score. Captures seams, hems, collar/sleeve lines,
                          print boundaries.
    2. Silhouette IoU   - foreground garment mask IoU. Captures the overall
                          shape / cut being preserved.
    3. Structural SSIM  - multiscale-ish SSIM on luminance. Captures local
                          contrast/texture structure independent of colour.

SPF = weighted geometric mean of the three in [0, 1], reported as a percentage.
A geometric mean is used so that failing *any* one structural aspect pulls the
score down (you can't hide a broken silhouette behind good edges).

This module is dependency-light (numpy, OpenCV, scikit-image) and fully
runnable on CPU - it is the metric behind the "structural preservation
fidelity" headline and is used both in evaluation and as an MLflow-logged
validation metric during training.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    return img.astype(np.uint8) if img.dtype != np.uint8 else img


def _canny(gray: np.ndarray, low=100, high=200) -> np.ndarray:
    return cv2.Canny(gray, low, high) > 0


def _garment_mask(gray: np.ndarray) -> np.ndarray:
    """Foreground garment mask via Otsu + largest-component clean-up.

    Product/garment shots have (near) uniform backgrounds, so Otsu on the
    inverted luminance isolates the garment well. Robust enough for a metric.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # background is usually brighter than the garment; try both polarities and
    # keep whichever yields the more "central" mask.
    _, m1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, m2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    def score(m):  # prefer mask whose centroid is near image centre & sane area
        a = (m > 0).mean()
        if a < 0.02 or a > 0.98:
            return -1
        ys, xs = np.where(m > 0)
        cy, cx = ys.mean() / m.shape[0], xs.mean() / m.shape[1]
        return 1 - (abs(cy - 0.5) + abs(cx - 0.5))
    mask = m1 if score(m1) >= score(m2) else m2
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return mask > 0


def _dilate(b: np.ndarray, k: int = 3) -> np.ndarray:
    return cv2.dilate(b.astype(np.uint8), np.ones((k, k), np.uint8)) > 0


# --------------------------------------------------------------------------- #
# individual components
# --------------------------------------------------------------------------- #
def edge_f1(ref_gray, gen_gray, tol: int = 3) -> float:
    """Tolerant edge F1: an edge counts as matched if within `tol` px."""
    er, eg = _canny(ref_gray), _canny(gen_gray)
    er_d, eg_d = _dilate(er, tol), _dilate(eg, tol)
    tp_p = (eg & er_d).sum()                       # gen edges near a ref edge
    precision = tp_p / (eg.sum() + 1e-9)
    tp_r = (er & eg_d).sum()                       # ref edges near a gen edge
    recall = tp_r / (er.sum() + 1e-9)
    return float(2 * precision * recall / (precision + recall + 1e-9))


def silhouette_iou(ref_gray, gen_gray) -> float:
    mr, mg = _garment_mask(ref_gray), _garment_mask(gen_gray)
    inter = (mr & mg).sum()
    union = (mr | mg).sum()
    return float(inter / (union + 1e-9))


def structural_ssim(ref_gray, gen_gray) -> float:
    return float(ssim(ref_gray, gen_gray, data_range=255))


# --------------------------------------------------------------------------- #
# top-level metric
# --------------------------------------------------------------------------- #
@dataclass
class SPFResult:
    spf: float                 # 0..1 overall
    edge_f1: float
    silhouette_iou: float
    ssim: float

    def as_percent(self) -> float:
        return round(100.0 * self.spf, 2)

    def to_dict(self):
        return {"spf": self.spf, "spf_percent": self.as_percent(),
                "edge_f1": self.edge_f1, "silhouette_iou": self.silhouette_iou,
                "ssim": self.ssim}


def structural_preservation_fidelity(
        reference: np.ndarray,
        generated: np.ndarray,
        weights=(0.45, 0.30, 0.25),
        size: int = 512) -> SPFResult:
    """Compute SPF between a reference and a generated garment image.

    Images are RGB or grayscale uint8 arrays. They are resized to a common
    canvas so the metric is resolution-independent. Returns an SPFResult.
    """
    ref = cv2.resize(reference, (size, size), interpolation=cv2.INTER_AREA)
    gen = cv2.resize(generated, (size, size), interpolation=cv2.INTER_AREA)
    rg, gg = _to_gray(ref), _to_gray(gen)

    e = edge_f1(rg, gg)
    s = silhouette_iou(rg, gg)
    q = max(structural_ssim(rg, gg), 0.0)          # clamp negative SSIM to 0

    we, ws, wq = weights
    # weighted geometric mean (any near-zero component tanks the score)
    eps = 1e-6
    spf = np.exp(we * np.log(e + eps) + ws * np.log(s + eps) + wq * np.log(q + eps))
    spf = float(np.clip(spf, 0.0, 1.0))
    return SPFResult(spf=spf, edge_f1=e, silhouette_iou=s, ssim=q)


def batch_spf(refs, gens, **kw):
    """Mean SPF over aligned lists of reference/generated images."""
    results = [structural_preservation_fidelity(r, g, **kw)
               for r, g in zip(refs, gens)]
    mean_spf = float(np.mean([r.spf for r in results]))
    return mean_spf, results
