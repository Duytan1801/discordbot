"""Masks -> muscle architecture (PA / FL / MT).

Two roles:
  A. Derive numeric training targets from fasc/apo masks when no label CSV
     exists (probable — verify in EDA).
  B. Phase-B inference path: predicted masks (SAM) -> geometric measurements.

Conventions (image coords, y down):
  - apo mask: superficial (upper) + deep (lower) aponeurosis bands.
  - fasc mask: fascicle fragments (thin oblique structures).
  - PA: angle between dominant fascicle direction and deep apo direction.
  - MT: perpendicular distance between apo midlines, averaged along x.
  - FL: apo-to-apo distance measured ALONG the fascicle direction
        (= MT / sin(PA) for straight parallel geometry).

px -> mm calibration is dataset dependent (device/depth settings). PA needs no
calibration. For FL/MT pass --px-per-mm once EDA reveals the scale convention
(check tif tags, embedded scale bars, or the Ritsche-2024 dataset docs).
If unknown, derive in px and fit a single global scale on validation.

Usage:
    python geometry.py --derive --manifest ../data/manifests/fasc_unique.csv \\
        --apo-manifest ../data/manifests/apo_unique.csv --px-per-mm 4.0 \\
        -o ../data/manifests/labels_from_masks.csv
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import cv2  # opencv-python-headless
except ImportError:
    cv2 = None


def load_mask(path: str) -> np.ndarray:
    return (np.asarray(Image.open(path).convert("L")) > 127).astype(np.uint8)


def _largest_components(mask: np.ndarray, k: int = 2) -> list[np.ndarray]:
    """Return up to k largest connected components as point clouds (x, y)."""
    if cv2 is None:
        ys, xs = np.nonzero(mask)
        return [np.column_stack([xs, ys])] if len(xs) else []
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    order = sorted(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA],
                   reverse=True)[:k]
    comps = []
    for i in order:
        ys, xs = np.nonzero(lab == i)
        if len(xs) >= 10:
            comps.append(np.column_stack([xs, ys]).astype(float))
    return comps


def _fit_line(pts: np.ndarray) -> tuple[float, float]:
    """Least-squares line; returns (angle_rad_of_direction, mean_y_at_mean_x)."""
    x, y = pts[:, 0], pts[:, 1]
    if np.std(x) < 1e-6:
        return math.pi / 2, float(np.mean(y))
    slope, intercept = np.polyfit(x, y, 1)
    return math.atan(slope), float(np.mean(y))


def apo_midlines(apo_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Split apo mask into superficial (top) / deep (bottom) point clouds."""
    comps = _largest_components(apo_mask, k=2)
    if len(comps) < 2:
        # Fallback: split single band horizontally by y-median gap.
        ys, xs = np.nonzero(apo_mask)
        if len(xs) < 20:
            return None
        hist, edges = np.histogram(ys, bins=50)
        valley = int(np.argmin(hist[len(hist) // 4:3 * len(hist) // 4])
                     + len(hist) // 4)
        cut = edges[valley]
        top = np.column_stack([xs[ys < cut], ys[ys < cut]]).astype(float)
        bot = np.column_stack([xs[ys >= cut], ys[ys >= cut]]).astype(float)
        if len(top) < 10 or len(bot) < 10:
            return None
        return top, bot
    comps.sort(key=lambda c: c[:, 1].mean())
    return comps[0], comps[1]


def fascicle_angle(fasc_mask: np.ndarray) -> float | None:
    """Dominant fascicle direction in radians (0 = horizontal)."""
    pts_all: list[np.ndarray] = []
    for comp in _largest_components(fasc_mask, k=6):
        if len(comp) < 15:
            continue
        # PCA direction of each fragment; length-weight the vote.
        c = comp - comp.mean(axis=0)
        _, _, vt = np.linalg.svd(c, full_matrices=False)
        direction = vt[0]
        length = float(np.linalg.norm(c @ direction))
        ang = math.atan2(direction[1], direction[0]) % math.pi
        pts_all.append((ang, length))
    if not pts_all:
        return None
    # Circular mean weighted by fragment length (angles modulo pi).
    angs = np.array([a for a, _ in pts_all])
    w = np.array([l for _, l in pts_all])
    vec = np.sum(w * np.exp(2j * angs)) / np.sum(w)
    return float(np.angle(vec) / 2) % math.pi


def measure(apo_mask: np.ndarray, fasc_mask: np.ndarray,
            px_per_mm: float = 1.0) -> dict[str, float | None]:
    """Full measurement in mm/deg. Returns dict with pa_deg/fl_mm/mt_mm (+px)."""
    out: dict[str, float | None] = {"pa_deg": None, "fl_mm": None,
                                    "mt_mm": None, "mt_px": None}
    apo = apo_midlines(apo_mask)
    if apo is None:
        return out
    sup, deep = apo
    deep_ang, _ = _fit_line(deep)
    sup_ang, _ = _fit_line(sup)
    apo_ang = float(np.angle((np.exp(2j * deep_ang) + np.exp(2j * sup_ang)) / 2) / 2)
    f_ang = fascicle_angle(fasc_mask)
    if f_ang is None:
        return out
    pa = abs((f_ang - apo_ang + math.pi / 2) % math.pi - math.pi / 2)
    pa_deg = float(np.degrees(pa))
    # MT: perpendicular distance between midlines along shared x-range.
    x0 = max(sup[:, 0].min(), deep[:, 0].min())
    x1 = min(sup[:, 0].max(), deep[:, 0].max())
    if x1 - x0 < 10:
        return out
    xs = np.linspace(x0, x1, 50)
    # Robust: bin means per column.
    def col_mean(pts: np.ndarray) -> np.ndarray:
        dig = np.digitize(pts[:, 0], xs)
        return np.array([pts[:, 1][dig == i].mean()
                         if (dig == i).any() else np.nan
                         for i in range(1, len(xs) + 1)])
    dy = np.nanmean(np.abs(col_mean(deep) - col_mean(sup)))
    if not np.isfinite(dy) or dy <= 0:
        return out
    mt_px = float(dy * math.cos(apo_ang))  # perpendicular (apo ~ horizontal)
    out.update({"pa_deg": pa_deg, "mt_px": mt_px,
                "mt_mm": mt_px / px_per_mm if px_per_mm else None})
    if pa_deg > 0.5:
        fl_px = mt_px / math.sin(math.radians(pa_deg))
        out["fl_mm"] = fl_px / px_per_mm if px_per_mm else None
    return out


def derive_labels(manifest_csv: Path, apo_manifest_csv: Path | None,
                  px_per_mm: float, out_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(manifest_csv)
    apo_lookup: dict[str, str] = {}
    if apo_manifest_csv and apo_manifest_csv.exists():
        apo = pd.read_csv(apo_manifest_csv)
        apo_lookup = dict(zip(apo["image_id"].astype(str),
                              apo["mask"].astype(str)))
    rows = []
    for _, r in df.iterrows():
        try:
            fasc = load_mask(r["mask"]) if pd.notna(r.get("mask")) else None
            apo_path = apo_lookup.get(str(r["image_id"]))
            apo_m = load_mask(apo_path) if apo_path else None
            if fasc is None or apo_m is None:
                m = {"pa_deg": None, "fl_mm": None, "mt_mm": None}
            else:
                m = measure(apo_m, fasc, px_per_mm)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] {r['image_id']}: {e}")
            m = {"pa_deg": None, "fl_mm": None, "mt_mm": None}
        rows.append({"image_id": r["image_id"], **m})
    out = pd.DataFrame(rows)
    ok = out.dropna()
    print(f"derived {len(ok)}/{len(out)} complete measurements "
          f"(px_per_mm={px_per_mm})")
    if len(ok):
        print(ok[["pa_deg", "fl_mm", "mt_mm"]].describe().to_string())
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"saved {out_csv}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--derive", action="store_true")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--apo-manifest", default=None)
    ap.add_argument("--px-per-mm", type=float, default=1.0)
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    if args.derive:
        derive_labels(Path(args.manifest),
                      Path(args.apo_manifest) if args.apo_manifest else None,
                      args.px_per_mm, Path(args.out))


if __name__ == "__main__":
    main()
