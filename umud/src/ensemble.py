"""Phase C helpers: UMUD-weighted averaging, range clipping, temporal smoothing.

1) Weighted average of per-target prediction CSVs (each: image_id + pred col).
   Weights chosen by grid search on validation UMUD score (see --tune).
2) Clip to physiological ranges (config.RANGES) — always safe.
3) Temporal smoothing: test contains 5-frame video sequences. If a grouping
   CSV (image_id, group) is provided, median-pool predictions within groups.

Usage:
    python ensemble.py --preds run1/val_preds_pa_deg.csv run2/val_preds_pa_deg.csv \\
        --target pa_deg --solution ../data/manifests/val_labels.csv --tune -o avg.csv
    python ensemble.py --preds test_pa.csv --target pa_deg --groups groups.csv -o test_pa_smooth.csv
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RANGES, TARGETS  # noqa: E402
from metrics import per_target_report, umud_score  # noqa: E402


def load_pred(path: str, target: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    pred_col = next((c for c in df.columns
                     if c not in ("image_id", "image") and
                     ("pred" in c.lower() or c == target)), None)
    if pred_col is None:
        raise SystemExit(f"no prediction column found in {path} "
                         f"(cols={list(df.columns)})")
    if "image_id" not in df.columns and "image" in df.columns:
        df["image_id"] = df["image"].map(
            lambda p: Path(str(p)).stem)  # noqa: PD008
    return df[["image_id", pred_col]].rename(columns={pred_col: "pred"})


def weighted_avg(dfs: list[pd.DataFrame], weights: list[float]) -> pd.DataFrame:
    base = dfs[0][["image_id"]].copy()
    mat = np.column_stack([d["pred"].to_numpy(dtype=float) for d in dfs])
    w = np.array(weights, dtype=float)
    base["pred"] = mat @ (w / w.sum())
    return base


def tune_weights(dfs: list[pd.DataFrame], solution: pd.DataFrame,
                 target: str, step: float = 0.1) -> list[float]:
    """Grid-search simplex weights minimizing single-target MAE/tau."""
    n = len(dfs)
    best, best_s = None, float("inf")
    grid = np.arange(0, 1 + 1e-9, step)
    for combo in itertools.product(grid, repeat=n):
        if sum(combo) <= 0:
            continue
        avg = weighted_avg(dfs, list(combo)).rename(columns={"pred": target})
        sub = solution[["image_id"]].merge(avg, on="image_id")
        sol = solution[["image_id", target]]
        s = float(np.mean(np.abs(sub[target].to_numpy()
                                   - sol[target].to_numpy())))
        if s < best_s:
            best_s, best = s, list(combo)
    print(f"best {target}: weights={best} val_MAE={best_s:.4f}")
    return best or [1.0] * n


def smooth_groups(df: pd.DataFrame, groups_csv: Path,
                  pred_col: str = "pred") -> pd.DataFrame:
    g = pd.read_csv(groups_csv)
    if not {"image_id", "group"} <= set(g.columns):
        raise SystemExit("groups csv needs columns: image_id, group")
    df = df.merge(g[["image_id", "group"]], on="image_id", how="left")
    df[pred_col] = df.groupby("group")[pred_col].transform("median")
    return df.drop(columns=["group"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--target", required=True, choices=TARGETS)
    ap.add_argument("--solution", default=None,
                    help="val labels csv (image_id + targets) for --tune/report")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--groups", default=None)
    ap.add_argument("--no-clip", action="store_true")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    dfs = [load_pred(p, args.target) for p in args.preds]
    ids = dfs[0]["image_id"]
    assert all(d["image_id"].equals(ids) for d in dfs[1:]), \
        "prediction row order/ids differ across inputs"
    sol = pd.read_csv(args.solution) if args.solution else None
    weights = args.weights or (
        tune_weights(dfs, sol, args.target) if (args.tune and sol is not None)
        else [1.0] * len(dfs))
    out = weighted_avg(dfs, weights).rename(columns={"pred": args.target})
    if args.groups:
        out = smooth_groups(out.rename(columns={args.target: "pred"}),
                            Path(args.groups)).rename(columns={"pred": args.target})
        print(f"temporal smoothing applied ({args.groups})")
    if not args.no_clip:
        lo, hi = RANGES[args.target]
        out[args.target] = out[args.target].clip(lo, hi)
    if sol is not None:
        sub = sol[["image_id"]].merge(out, on="image_id")
        print(per_target_report(sol[["image_id", args.target]], sub).to_string(index=False))
        if set(TARGETS) <= set(sub.columns):
            print(f"UMUD(val) = {umud_score(sol, sub):.5f}")
    out.to_csv(args.out, index=False)
    print(f"saved {args.out} (weights={weights})")


if __name__ == "__main__":
    main()
