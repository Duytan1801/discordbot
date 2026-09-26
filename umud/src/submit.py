"""Build submission.csv from per-target prediction CSVs.

Each input csv: image_id + prediction column (auto-detected).
Order/rows follow sample_submission.csv when available.

Usage:
    python submit.py --pa test_pa.csv --fl test_fl.csv --mt test_mt.csv \\
        --sample sample_submission.csv -o submission.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RANGES, TARGETS, find_folder, resolve_input_dir  # noqa: E402
from ensemble import load_pred  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pa", required=True)
    ap.add_argument("--fl", required=True)
    ap.add_argument("--mt", required=True)
    ap.add_argument("--sample", default=None)
    ap.add_argument("--input-dir", default=None)
    ap.add_argument("-o", "--out", default="submission.csv")
    args = ap.parse_args()

    sample = args.sample
    if sample is None:
        try:
            root = resolve_input_dir(args.input_dir)
            cands = [c for c in root.rglob("*.csv")
                     if "sample" in c.name.lower()]
            sample = str(cands[0]) if cands else None
        except FileNotFoundError:
            sample = None
    parts = {"pa_deg": load_pred(args.pa, "pa_deg"),
             "fl_mm": load_pred(args.fl, "fl_mm"),
             "mt_mm": load_pred(args.mt, "mt_mm")}
    sub = parts["pa_deg"].rename(columns={"pred": "pa_deg"})
    for t in ("fl_mm", "mt_mm"):
        sub = sub.merge(parts[t].rename(columns={"pred": t}), on="image_id")
    if sample:
        # Sample uses ';' separator — auto-detect delimiter.
        order = pd.read_csv(sample, sep=None, engine="python")["image_id"].astype(str)
        sub["image_id"] = sub["image_id"].astype(str)
        missing = set(order) - set(sub["image_id"])
        if missing:
            raise SystemExit(f"missing {len(missing)} test ids, e.g. "
                             f"{sorted(missing)[:3]}")
        sub = order.to_frame().merge(sub, on="image_id", how="left")
    assert not sub[TARGETS].isna().any().any(), "NaNs in submission!"
    for t in TARGETS:
        lo, hi = RANGES[t]
        oob = ((sub[t] < lo) | (sub[t] > hi)).sum()
        print(f"{t}: range [{sub[t].min():.2f}, {sub[t].max():.2f}] "
              f"out-of-phys-range={oob}")
    sub[["image_id", *TARGETS]].to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(sub)} rows)")


if __name__ == "__main__":
    main()
