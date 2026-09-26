"""EDA: manifest stats, image/mask sampling, label distributions, figures.

Usage: python eda.py [--input-dir X] [--labels labels.csv] [--n-samples 6]
Outputs: reports/eda_summary.json + PNG figures.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import TARGETS, manifests_dir, reports_dir, resolve_input_dir  # noqa: E402
from data import discover  # noqa: E402


def image_info(path: str) -> dict:
    try:
        with Image.open(path) as im:
            return {"size": list(im.size), "mode": im.mode,
                    "format": im.format, "ok": True}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--n-samples", type=int, default=6)
    args = ap.parse_args()

    root = resolve_input_dir(args.input_dir)
    info = discover(root)
    man, rep = manifests_dir(), reports_dir()
    summary: dict = {"input_root": str(root), "folders": info["folders"],
                     "csvs": info["csvs"], "label_csvs": info["label_csvs"]}

    for name in ("fasc_all", "fasc_unique", "apo_all", "apo_unique", "test"):
        p = man / f"{name}.csv"
        if not p.exists():
            print(f"[skip] {p} missing (run data.py --manifests first)")
            continue
        df = pd.read_csv(p)
        print(f"\n== {name}: {len(df)} rows ==")
        summary[name] = {"n": len(df)}
        # Sample image readability + sizes + filename patterns.
        sample = df.sample(min(len(df), max(20, args.n_samples)),
                           random_state=0)
        infos = [image_info(x) for x in sample["image"]]
        ok = [i for i in infos if i.get("ok")]
        sizes = {tuple(i["size"]) for i in ok}
        modes = {i["mode"] for i in ok}
        print(f"  readable {len(ok)}/{len(infos)}, sizes={sorted(sizes)[:8]}, "
              f"modes={sorted(modes)}")
        stems = df["image_id"].astype(str)
        print(f"  id sample: {stems.head(3).tolist()}")
        # Filename pattern probe: common prefixes/suffixes (subject/device/muscle?)
        prefix = stems.str.extract(r"^([A-Za-z]+)")[0].value_counts().head(5)
        print(f"  alpha-prefix counts:\n{prefix.to_string()}")
        summary[name].update({
            "readable": f"{len(ok)}/{len(infos)}",
            "sizes": sorted([list(s) for s in sizes])[:8],
            "modes": sorted(modes),
            "id_head": stems.head(5).tolist(),
        })
        # Mask coverage probe.
        if "mask" in df.columns:
            m = df["mask"].dropna()
            if len(m):
                cov = []
                for mp in m.sample(min(len(m), 30), random_state=1):
                    try:
                        a = np.asarray(Image.open(mp).convert("L"))
                        cov.append(float((a > 127).mean()))
                    except Exception:  # noqa: BLE001
                        pass
                if cov:
                    print(f"  mask foreground frac: mean={np.mean(cov):.3f} "
                          f"min={np.min(cov):.3f} max={np.max(cov):.3f}")
                    summary[name]["mask_fg_frac"] = {
                        "mean": float(np.mean(cov)),
                        "min": float(np.min(cov)), "max": float(np.max(cov))}
        # Sample figure.
        fig, axes = plt.subplots(min(len(sample), args.n_samples), 2,
                                 figsize=(10, 3 * args.n_samples))
        axes = np.atleast_2d(axes)
        for ax_row, (_, row) in zip(axes, sample.head(args.n_samples).iterrows()):
            try:
                ax_row[0].imshow(np.asarray(Image.open(row["image"]).convert("L")),
                                 cmap="gray")
            except Exception as e:  # noqa: BLE001
                ax_row[0].text(0.5, 0.5, f"ERR {e}", ha="center")
            ax_row[0].set_title(row["image_id"])
            if "mask" in df.columns and pd.notna(row["mask"]):
                try:
                    ax_row[1].imshow(np.asarray(
                        Image.open(row["mask"]).convert("L")), cmap="gray")
                except Exception as e:  # noqa: BLE001
                    ax_row[1].text(0.5, 0.5, f"ERR {e}", ha="center")
                ax_row[1].set_title(f"{row['image_id']} mask")
            for ax in ax_row:
                ax.axis("off")
        fig.tight_layout()
        fig.savefig(rep / f"samples_{name}.png", dpi=90)
        plt.close(fig)
        print(f"  saved reports/samples_{name}.png")

    # Label distributions (organizer CSV or geometry-derived).
    if args.labels and Path(args.labels).exists():
        lab = pd.read_csv(args.labels)
        print(f"\n== labels {args.labels}: {len(lab)} rows ==")
        desc = lab[[c for c in TARGETS if c in lab.columns]].describe().T
        print(desc.to_string())
        summary["labels"] = {"n": len(lab),
                             "describe": desc.to_dict(orient="index")}
        fig, axes = plt.subplots(1, len(desc), figsize=(4 * len(desc), 3))
        axes = np.atleast_1d(axes)
        for ax, c in zip(axes, desc.index):
            ax.hist(lab[c].dropna(), bins=40)
            ax.set_title(c)
        fig.tight_layout()
        fig.savefig(rep / "label_hist.png", dpi=90)
        plt.close(fig)
    else:
        print("\n[note] no --labels given: numeric-label EDA skipped. "
              "If organizers ship no label CSV, derive with geometry.py.")

    (rep / "eda_summary.json").write_text(json.dumps(summary, indent=2,
                                                     default=str))
    print("\nsummary -> reports/eda_summary.json")


if __name__ == "__main__":
    main()
