"""Phase A: AutoMM image -> regression, ONE predictor per target.

AutoMM is single-label, so run once per target (pa_deg / fl_mm / mt_mm).
Trains on `mae` internally (no custom metric in AutoMM); use metrics.py
(UMUD score) externally for model selection.

Key customization for ultrasound geometry:
  - NO vertical flip / aggressive crops (would destroy apo layout).
  - horizontal flip + mild photometric/affine only.
  - label scaling built in (standardscaler default).

Examples (Kaggle free GPU):
    python train_regression.py --target pa_deg --preset medium_quality \\
        --time-limit 900 --ballpark
    python train_regression.py --target mt_mm --preset best_quality \\
        --backbone convnext_base_384 --max-epochs 20 --time-limit 7200
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RANGES, SEED, manifests_dir, runs_dir  # noqa: E402

# Geometry-safe training augmentation for longitudinal muscle ultrasound.
SAFE_TRAIN_TRANSFORMS = [
    "resize_shorter_side", "center_crop",
    "random_horizontal_flip", "color_jitter",
]
SAFE_VAL_TRANSFORMS = ["resize_shorter_side", "center_crop"]


def build_hparams(backbone: str | None, max_epochs: int | None) -> dict:
    hp: dict = {
        "model.names": ["timm_image"],
        "model.timm_image.train_transforms": SAFE_TRAIN_TRANSFORMS,
        "model.timm_image.val_transforms": SAFE_VAL_TRANSFORMS,
        "optim.patience": 10,
        "optim.val_check_interval": 0.5,
        "env.num_workers": 2,
    }
    if backbone:
        hp["model.timm_image.checkpoint_name"] = backbone
    if max_epochs:
        hp["optim.max_epochs"] = max_epochs
    return hp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True,
                    choices=["pa_deg", "fl_mm", "mt_mm"])
    ap.add_argument("--train-csv", default=None)
    ap.add_argument("--val-csv", default=None)
    ap.add_argument("--image-col", default="image")
    ap.add_argument("--preset", default="medium_quality",
                    choices=["medium_quality", "high_quality", "best_quality"])
    ap.add_argument("--backbone", default=None,
                    help="timm checkpoint, e.g. convnext_base_384, "
                         "swinv2_base_window12_384, vit_base_patch16_384")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--time-limit", type=int, default=900,
                    help="seconds for THIS target's fit")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--ballpark", action="store_true",
                    help="also train/eval quickly and print val MAE + range check")
    args = ap.parse_args()

    from autogluon.multimodal import MultiModalPredictor

    man = manifests_dir()
    train_df = pd.read_csv(args.train_csv or man / "train.csv")
    val_df = pd.read_csv(args.val_csv or man / "val.csv")
    keep = [args.image_col, args.target]
    train_df, val_df = train_df[keep], val_df[keep]
    print(f"train {len(train_df)} / val {len(val_df)} for {args.target}")

    run_name = args.run_name or f"{args.target}_{args.preset}"
    path = str(runs_dir() / run_name)
    predictor = MultiModalPredictor(
        label=args.target, problem_type="regression", eval_metric="mae",
        path=path, presets=args.preset, verbosity=2,
    )
    predictor.fit(
        train_data=train_df, tuning_data=val_df,
        time_limit=args.time_limit,
        hyperparameters=build_hparams(args.backbone, args.max_epochs),
        seed=SEED,
    )
    print(predictor.evaluate(val_df))

    if args.ballpark:
        preds = predictor.predict(val_df).to_numpy().ravel()
        import numpy as np
        mae = float(np.mean(np.abs(
            preds - val_df[args.target].to_numpy())))
        lo, hi = RANGES[args.target]
        oob = float(((preds < lo) | (preds > hi)).mean())
        print(f"[ballpark] val MAE={mae:.3f}  out-of-range frac={oob:.3f} "
              f"(range {lo}-{hi})")
    # Save val predictions for external UMUD-score selection/stacking.
    out = val_df[[args.image_col]].copy() if args.image_col in val_df else val_df.copy()
    out["image_id"] = val_df.get("image_id", val_df.index)
    out[f"pred_{args.target}"] = predictor.predict(val_df).to_numpy().ravel()
    out.to_csv(Path(path) / f"val_preds_{args.target}.csv", index=False)
    print(f"run saved to {path}")


if __name__ == "__main__":
    main()
