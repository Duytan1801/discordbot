"""Phase B scaffold: finetune SAM (AutoMM) on apo / fasc masks.

Two binary segmentation tasks (foreground/background, num_classes=1):
    python train_segmentation.py --task apo --preset medium_quality --time-limit 1800
    python train_segmentation.py --task fasc --preset medium_quality --time-limit 1800

Expects manifests/apo_unique.csv / fasc_unique.csv with `image,mask` columns.
On Kaggle free GPU use sam-vit-base; scale to -large/-huge on better hardware.
Predicted masks -> geometry.py -> pa/fl/mt (see ensemble.py Phase B notes).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import SEED, manifests_dir, runs_dir  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["apo", "fasc"])
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--preset", default="medium_quality",
                    choices=["medium_quality", "high_quality", "best_quality"])
    ap.add_argument("--sam", default="facebook/sam-vit-base",
                    help="facebook/sam-vit-base|large|huge")
    ap.add_argument("--time-limit", type=int, default=1800)
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()

    from autogluon.multimodal import MultiModalPredictor

    man = manifests_dir()
    df = pd.read_csv(args.manifest or man / f"{args.task}_unique.csv")
    df = df.dropna(subset=["mask"])[["image", "mask"]].rename(
        columns={"mask": "label"})
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    n_val = max(1, int(len(df) * args.val_frac))
    val_df, train_df = df.iloc[:n_val], df.iloc[n_val:]
    print(f"{args.task}: train {len(train_df)} / val {len(val_df)}")

    path = str(runs_dir() / f"seg_{args.task}")
    predictor = MultiModalPredictor(
        problem_type="semantic_segmentation", label="label",
        path=path, presets=args.preset, num_classes=1,
        hyperparameters={"model.sam.checkpoint_name": args.sam},
        verbosity=2,
    )
    predictor.fit(train_data=train_df, tuning_data=val_df,
                  time_limit=args.time_limit, seed=SEED)
    print(predictor.evaluate(val_df))
    print(f"run saved to {path}")
    print("Next: predictor.predict({'image': [...]}) -> masks -> geometry.measure()")


if __name__ == "__main__":
    main()
