"""Official UMUD score (port of the organizer metric notebook) + reporting.

Primary score: mean over targets of MAE_c / tau_c, with
  tau_pa = 6.0 deg, tau_fl = 12.0 mm, tau_mt = 3.0 mm.
Lower is better. Deterministic MedAE/RMSE tie-breakers included for parity.
"""
import numpy as np
import pandas as pd

from config import TARGETS, TOLERANCES


def umud_score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str = "image_id",
) -> float:
    target_cols = tuple(TARGETS)
    eps_secondary, eps_tertiary = 1e-6, 1e-9

    for df, name in ((solution, "solution"), (submission, "submission")):
        if row_id_column_name not in df.columns:
            raise ValueError(f"{name} is missing id column '{row_id_column_name}'.")
        for col in target_cols:
            if col not in df.columns:
                raise ValueError(f"{name} is missing required column '{col}'.")
        if df[row_id_column_name].duplicated().any():
            raise ValueError(f"Duplicate ids in {name} column '{row_id_column_name}'.")

    merged = solution.merge(
        submission, on=row_id_column_name, how="inner",
        suffixes=("_true", "_pred"),
    )
    if len(merged) != len(solution):
        raise ValueError(f"Submission is missing {len(solution) - len(merged)} row ids.")

    for col in target_cols:
        pred = pd.to_numeric(merged[f"{col}_pred"], errors="coerce")
        if pred.isna().any() or np.isinf(pred.to_numpy()).any():
            raise ValueError(f"Column '{col}' must be numeric, finite, no NaNs.")
        merged[f"{col}_pred"] = pred

    primary = secondary = tertiary = 0.0
    for c in target_cols:
        y_true = merged[f"{c}_true"].to_numpy(dtype=float)
        y_pred = merged[f"{c}_pred"].to_numpy(dtype=float)
        err = np.abs(y_pred - y_true)
        tau = float(TOLERANCES[c])
        primary += float(np.mean(err)) / tau
        secondary += float(np.median(err)) / tau
        tertiary += float(np.sqrt(np.mean((y_pred - y_true) ** 2))) / tau
    n = len(target_cols)
    return float(primary / n + eps_secondary * secondary / n + eps_tertiary * tertiary / n)


def per_target_report(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str = "image_id",
) -> pd.DataFrame:
    """MAE/MedAE/RMSE/bias per target + normalized contribution to the score."""
    merged = solution.merge(
        submission, on=row_id_column_name, how="inner",
        suffixes=("_true", "_pred"),
    )
    rows = []
    for c in TARGETS:
        err = (
            merged[f"{c}_pred"].to_numpy(dtype=float)
            - merged[f"{c}_true"].to_numpy(dtype=float)
        )
        mae = float(np.mean(np.abs(err)))
        rows.append({
            "target": c,
            "mae": mae,
            "medae": float(np.median(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "bias": float(np.mean(err)),
            "tau": float(TOLERANCES[c]),
            "mae_over_tau": mae / float(TOLERANCES[c]),
        })
    return pd.DataFrame(rows)
