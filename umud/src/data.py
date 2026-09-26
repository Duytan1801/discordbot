"""Data discovery, manifests, SHA-256 dedup, numeric-label search, splits.

Usage (from umud/src):
    python data.py --discover
    python data.py --manifests [--input-dir X]
    python data.py --splits --labels <labels.csv> [--group-col Col]
    python data.py --download   # local only; on Kaggle use Add Data instead
"""
import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    COMPETITION, FOLDER_HINTS, SEED, TARGETS, find_folder, manifests_dir,
    resolve_input_dir,
)

IMG_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def download(output_dir: Path) -> Path:
    """Download + unzip competition data via the Kaggle API (needs kaggle.json)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "competitions", "download", "-c", COMPETITION,
           "-p", str(output_dir)]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)
    for z in output_dir.glob("*.zip"):
        print(f"unzipping {z.name} ...")
        subprocess.run(["unzip", "-q", "-o", str(z), "-d", str(output_dir)],
                       check=True)
    return output_dir


def discover(root: Path) -> dict:
    """Inventory folders/files; returns resolved paths dict (also printed)."""
    print(f"input root: {root}")
    info: dict = {"root": str(root), "folders": {}, "csvs": []}
    for key, frags in FOLDER_HINTS.items():
        p = find_folder(root, *frags)
        info["folders"][key] = str(p) if p else None
        if p:
            n = sum(1 for f in p.rglob("*")
                    if f.is_file() and f.suffix.lower() in IMG_EXTS)
            print(f"  {key:12s} {p}  ({n} images)")
        else:
            print(f"  {key:12s} NOT FOUND (fragments={frags})")
    for csv in sorted(root.rglob("*.csv")):
        info["csvs"].append(str(csv))
        try:
            df = pd.read_csv(csv, nrows=3)
            print(f"  csv: {csv.relative_to(root)} cols={list(df.columns)}")
        except Exception as e:  # noqa: BLE001
            print(f"  csv: {csv.relative_to(root)} (unreadable: {e})")
    # Any numeric-label CSV?
    label_csvs = [c for c in info["csvs"] if _has_targets(c)]
    info["label_csvs"] = label_csvs
    print(f"  label-like csvs (contain pa/fl/mt cols): {label_csvs or 'NONE'}")
    return info


def _has_targets(csv_path: str) -> bool:
    try:
        cols = [c.lower() for c in pd.read_csv(csv_path, nrows=0).columns]
    except Exception:  # noqa: BLE001
        return False
    joined = " ".join(cols)
    return ("pa" in joined and "fl" in joined and "mt" in joined) or set(
        TARGETS) <= set(cols)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pair_images_masks(img_dir: Path, mask_dir: Path | None) -> pd.DataFrame:
    imgs = sorted([p for p in img_dir.rglob("*")
                   if p.is_file() and p.suffix.lower() in IMG_EXTS])
    masks = {}
    if mask_dir is not None:
        for p in mask_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                masks.setdefault(p.stem, p)
    rows = []
    for img in imgs:
        m = masks.get(img.stem)
        rows.append({"image_id": img.stem, "image": str(img),
                     "mask": str(m) if m else None})
    return pd.DataFrame(rows)


def build_manifests(root: Path, out_dir: Path | None = None) -> dict[str, Path]:
    """Build fasc/apo/test manifests with SHA-256 hashes; dedup fasc pairs.

    Saves: fasc_all.csv, fasc_unique.csv, fasc_dup_map.csv, apo_all.csv,
           test.csv, sample_submission copy note. Returns {name: path}.
    """
    out_dir = out_dir or manifests_dir()
    folders = {k: (Path(find_folder(root, *frags)) if find_folder(root, *frags) else None)
               for k, frags in FOLDER_HINTS.items()}
    saved: dict[str, Path] = {}

    for key in ("fasc", "apo"):
        img_dir, mask_dir = folders.get(f"{key}_imgs"), folders.get(f"{key}_masks")
        if img_dir is None:
            print(f"[warn] {key}_imgs not found, skipping")
            continue
        df = _pair_images_masks(img_dir, mask_dir)
        print(f"{key}: {len(df)} images, "
              f"{df['mask'].notna().sum()} with matched masks")
        df["image_sha256"] = [sha256_of(Path(p)) for p in df["image"]]
        df["mask_sha256"] = [sha256_of(Path(p)) if pd.notna(p) else ""
                             for p in df["mask"]]
        df["pair_key"] = df["image_sha256"] + "|" + df["mask_sha256"]
        df.to_csv(out_dir / f"{key}_all.csv", index=False)
        saved[f"{key}_all"] = out_dir / f"{key}_all.csv"
        # Dedup: keep first row per (image,mask) hash pair.
        dup = df.duplicated("pair_key", keep=False)
        uniq = df.drop_duplicates("pair_key", keep="first").copy()
        df.assign(keep=lambda d: ~d.duplicated("pair_key", keep="first")).to_csv(
            out_dir / f"{key}_dup_map.csv", index=False)
        uniq.to_csv(out_dir / f"{key}_unique.csv", index=False)
        saved[f"{key}_unique"] = out_dir / f"{key}_unique.csv"
        print(f"{key}: {int(dup.sum())} rows in dup groups -> "
              f"{len(uniq)} unique pairs kept")

    test_dir = folders.get("test_images")
    if test_dir is not None:
        test_imgs = sorted([p for p in test_dir.rglob("*")
                            if p.is_file() and p.suffix.lower() in IMG_EXTS])
        tdf = pd.DataFrame([{"image_id": p.stem, "image": str(p)}
                            for p in test_imgs])
        tdf.to_csv(out_dir / "test.csv", index=False)
        saved["test"] = out_dir / "test.csv"
        print(f"test: {len(tdf)} images")
    print(f"manifests in {out_dir}")
    return saved


def make_splits(manifest_csv: Path, labels_csv: Path | None,
                group_col: str | None = None,
                val_frac: float = 0.2, seed: int = SEED,
                out_dir: Path | None = None) -> tuple[Path, Path]:
    """Join manifest with numeric labels -> stratified-ish train/val CSVs.

    labels_csv must contain image_id + pa_deg/fl_mm/mt_mm (from organizers or
    from geometry.py derivation). If group_col given, groups stay in one fold.
    """
    out_dir = out_dir or manifests_dir()
    df = pd.read_csv(manifest_csv)
    if labels_csv is None:
        raise SystemExit(
            "No numeric labels provided. If the organizers shipped no label CSV, "
            "derive targets from masks first: python geometry.py --derive ...")
    labels = pd.read_csv(labels_csv)
    missing = (set(TARGETS) | {"image_id"}) - set(labels.columns)
    if missing:
        raise SystemExit(f"labels csv missing columns: {sorted(missing)}")
    df = df.merge(labels[["image_id", *TARGETS]], on="image_id", how="inner")
    print(f"joined {len(df)}/{len(pd.read_csv(manifest_csv))} rows with labels")
    rng = __import__("numpy").random.default_rng(seed)
    if group_col and group_col in df.columns:
        groups = df[group_col].fillna("na").unique()
        rng.shuffle(groups)
        n_val = max(1, int(len(groups) * val_frac))
        val_groups = set(groups[:n_val])
        val_mask = df[group_col].fillna("na").isin(val_groups)
    else:
        idx = df.index.to_numpy()
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_frac))
        val_mask = df.index.isin(idx[:n_val])
    train_df, val_df = df[~val_mask], df[val_mask]
    tp, vp = out_dir / "train.csv", out_dir / "val.csv"
    train_df.to_csv(tp, index=False)
    val_df.to_csv(vp, index=False)
    print(f"train {len(train_df)} -> {tp}\nval   {len(val_df)} -> {vp}")
    return tp, vp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=None)
    ap.add_argument("--discover", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--manifests", action="store_true")
    ap.add_argument("--splits", action="store_true")
    ap.add_argument("--manifest", default=None, help="default: manifests/fasc_unique.csv")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--group-col", default=None)
    ap.add_argument("--val-frac", type=float, default=0.2)
    args = ap.parse_args()

    if args.download:
        download(Path(args.input_dir or "data"))
        return
    root = resolve_input_dir(args.input_dir)
    if args.discover or not (args.manifests or args.splits):
        discover(root)
    if args.manifests:
        build_manifests(root)
    if args.splits:
        manifest = Path(args.manifest or manifests_dir() / "fasc_unique.csv")
        make_splits(manifest,
                    Path(args.labels) if args.labels else None,
                    group_col=args.group_col, val_frac=args.val_frac)


if __name__ == "__main__":
    main()
