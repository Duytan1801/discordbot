"""Shared constants and path resolution for the UMUD pipeline."""
import os
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent          # umud/
REPO_ROOT = PROJECT_DIR.parent

COMPETITION = "umud-challenge-muscle-architecture-in-ultrasound-data"

TARGETS = ["pa_deg", "fl_mm", "mt_mm"]
# Organizer tolerances (MDC-based) for UMUD-score normalization.
TOLERANCES = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}
# Physiological test ranges (used for clipping + sanity checks).
RANGES = {"pa_deg": (5.0, 45.0), "fl_mm": (30.0, 200.0), "mt_mm": (10.0, 50.0)}

SEED = 42

# Folder-name fragments to discover inside the competition input dir.
FOLDER_HINTS = {
    "fasc_imgs": ("fasc", "img"),
    "fasc_masks": ("fasc", "mask"),
    "apo_imgs": ("apo", "img"),
    "apo_masks": ("apo", "mask"),
    "test_images": ("test",),
}


def resolve_input_dir(explicit: str | None = None) -> Path:
    """Find the competition data directory.

    Order: explicit arg > UMUD_DATA env > /kaggle/input/* > ./data > ~/data.
    """
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("UMUD_DATA"):
        candidates.append(Path(os.environ["UMUD_DATA"]))
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        # `kaggle/input/<slug>` or with -vN suffix
        for child in sorted(kaggle_input.iterdir()):
            if child.is_dir() and "umud" in child.name.lower():
                candidates.append(child)
        candidates.append(kaggle_input)
    candidates += [PROJECT_DIR / "data", REPO_ROOT / "data", Path.home() / "data"]

    for c in candidates:
        if c.exists() and any(c.iterdir()):
            return c
    raise FileNotFoundError(
        "Could not find competition data. Pass --input-dir, set UMUD_DATA, "
        "or add the dataset to your Kaggle notebook (Add Data)."
    )


def find_folder(root: Path, *fragments: str) -> Path | None:
    """First subdir (recursive, shallow-first) whose name contains all fragments."""
    cands = sorted(
        [p for p in root.rglob("*") if p.is_dir()],
        key=lambda p: len(p.parts),
    )
    for p in cands:
        name = p.name.lower()
        if all(f in name for f in fragments):
            return p
    return None


def manifests_dir() -> Path:
    d = PROJECT_DIR / "data" / "manifests"
    d.mkdir(parents=True, exist_ok=True)
    return d


def reports_dir() -> Path:
    d = PROJECT_DIR / "reports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def runs_dir() -> Path:
    d = PROJECT_DIR / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d
