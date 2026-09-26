# UMUD Challenge — Muscle Architecture in Ultrasound Data

Kaggle: <https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data/overview>

**Task:** from a single B-mode ultrasound frame, predict 3 muscle-architecture
parameters — pennation angle `pa_deg` (°), fascicle length `fl_mm` (mm),
muscle thickness `mt_mm` (mm) — for 309 test images.

**Metric (lower = better):** mean of tolerance-normalized MAEs

```
S = mean(MAE_pa/6.0, MAE_fl/12.0, MAE_mt/3.0)
```

**Stack:** AutoGluon (AutoMM image→regression + SAM segmentation + Tabular
stacking), PyTorch/timm backbones. See `src/` for the pipeline.

## Key facts driving the design

1. **No numeric train labels (to verify in EDA).** The training release is
   images + segmentation masks (`fasc_*`: 2761 pairs, `apo_*`: 1048 pairs).
   If no CSV with pa/fl/mt exists, numeric targets must be **derived from
   masks via geometry** (`src/geometry.py`) — the same way the organizers'
   reference tools (DLTrack/UltraTimTrack) work.
2. **670 exact duplicate fascicle pairs** (organizer-confirmed). Only ~2091
   unique pairs. `src/data.py` dedups by SHA-256 before any split.
3. **Domain shift is the competition:** test has an unseen muscle (Rectus
   Femoris), an unseen device (Philips Lumify), CP patients, unseen subjects,
   plus 5-frame video sequences (temporal smoothing helps).
4. AutoMM is **single-label** → one predictor per target (pa / fl / mt).
5. AutoMM has **no custom metric** → train on `mae`, select/ensemble on the
   true UMUD score implemented in `src/metrics.py`.
6. Top-3 needs **OSI license + FAIR + re-runnable repo** → this repo is MIT
   licensed, everything runs via `python src/*.py` with pinned requirements.

## Layout

```
umud/
├── requirements-kaggle.txt     # Kaggle notebook env
├── notebooks/
│   └── 01_eda_and_baseline.ipynb  # upload to Kaggle, Add Data, Run All
└── src/
    ├── config.py               # paths, tolerances, ranges, seeds
    ├── metrics.py              # official UMUD score + per-target report
    ├── data.py                 # download/discover/manifests/dedup/splits
    ├── eda.py                  # stats + figures -> reports/
    ├── geometry.py             # masks -> pa/fl/mt (+ mm/px calibration hook)
    ├── train_regression.py     # Phase A: AutoMM image->regression per target
    ├── train_segmentation.py   # Phase B: SAM finetune on apo/fasc masks
    ├── ensemble.py             # Phase C: weighting, TTA, temporal smoothing
    └── submit.py               # merge preds -> submission.csv + validate
```

## Quickstart (Kaggle notebook, free GPU)

1. Create a notebook, **Add Data → UMUD competition data**
   (mounts at `/kaggle/input/...`), attach a GPU accelerator.
2. Upload `notebooks/01_eda_and_baseline.ipynb` (or paste cells) and Run All.
   It installs deps, builds manifests, runs EDA, trains 3 quick AutoMM
   regressors, and writes `submission.csv`.

Or step by step in a notebook terminal:

```bash
pip install -q -r umud/requirements-kaggle.txt
cd umud/src
python data.py --discover            # inventory of /kaggle/input
python data.py --manifests           # build + dedup manifests -> ../data/manifests
python eda.py                        # stats + figures -> ../reports
python data.py --splits              # train/val split (needs labels or geometry)
python train_regression.py --target pa_deg --preset medium_quality --time-limit 900
python train_regression.py --target fl_mm  --preset medium_quality --time-limit 900
python train_regression.py --target mt_mm  --preset medium_quality --time-limit 900
python submit.py --run-dirs runs/pa runs/fl runs/mt -o submission.csv
```

## Phases (podium plan)

- [x] **Phase 0** — scaffold: data/metric/EDA/geometry/submit
- [ ] **Phase A** — 3× AutoMM regression baselines → first LB score
- [ ] **Phase B** — SAM segmentation → geometric PA/FL/MT → compare/fuse
- [ ] **Phase C** — best_quality, 384px backbones, custom aug, HPO,
      embedding+Tabular stacking, TTA, temporal smoothing, UMUD-weighted ensemble
- [ ] **Phase D** — freeze reproducible repo + method writeup (prize eligibility)

## Open questions for EDA

- Is there any numeric-label CSV, or must targets come from masks?
- How to convert px → mm (metadata in tif? fixed scale per device? scale bar)?
  PA is scale-free; FL/MT need calibration.
- Do filenames encode muscle/device/subject (for grouping + fusion features)?
- Test image_id → 5-frame group mapping (for temporal smoothing)?
