# CLAUDE.md

## Project Overview

Neuroimaging reliability analysis for fetal brain subplate segmentation at FNNDSC,
written up as a paper. Measures reproducibility across four reconstruction splits
(S1–S4) per subject using Dice, Jaccard, Hausdorff, SNR and CNR. Replicates an
older lab study that measured reliability of the cortical plate model, which lacked
the subplate feature.

**Start with `docs/`** — `docs/handoff_notes.md` lists the traps,
`docs/pipeline.md` the exact commands, `docs/data_dictionary.md` what every CSV
column is and which script wrote it.

## Commands

```bash
uv sync                      # install deps from uv.lock
uv run ruff check .          # lint    (~80 pre-existing errors, mostly in notebooks)
uv run ruff format .         # format
uv run ty check              # type check
uv run jupyter lab           # notebooks
```

Analysis (no cluster needed — `data/*.csv` is tracked):

```bash
cd src && uv run jupyter lab      # paper_models.ipynb, then bootstrap_covariates.ipynb
```

Data generation (needs FNNDSC cluster access) — see `docs/pipeline.md`.

## Architecture

### Data model

Long format: one row per subject-split or subject-split-pair. Join on
`subject_id` + `session_id`. Derived metrics computed at analysis time.
Core CSVs in `data/` — `subject.csv` and `raw_subject_data.csv` are hand-curated
and irreplaceable; the rest are script-derived. Full column-by-column provenance
is in `docs/data_dictionary.md`.

### Module structure

| Directory | Purpose |
|-----------|---------|
| `src/` | Current analysis: `paper_models*.ipynb`, `bootstrap_covariates.ipynb`, `slides_helpers.py` (regression + result tables), `image_quality_metrics.py` (SNR/CNR/volume CLI) |
| `src/functions/` | `helpers.py` (image ops, segmentation metrics), `CNR.py` (boundary contrast ratio) |
| `src/reliability/` | Per-subject Dice/Jaccard reports: `Reliability.ipynb` + `batch_process.py` (papermill) |
| `src/processing/` | Cluster-side: SP prediction, surface extraction, surface/thickness audits |
| `src/surface_analysis/` | Surface and thickness notebooks |
| `src/multivariate_analysis/`, `src/cnr_analisys/`, `src/summary_analisys/` | Legacy, superseded by `paper_models.ipynb` |
| `assets/` | Result tables + reference-paper PDFs; `assets/artifacts/` holds generated figures |
| `archive/` | Dead code, kept for traceability |

### Key conventions

- **Tissue labels**: Subplate (SP) = 4, 5; Cortical Plate (CP) = 1, 42; Inner zone = 160, 161
- **Splits**: S1–S4; only the non-overlapping pairs S1-S2 and S3-S4 are modelled
- **Working directory matters** — notebooks and scripts use relative paths and
  CWD-based imports. Run each from the directory it lives in. `src/functions/CNR.py`
  is the exception: `uv run python -m src.functions.CNR` from the repo root.
- Do not rename `cnr_analisys/` or `summary_analisys/` — a `sys.path.insert`
  in `cnr_analysis.ipynb` depends on the spelling
- **Base data path**: `/neuro/labs/grantlab/research/MRI_processing/seungyoon.jeong/2025/Reliability/TEST/` — not portable
- **FreeSurfer LUT**: `reference/FreeSurferColorLUT.txt` (code reads the cluster copy)
- Several scripts and notebooks **rewrite CSVs in place** — see `docs/handoff_notes.md`
