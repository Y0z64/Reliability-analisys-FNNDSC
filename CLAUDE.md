# CLAUDE.md
## Project Overview

Neuroimaging reliability analysis project for fetal brain segmentation at FNNDSC with the intention of writting a paper. Measures reproducibility of segmentation results across multiple processing splits (S1–S4) per subject using metrics like Dice coefficient, Hausdorff distance, Jaccard index, SNR, and CNR. The measurements are done in a specific model created to segment the subplate with the intention of recreating an older paper that measured reliability in older models without this feature.

## Commands

### Package Management (uv)

```bash
uv sync              # Install/sync dependencies from uv.lock
uv add <package>     # Add a new dependency
uv run <script>      # Run a script in the project environment
```

### Linting and Type Checking

```bash
uv run ruff check .          # Lint
uv run ruff format .         # Format
uv run ty check              # Type check
```

### Running Analysis Scripts

```bash
# Batch reliability analysis (requires Reliability.ipynb in src/reliability/)
cd src/reliability && uv run python batch_process.py

# Surface extraction batch (multiprocessing)
uv run python src/processing/run_SP_batch.py --subjects data/subject.csv

# Image quality metrics
uv run python src/image_quality_metrics.py --subjects data/subject.csv --base_path /neuro/labs/grantlab/research/MRI_processing/
```

### Jupyter Notebooks

```bash
uv run jupyter lab   # Start JupyterLab
```

## Architecture

### Data Model

All analysis data is in **long format** (one row per subject-split combination). Derived metrics are computed at analysis time. Core CSVs in `data/`:
- `raw_subject_data.csv` — per-subject volume and quality metrics
- `split_comparision_data.csv` — pairwise reliability metrics across splits
- `cross_split_metrics.csv` — aggregated cross-split comparisons
- `image_quality_metrics.csv` — SNR/CNR per subject

### Module Structure

| Directory | Purpose |
|-----------|---------|
| `src/functions/` | Shared utilities: `helpers.py` (image ops, segmentation metrics), `CNR.py` (contrast-to-noise ratio computation) |
| `src/reliability/` | Core reliability analysis: `Reliability.py` (Dice/Jaccard/Hausdorff across splits), `batch_process.py` (papermill executor) |
| `src/processing/` | Pipeline: surface extraction, prediction batch jobs, multiprocessing workers |
| `src/multivariate_analysis/` | Multivariate regression framework; `_functions.py` prepares model inputs |
| `src/cnr_analisys/` | CNR-focused multivariate analysis; generates plots and summary tables |
| `src/summary_analisys/` | Summary statistics helpers for cross-split comparisons |

### Key Conventions

- **Tissue labels**: Subplate (SP) = 4, 5; Cortical Plate (CP) = 1, 42; Inner zones = 160, 161
- **Splits**: S1, S2, S3, S4 — different processing runs of the same subject used to measure reproducibility
- **Notebook-driven**: Primary workflow is Jupyter notebooks; batch execution uses [papermill](https://papermill.readthedocs.io/)
- **Base data path**: `/neuro/labs/grantlab/research/MRI_processing/` — FNNDSC lab infrastructure, not portable
- **FreeSurfer LUT**: `assets/FreeSurferColorLUT.txt` used for brain region color mapping
