# Environment

## Local / analysis (stage 4 — no cluster needed)

Python 3.12, managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync                  # install from uv.lock
uv run jupyter lab       # notebooks
uv run ruff check .      # lint
uv run ruff format .     # format
uv run ty check          # type check
```

Remote JupyterLab on the cluster:

```bash
uv run jupyter lab --port 8888 --IdentityProvider.token MY_TOKEN --ip 0.0.0.0
```

## Cluster (stages 1–3)

The processing scripts do not use the uv environment. They need the lab envs:

```bash
# general MRI tooling
micromamba activate -p /neuro/labs/grantlab/research/MRI_processing/environment

# run_SP_prediction.py only — does NOT work on el-jobo
source ~/miniconda3/bin/activate \
  /neuro/labs/grantlab/research/MRI_processing/milton.candela/code/brain_age/conda_env/original_env/
```

`extract_SP_surface.py` and `run_SP_batch.py` have a shebang pointing at
`/neuro/users/mri.team/packages/env_MRI_team`.

## Hardcoded cluster paths

None of these are portable. Grep for `/neuro/` before moving this project anywhere.

**Data**

| Path | Used by |
|---|---|
| `…/seungyoon.jeong/2025/Reliability/TEST/` | the data root — `CNR.py`, `Reliability.ipynb`, `surface_audit.py`, `thickness_audit.py`, `generate_native_surfaces.py`, `run_SP_batch.py`, `run_SP_prediction.py`, `regional_thickness_demo.ipynb` |
| `…/seungyoon.jeong/Data/Placenta_protocol` | `run_SP_prediction.py` input |
| `…/andrea.gondova/DerivedData/subjects` | `run_SP_prediction.py` output |
| `/neuro/labs/grantlab/research/andrea.gondova/21_annotation.csv` | `regional_thickness_demo.ipynb` (note: **not** under `MRI_processing`) |

**Templates and reference**

| Path | Used by |
|---|---|
| `…/yair.beltran/Reliability/FreeSurferColorLUT.txt` | `functions/helpers.py`, `Reliability.ipynb`. A local copy is in `reference/`. |
| `/neuro/users/mri.team/fetal_mri/Surface_template/template-29/old/lh.fetal21.1D.dset` | `regional_thickness_demo.ipynb` |

**Tooling and models**

| Path | Used by |
|---|---|
| `…/yair.beltran/Reliability/extract_SP_surface.py` | invoked by `run_SP_batch.py` — a **copy** of `src/processing/extract_SP_surface.py`; edits here must be copied there |
| `…/andrea.gondova/Scripts/CP_SP_coevolution/SP_surface_extraction/SP_singularities` | `extract_SP_surface.py` containers |
| `…/andrea.gondova/Scripts/CP_SP_coevolution/surface_processing/convert_obj2gii.sh` | `extract_SP_surface.py` |
| `…/milton.candela/highres_subplate/predict_sp.py` | `run_SP_prediction.py` |
| `…/milton.candela/fetal_subplate/models/C120` | `run_SP_prediction.py` — the subplate model |

(`…` = `/neuro/labs/grantlab/research/MRI_processing`)
