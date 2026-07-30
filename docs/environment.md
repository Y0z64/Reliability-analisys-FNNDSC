# Environment

## Local

Python 3.12, managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync                  # install from uv.lock
uv run jupyter lab       # notebooks
uv run ruff check .      # lint
uv run ruff format .     # format
uv run ty check          # type check
```

Open Jupyter Lab:

```bash
uv run jupyter lab
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

## Hardcoded lab paths

Environment is created with the intention of being able to run locally as long as there is access to the `/neuro` directory. Meaning this project could be cloned or moved but lab cluster has to be mounted (using `sshfs` or similar) if needed.

 The following paths are hardcoded in the code and notebooks, point to locations outside the repository found on lab cluster: 

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

## Reprocessing surfaces

Run on a node with apptainer/singularity (hanyang; el-jobo does not have it).
The host also needs GNU `parallel` (`apt install parallel`).

```bash
# 1. Shell env: FreeSurfer, FSL, and the CIVET quarantine that provides
#    adapt_object_mesh and vertstats_math
source /neuro/users/mri.team/packages/env_MRI_team

cd src/processing

# 2. Segmentations, only if missing. Needs milton's env (py3.8 + TF 2.10).
#    Do not run this one under uv.
source ~/miniconda3/bin/activate /neuro/labs/grantlab/research/MRI_processing/milton.candela/code/brain_age/conda_env/original_env/
python3 run_SP_prediction.py --subjects ../../data/subject.csv

# 3. Surfaces, all four splits per subject. uv supplies pandas and a `python`
#    on PATH, which steps 7/11/13 of the extractor need.
uv run python run_SP_batch.py --subjects ../../data/subject.csv --max_workers 16

# 4. Native-space surfaces and thickness
uv run python generate_native_surfaces.py

# 5. Audits
uv run python surface_audit.py --batch
uv run python thickness_audit.py
```

Cap `--max_workers` at roughly `nproc/4`: every worker spawns
`extract_cp -J $(nproc)` and `gifit --threads $(nproc)`.

`--force_rerun` runs `rm -rf default_surfaces/*`, deleting the `*.native.obj`
and `*.innersp.native.thk` written in step 4. Re-run step 4 after using it.

Failures land in `src/processing/failed.csv` and in per-split
`{subject}/{session}/{split}/{subject}_{session}_{split}_error.log`.

Step 3 requires `{subject}_{session}_nuc_deep_subplate_dilate_mc.nii` in each
split's `segmentations/`. There is no fallback to the non-`_mc` name; a missing
file is reported as `Data not found` and the split is skipped.

## Subplate model prediction

SP model requires loading Milton's environment (previous intern):

```source ~/tools/miniconda3/bin/activate /neuro/labs/grantlab/research/MRI_processing/milton.candela/code/brain_age/conda_env/original_env/```

Then prediction is done by:
```/neuro/labs/grantlab/research/MRI_processing/andrea.gondova/Scripts/CP_SP_coevolution/helper_scripts/run_SP_predicton.py --subjects [subject.csv]```

> Expects .csv file with subject_id, session_id columns per row