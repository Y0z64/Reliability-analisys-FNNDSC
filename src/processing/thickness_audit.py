"""
Subplate thickness audit & computation.

For each subject/session/split, checks whether SP pipeline produced both
`{lh,rh}.wm..obj` (outer SP boundary, == WM surface) and `{lh,rh}.innersp.obj`
(inner SP boundary). When both exist for a hemisphere, computes per-vertex
Euclidean distance (the innersp mesh is deformed from the WM mesh, so vertex i
in one corresponds to vertex i in the other) and aggregates to mean thickness.

Then appends the split-pair columns the paper notebooks model on -- split_pair,
thk_diff, thk_mean, apd_thk -- plus qa_mean and stack_count joined from
data/raw_subject_data.csv. Those six columns used to be added by hand outside
this script; `add_pair_columns` below reproduces the committed values exactly.

Outputs: data/thickness_audit.csv
Requires FNNDSC cluster access to BASE_PATH.

Run:  uv run python src/processing/thickness_audit.py
"""

import os
import re
from typing import Any

import numpy as np
import pandas as pd

BASE_PATH = "/neuro/labs/grantlab/research/MRI_processing/seungyoon.jeong/2025/Reliability/TEST/"
SPLITS = ["S1", "S2", "S3", "S4"]
HEMIS = ["lh", "rh"]
GA_MAX = 32.0  # exclude subjects with GA > 32 weeks
SUBJECT_CSV = os.path.join(os.path.dirname(__file__), "..", "..", "data", "subject.csv")
RAW_SUBJECT_CSV = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw_subject_data.csv")
OUT_CSV = os.path.join(os.path.dirname(__file__), "..", "..", "data", "thickness_audit.csv")


def read_native_xfm(path: str) -> np.ndarray | None:
    """Parse an MNI Transform File and return the 3x4 linear matrix, or None.

    For this project's xfms the matrix is diagonal scaling (no rotation/trans),
    but we parse the full matrix and apply it generically so the code remains
    correct if that ever changes.
    """
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        txt = f.read()
    m = re.search(r"Linear_Transform\s*=\s*([-\d\.\seE+]+);", txt)
    if not m:
        return None
    nums = list(map(float, m.group(1).split()))
    if len(nums) != 12:
        return None
    return np.array(nums, dtype=np.float64).reshape(3, 4)


def apply_xfm(verts: np.ndarray, xfm: np.ndarray) -> np.ndarray:
    """Apply a 3x4 affine (R|t) to an (N,3) vertex array."""
    return verts @ xfm[:, :3].T + xfm[:, 3]


def read_obj_vertices(path: str) -> np.ndarray:
    """Read vertex coordinates from an MNI `.obj` surface file.

    Header line: `P <a> <d> <s> <sh> <t> <n_vertices>` followed by n_vertices
    lines of `x y z`. Returns array of shape (n_vertices, 3).
    """
    with open(path) as f:
        header = f.readline().split()
        n = int(header[-1])
        verts = np.empty((n, 3), dtype=np.float64)
        for i in range(n):
            verts[i] = [float(x) for x in f.readline().split()]
    return verts


def hemisphere_thickness(wm_path: str, innersp_path: str,
                         xfm: np.ndarray | None) -> dict:
    wm = read_obj_vertices(wm_path)
    inn = read_obj_vertices(innersp_path)
    if wm.shape != inn.shape:
        return {"n_vertices": np.nan, "mean": np.nan, "median": np.nan,
                "std": np.nan, "p95": np.nan, "max": np.nan,
                "error": f"shape_mismatch wm={wm.shape} innersp={inn.shape}"}
    if xfm is not None:
        wm = apply_xfm(wm, xfm)
        inn = apply_xfm(inn, xfm)
    d = np.linalg.norm(wm - inn, axis=1)
    return {
        "n_vertices": int(d.size),
        "mean": float(d.mean()),
        "median": float(np.median(d)),
        "std": float(d.std()),
        "p95": float(np.percentile(d, 95)),
        "max": float(d.max()),
        "error": "",
    }


def audit_split(subject_id: str, session_id: str, split: str) -> dict[str, Any]:
    split_dir = os.path.join(BASE_PATH, subject_id, session_id, split)
    sp_dir = os.path.join(split_dir, "default_surfaces")
    xfm_path = os.path.join(split_dir, "recon_segmentation", "recon_native.xfm")
    xfm = read_native_xfm(xfm_path)

    record: dict[str, Any] = {
        "subject_id": subject_id,
        "session_id": session_id,
        "split": split,
        "sp_dir_exists": os.path.isdir(sp_dir),
        "xfm_exists": xfm is not None,
    }
    if xfm is not None:
        record["xfm_scale_x"] = float(xfm[0, 0])
        record["xfm_scale_y"] = float(xfm[1, 1])
        record["xfm_scale_z"] = float(xfm[2, 2])

    per_hemi_means = []
    per_hemi_nverts = []
    for hemi in HEMIS:
        wm = os.path.join(sp_dir, f"{hemi}.wm..obj")
        inn = os.path.join(sp_dir, f"{hemi}.innersp.obj")
        record[f"{hemi}_wm_exists"] = os.path.isfile(wm)
        record[f"{hemi}_innersp_exists"] = os.path.isfile(inn)

        if record[f"{hemi}_wm_exists"] and record[f"{hemi}_innersp_exists"]:
            try:
                stats = hemisphere_thickness(wm, inn, xfm)
            except Exception as e:
                stats = {"n_vertices": np.nan, "mean": np.nan, "median": np.nan,
                         "std": np.nan, "p95": np.nan, "max": np.nan,
                         "error": f"read_failed: {e}"}
            for k, v in stats.items():
                record[f"{hemi}_thk_{k}"] = v
            if not np.isnan(stats["mean"]):
                per_hemi_means.append(stats["mean"])
                per_hemi_nverts.append(stats["n_vertices"])
        else:
            for k in ("n_vertices", "mean", "median", "std", "p95", "max"):
                record[f"{hemi}_thk_{k}"] = np.nan
            record[f"{hemi}_thk_error"] = "missing_input"

    if per_hemi_means:
        weights = np.array(per_hemi_nverts, dtype=float)
        record["wholebrain_thk_mean"] = float(
            np.average(per_hemi_means, weights=weights)
        )
        record["has_thickness"] = True
    else:
        record["wholebrain_thk_mean"] = np.nan
        record["has_thickness"] = False

    return record


def add_pair_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the split-pair thickness columns consumed by the paper notebooks.

    For each pair (S1-S2, S3-S4) computed from `wholebrain_thk_mean`:
        thk_diff = |a - b|
        thk_mean = (a + b) / 2
        apd_thk  = thk_diff / thk_mean * 100      (absolute percent difference)

    The pair-level values are written onto *both* split rows of the pair, which
    is the shape paper_models*.ipynb expects. qa_mean and stack_count are joined
    from raw_subject_data.csv (QA12_mean for S1-S2, QA34_mean for S3-S4).
    """
    pairs = {"S1": "S1-S2", "S2": "S1-S2", "S3": "S3-S4", "S4": "S3-S4"}
    df = df.copy()
    df["split_pair"] = df["split"].map(pairs)

    records = []
    for (subj, sess, pair), g in df.groupby(["subject_id", "session_id", "split_pair"]):
        vals = g["wholebrain_thk_mean"].dropna()
        if len(vals) != 2:
            continue
        a, b = vals.iloc[0], vals.iloc[1]
        mean = (a + b) / 2
        records.append({
            "subject_id": subj, "session_id": sess, "split_pair": pair,
            "thk_diff": abs(a - b),
            "thk_mean": mean,
            "apd_thk": abs(a - b) / mean * 100 if mean else np.nan,
        })

    if records:
        df = df.merge(pd.DataFrame(records),
                      on=["subject_id", "session_id", "split_pair"], how="left")
    else:
        for col in ("thk_diff", "thk_mean", "apd_thk"):
            df[col] = np.nan

    if os.path.isfile(RAW_SUBJECT_CSV):
        raw = pd.read_csv(RAW_SUBJECT_CSV).astype({"subject_id": str})
        qa_map = {"S1-S2": "QA12_mean", "S3-S4": "QA34_mean"}
        cols = ["subject_id"] + [c for c in ("QA12_mean", "QA34_mean", "stack_count")
                                 if c in raw.columns]
        df = df.merge(raw[cols], on="subject_id", how="left")
        df["qa_mean"] = df.apply(
            lambda r: r.get(qa_map.get(r["split_pair"], ""), np.nan), axis=1
        )
        df = df.drop(columns=[c for c in ("QA12_mean", "QA34_mean") if c in df.columns])
    else:
        print(f"WARNING: {RAW_SUBJECT_CSV} missing; qa_mean/stack_count not added")

    return df


def main():
    subjects = pd.read_csv(SUBJECT_CSV)
    n_before = len(subjects)
    subjects = subjects[subjects["GA"] <= GA_MAX].reset_index(drop=True)
    print(f"Subjects after GA<={GA_MAX} filter: {len(subjects)} (dropped {n_before - len(subjects)})")
    rows = []
    for _, r in subjects.iterrows():
        for split in SPLITS:
            rows.append(audit_split(str(r["subject_id"]), str(r["session_id"]), split))

    df = pd.DataFrame(rows)
    df = df.merge(subjects[["subject_id", "session_id", "GA"]].astype({"subject_id": str}),
                  on=["subject_id", "session_id"], how="left")

    drop_cols = [
        "sp_dir_exists", "xfm_exists",
        "lh_wm_exists", "lh_innersp_exists", "lh_thk_error",
        "rh_wm_exists", "rh_innersp_exists", "rh_thk_error",
        "has_thickness",
    ]
    df_out = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df_out = add_pair_columns(df_out)
    df_out.to_csv(OUT_CSV, index=False)

    n_total = len(df)
    n_with = int(df["has_thickness"].sum())
    print(f"Wrote {OUT_CSV}")
    print(f"Total subject/split rows : {n_total}")
    print(f"With thickness computed  : {n_with}")
    print(f"With both hemispheres    : "
          f"{int(((df['lh_wm_exists'] & df['lh_innersp_exists']) & (df['rh_wm_exists'] & df['rh_innersp_exists'])).sum())}")
    print(f"With only one hemisphere : "
          f"{int(((df['lh_wm_exists'] & df['lh_innersp_exists']) ^ (df['rh_wm_exists'] & df['rh_innersp_exists'])).sum())}")
    print("\nPer-split coverage:")
    print(df.groupby("split")["has_thickness"].sum().to_string())
    print("\nWhole-brain mean thickness summary (mm):")
    print(df["wholebrain_thk_mean"].describe().to_string())


if __name__ == "__main__":
    main()
