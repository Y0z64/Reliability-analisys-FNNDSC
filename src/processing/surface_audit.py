#!/usr/bin/env python3
"""
Surface Pipeline Audit & Data Extraction
=========================================
Checks which subjects/splits succeeded or failed for:
  1. CP surface pipeline (surfaces/)
  2. SP surface pipeline (default_surfaces/)

Extracts surface measurements (Area_Depth_aMC_Thk.txt, GI_info_final.txt),
WM surface file info, and SP surface areas (computed from `*.innersp.gii`
meshes — sum of triangle face areas, in mm²) into a consolidated DataFrame.

Usage:
    # Single subject test
    python surface_audit.py --subject FCB145 --session 2019.03.20-032Y-MR_EI_Fetal_Neuro-29197

    # Full batch
    python surface_audit.py --batch --subjects subject.csv

    # Skip expensive checks (file size computation)
    python surface_audit.py --batch --subjects subject.csv --skip-filesize

    # Skip SP surface area computation (slightly faster — avoids GIFTI parsing)
    python surface_audit.py --batch --subjects subject.csv --skip-sp-area
"""

import pandas as pd
import numpy as np
import os
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

nib = None  # populated below if nibabel is available
try:
    import nibabel as nib  # noqa: F811
except ImportError:
    pass

# ============================================================
# CONFIGURATION
# ============================================================
BASE_PATH = "/neuro/labs/grantlab/research/MRI_processing/seungyoon.jeong/2025/Reliability/TEST/"
SPLITS = ["S1", "S2", "S3", "S4"]

# Expected files per pipeline
CP_SURFACE_CRITICAL = [
    "lh.wm_81920.obj",
    "rh.wm_81920.obj",
    "lh.pial_81920.obj",
    "rh.pial_81920.obj",
]
CP_SURFACE_MEASUREMENTS = [
    "Area_Depth_aMC_Thk.txt",
    "GI_info_final.txt",
]
CP_SURFACE_VERTEX_MAPS = [
    "lh.smoothwm.native_81920.thk.s5",
    "rh.smoothwm.native_81920.thk.s5",
    "lh.smoothwm.native_81920.area.s5",
    "rh.smoothwm.native_81920.area.s5",
    "lh.smoothwm.native_81920.depth.s5",
    "rh.smoothwm.native_81920.depth.s5",
    "lh.smoothwm.mni_81920.mc.s5",
    "rh.smoothwm.mni_81920.mc.s5",
]
CP_SURFACE_TEMPLATES = ["template-29", "template-31", "template-adult"]

SP_SURFACE_CRITICAL = [
    "lh.innersp.obj",
    "rh.innersp.obj",
]
SP_SURFACE_OPTIONAL = [
    "lh.wm.obj",
    "rh.wm.obj",
    "lh.wm.taubin100.obj",
    "rh.wm.taubin100.obj",
    "lh.innersp.gii",
    "rh.innersp.gii",
]


# ============================================================
# LIGHTWEIGHT: File existence checks (fast, no I/O beyond stat)
# ============================================================


def check_cp_pipeline(surfaces_dir: str) -> Dict:
    """Check CP surface pipeline status. FAST — only checks file existence."""
    result = {
        "cp_dir_exists": os.path.isdir(surfaces_dir),
        "cp_status": "missing_dir",
        "cp_critical_missing": [],
        "cp_measurements_missing": [],
        "cp_vertex_maps_missing": [],
        "cp_templates_missing": [],
        "cp_n_files_total": 0,
    }

    if not result["cp_dir_exists"]:
        return result

    # Count total files
    result["cp_n_files_total"] = len(
        [
            f
            for f in os.listdir(surfaces_dir)
            if os.path.isfile(os.path.join(surfaces_dir, f))
        ]
    )

    # Check critical surface files
    for f in CP_SURFACE_CRITICAL:
        if not os.path.isfile(os.path.join(surfaces_dir, f)):
            result["cp_critical_missing"].append(f)

    # Check measurement files
    for f in CP_SURFACE_MEASUREMENTS:
        if not os.path.isfile(os.path.join(surfaces_dir, f)):
            result["cp_measurements_missing"].append(f)

    # Check vertex maps
    for f in CP_SURFACE_VERTEX_MAPS:
        if not os.path.isfile(os.path.join(surfaces_dir, f)):
            result["cp_vertex_maps_missing"].append(f)

    # Check template directories
    for t in CP_SURFACE_TEMPLATES:
        if not os.path.isdir(os.path.join(surfaces_dir, t)):
            result["cp_templates_missing"].append(t)

    # Determine overall status
    if result["cp_critical_missing"]:
        if result["cp_n_files_total"] == 0:
            result["cp_status"] = "empty_dir"
        else:
            result["cp_status"] = "partial_fail"
    elif result["cp_measurements_missing"]:
        result["cp_status"] = "surfaces_ok_no_measures"
    elif result["cp_vertex_maps_missing"] or result["cp_templates_missing"]:
        result["cp_status"] = "measures_ok_incomplete"
    else:
        result["cp_status"] = "complete"

    return result


def check_sp_pipeline(sp_dir: str) -> Dict:
    """Check SP surface pipeline status. FAST — only checks file existence."""
    result = {
        "sp_dir_exists": os.path.isdir(sp_dir),
        "sp_status": "missing_dir",
        "sp_critical_missing": [],
        "sp_optional_missing": [],
        "sp_has_log": False,
        "sp_log_status": None,
        "sp_n_files_total": 0,
    }

    if not result["sp_dir_exists"]:
        return result

    result["sp_n_files_total"] = len(
        [f for f in os.listdir(sp_dir) if os.path.isfile(os.path.join(sp_dir, f))]
    )

    for f in SP_SURFACE_CRITICAL:
        if not os.path.isfile(os.path.join(sp_dir, f)):
            result["sp_critical_missing"].append(f)

    for f in SP_SURFACE_OPTIONAL:
        if not os.path.isfile(os.path.join(sp_dir, f)):
            result["sp_optional_missing"].append(f)

    # Check pipeline log if it exists
    log_files = glob.glob(os.path.join(sp_dir, "*_pipeline.log"))
    if log_files:
        result["sp_has_log"] = True
        result["sp_log_status"] = _parse_sp_log_status(log_files[0])

    # Check error log in parent directory
    parent_dir = os.path.dirname(sp_dir)
    error_logs = glob.glob(os.path.join(parent_dir, "*_error.log"))
    if error_logs:
        result["sp_error_log"] = error_logs[0]

    # Determine status
    if result["sp_critical_missing"]:
        if result["sp_n_files_total"] == 0:
            result["sp_status"] = "empty_dir"
        else:
            result["sp_status"] = "partial_fail"
    elif result["sp_optional_missing"]:
        result["sp_status"] = "complete_missing_optional"
    else:
        result["sp_status"] = "complete"

    return result


def _parse_sp_log_status(log_path: str) -> str:
    """Parse SP pipeline log to determine failure point. Reads only last 2KB."""
    try:
        with open(log_path, "r") as f:
            # Seek to last 2KB
            f.seek(0, 2)
            fsize = f.tell()
            f.seek(max(0, fsize - 2048))
            tail = f.read()

        if "SUCCESS: All files created" in tail:
            return "success"
        elif "FATAL ERROR" in tail:
            # Extract which step failed
            for line in tail.split("\n"):
                if "FATAL ERROR" in line:
                    return f"failed: {line.strip()[:120]}"
            return "failed: unknown step"
        elif "Pipeline completed" in tail:
            if "FAILED: Missing files" in tail:
                return "completed_with_missing_files"
            return "completed"
        else:
            return "incomplete_log"
    except Exception as e:
        return f"log_read_error: {e}"


def _parse_error_log_snippet(error_log_path: str, n_chars: int = 500) -> str:
    """Read last N chars of an error log. CHEAP."""
    try:
        with open(error_log_path, "r") as f:
            f.seek(0, 2)
            fsize = f.tell()
            f.seek(max(0, fsize - n_chars))
            return f.read().strip()
    except:
        return ""


# ============================================================
# MODERATE COST: Read measurement files
# ============================================================


def read_adt_file(surfaces_dir: str) -> Optional[Dict]:
    """Read Area_Depth_aMC_Thk.txt — 8 values from a single line."""
    fpath = os.path.join(surfaces_dir, "Area_Depth_aMC_Thk.txt")
    if not os.path.isfile(fpath):
        return None
    try:
        with open(fpath, "r") as f:
            vals = f.readline().strip().split()
        if len(vals) != 8:
            return {"adt_error": f"expected 8 values, got {len(vals)}"}
        keys = [
            "lh_area_sum",
            "rh_area_sum",
            "lh_depth_mean",
            "rh_depth_mean",
            "lh_curv_absmean",
            "rh_curv_absmean",
            "lh_thk_mean",
            "rh_thk_mean",
        ]
        return {k: float(v) for k, v in zip(keys, vals)}
    except Exception as e:
        return {"adt_error": str(e)}


def read_gi_file(surfaces_dir: str) -> Optional[Dict]:
    """Read GI_info_final.txt — gyrification index."""
    fpath = os.path.join(surfaces_dir, "GI_info_final.txt")
    if not os.path.isfile(fpath):
        return None
    try:
        with open(fpath, "r") as f:
            content = f.read().strip()
        # GI file format varies — try to parse key-value or single values
        # Common formats: "lh_GI rh_GI" or key=value lines
        vals = content.split()
        if len(vals) == 2:
            return {"lh_GI": float(vals[0]), "rh_GI": float(vals[1])}
        elif len(vals) == 1:
            return {"GI": float(vals[0])}
        else:
            # Try parsing as key-value
            result = {}
            for line in content.split("\n"):
                line = line.strip()
                if "=" in line:
                    k, v = line.split("=", 1)
                    try:
                        result[k.strip()] = float(v.strip())
                    except ValueError:
                        result[k.strip()] = v.strip()
                elif ":" in line:
                    k, v = line.split(":", 1)
                    try:
                        result[k.strip()] = float(v.strip())
                    except ValueError:
                        result[k.strip()] = v.strip()
            return result if result else {"gi_raw": content[:200]}
    except Exception as e:
        return {"gi_error": str(e)}


# ============================================================
# MODERATE COST: SP surface area from GIFTI meshes
# ============================================================
# The CIVET-style WM area in `Area_Depth_aMC_Thk.txt` is computed from the
# Voronoi areas of the smoothed WM mesh (`lh.smoothwm.native_81920.area.s5`),
# summed via `depth_potential -area_voronoi`. The SP pipeline does not produce
# an equivalent aggregated text file, so we compute the SP area directly from
# the triangulated `*.innersp.gii` mesh (sum of triangle face areas). Vertex
# coordinates in these GIFTIs are in millimeters (native scanner space), so the
# returned values are in mm². This is mathematically equivalent to running
# `depth_potential -area_voronoi` on the as-extracted SP mesh — it differs from
# the WM `lh_area_sum` only because that one operates on a smoothed mesh.


def compute_mesh_area_from_gii(gii_path: str) -> Optional[float]:
    """Sum of triangle face areas (mm²) for a GIFTI surface mesh.

    Returns None if nibabel is unavailable, the file is missing, or it cannot
    be parsed (e.g. unexpected DataArray layout).
    """
    if nib is None:
        return None
    if not os.path.isfile(gii_path):
        return None
    try:
        gii = nib.load(gii_path)
        verts = None
        tris = None
        for arr in gii.darrays:  # ty: ignore[unresolved-attribute]
            # GIFTI intent codes: 1008 = NIFTI_INTENT_POINTSET, 1009 = TRIANGLE
            intent = getattr(arr, "intent", None)
            if intent == 1008 or (verts is None and arr.data.ndim == 2 and arr.data.shape[1] == 3 and np.issubdtype(arr.data.dtype, np.floating)):
                verts = arr.data
            elif intent == 1009 or (tris is None and arr.data.ndim == 2 and arr.data.shape[1] == 3 and np.issubdtype(arr.data.dtype, np.integer)):
                tris = arr.data
        if verts is None or tris is None:
            return None
        p0 = verts[tris[:, 0]]
        p1 = verts[tris[:, 1]]
        p2 = verts[tris[:, 2]]
        areas = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
        return float(areas.sum())
    except Exception:
        return None


def read_sp_area_func(sp_dir: str) -> Dict:
    """Compute SP surface area sums from `lh.innersp.gii` and `rh.innersp.gii`.

    Returns a dict with `lh_sp_area_sum`, `rh_sp_area_sum`, and `total_sp_area`
    (all in mm²). Missing or unreadable hemispheres yield None for that field
    and total_sp_area is only set when both hemispheres are available.
    """
    lh = compute_mesh_area_from_gii(os.path.join(sp_dir, "lh.innersp.gii"))
    rh = compute_mesh_area_from_gii(os.path.join(sp_dir, "rh.innersp.gii"))
    out = {"lh_sp_area_sum": lh, "rh_sp_area_sum": rh}
    if lh is not None and rh is not None:
        out["total_sp_area"] = lh + rh
    else:
        out["total_sp_area"] = None
    return out


# ============================================================
# EXPENSIVE: File sizes, vertex counts (disable with flags)
# ============================================================


def get_wm_surface_info(surfaces_dir: str) -> Dict:
    """Get WM surface file paths and sizes. MODERATE cost (stat calls)."""
    info = {}
    for hemi in ["lh", "rh"]:
        for ext in ["obj", "asc", "gii"]:
            fname = f"{hemi}.wm_81920.{ext}"
            fpath = os.path.join(surfaces_dir, fname)
            if os.path.isfile(fpath):
                info[f"{hemi}_wm_{ext}_size_mb"] = os.path.getsize(fpath) / (
                    1024 * 1024
                )
                info[f"{hemi}_wm_{ext}_path"] = fpath
            else:
                info[f"{hemi}_wm_{ext}_size_mb"] = None
    return info


# ============================================================
# SINGLE SUBJECT AUDIT
# ============================================================


def audit_subject_split(
    subject_id: str,
    session_id: str,
    split: str,
    base_path: str = BASE_PATH,
    read_measurements: bool = True,
    read_filesize: bool = False,
    read_sp_area: bool = True,
) -> Dict:
    """
    Audit a single subject/split combination.

    Args:
        subject_id, session_id, split: identifiers
        base_path: root directory
        read_measurements: if True, parse ADT and GI files (moderate cost)
        read_filesize: if True, compute file sizes (slightly more I/O)

    Returns:
        Dict with all audit fields
    """
    split_path = os.path.join(base_path, subject_id, session_id, split)
    surfaces_dir = os.path.join(split_path, "surfaces")
    sp_dir = os.path.join(split_path, "default_surfaces")

    record = {
        "subject_id": subject_id,
        "session_id": session_id,
        "split": split,
        "split_dir_exists": os.path.isdir(split_path),
    }

    if not record["split_dir_exists"]:
        record["cp_status"] = "no_split_dir"
        record["sp_status"] = "no_split_dir"
        return record

    # --- CP pipeline check (fast) ---
    cp_info = check_cp_pipeline(surfaces_dir)
    record.update(cp_info)

    # --- SP pipeline check (fast) ---
    sp_info = check_sp_pipeline(sp_dir)
    record.update(sp_info)

    # --- SP surface area from GIFTI meshes (moderate cost, optional) ---
    if read_sp_area and sp_info.get("sp_dir_exists"):
        record.update(read_sp_area_func(sp_dir))

    # --- Read measurements (moderate cost, optional) ---
    if read_measurements and cp_info["cp_status"] not in ("missing_dir", "empty_dir"):
        adt = read_adt_file(surfaces_dir)
        if adt:
            record.update(adt)

        gi = read_gi_file(surfaces_dir)
        if gi:
            record.update(gi)

    # --- File sizes (optional) ---
    if read_filesize and cp_info["cp_dir_exists"]:
        wm_info = get_wm_surface_info(surfaces_dir)
        record.update(wm_info)

    # --- Error log snippet for failures ---
    if sp_info.get("sp_status") in ("partial_fail", "empty_dir", "missing_dir"):
        error_log = sp_info.get("sp_error_log")
        if error_log and os.path.isfile(error_log):
            record["sp_error_snippet"] = _parse_error_log_snippet(error_log)

    return record


# ============================================================
# BATCH AUDIT
# ============================================================


def audit_all_subjects(
    subjects_csv: str,
    base_path: str = BASE_PATH,
    read_measurements: bool = True,
    read_filesize: bool = False,
    read_sp_area: bool = True,
) -> pd.DataFrame:
    """
    Audit all subjects × splits from a CSV file.

    Returns a long-format DataFrame with one row per subject/split.
    """
    subjects = pd.read_csv(subjects_csv)

    records = []
    total = len(subjects) * len(SPLITS)

    for idx, row in subjects.iterrows():
        subject_id = str(row["subject_id"])
        session_id = str(row["session_id"])
        ga = row.get("GA", None)

        for split in SPLITS:
            record = audit_subject_split(
                subject_id,
                session_id,
                split,
                base_path=base_path,
                read_measurements=read_measurements,
                read_filesize=read_filesize,
                read_sp_area=read_sp_area,
            )
            record["GA"] = ga
            records.append(record)

            n = len(records)
            status_str = (
                f"CP:{record.get('cp_status','?')} SP:{record.get('sp_status','?')}"
            )
            print(f"[{n}/{total}] {subject_id}/{split}: {status_str}")

    df = pd.DataFrame(records)
    return df


# ============================================================
# SUMMARY REPORTING
# ============================================================


def print_summary(df: pd.DataFrame):
    """Print a readable summary of audit results."""

    print("\n" + "=" * 80)
    print("CP SURFACE PIPELINE STATUS")
    print("=" * 80)
    cp_counts = df["cp_status"].value_counts()
    for status, count in cp_counts.items():
        print(f"  {status:35s}: {count:3d} ({count/len(df)*100:.0f}%)")

    print(f"\n  Total split-runs: {len(df)}")
    print(f"  Unique subjects: {df['subject_id'].nunique()}")

    # Subjects where ALL splits failed
    cp_by_subj = df.groupby("subject_id")["cp_status"].apply(
        lambda x: (
            "all_fail"
            if all(s != "complete" for s in x)
            else "all_pass" if all(s == "complete" for s in x) else "mixed"
        )
    )
    print(f"\n  Subjects all-pass:  {(cp_by_subj == 'all_pass').sum()}")
    print(f"  Subjects mixed:     {(cp_by_subj == 'mixed').sum()}")
    print(f"  Subjects all-fail:  {(cp_by_subj == 'all_fail').sum()}")

    if (cp_by_subj == "all_fail").any():
        print(f"  Failed subjects: {list(cp_by_subj[cp_by_subj == 'all_fail'].index)}")

    print("\n" + "=" * 80)
    print("SP SURFACE PIPELINE STATUS")
    print("=" * 80)
    sp_counts = df["sp_status"].value_counts()
    for status, count in sp_counts.items():
        print(f"  {status:35s}: {count:3d} ({count/len(df)*100:.0f}%)")

    sp_by_subj = df.groupby("subject_id")["sp_status"].apply(
        lambda x: (
            "all_fail"
            if all(s not in ("complete", "complete_missing_optional") for s in x)
            else (
                "all_pass"
                if all(s in ("complete", "complete_missing_optional") for s in x)
                else "mixed"
            )
        )
    )
    print(f"\n  Subjects all-pass:  {(sp_by_subj == 'all_pass').sum()}")
    print(f"  Subjects mixed:     {(sp_by_subj == 'mixed').sum()}")
    print(f"  Subjects all-fail:  {(sp_by_subj == 'all_fail').sum()}")

    if (sp_by_subj == "all_fail").any():
        print(f"  Failed subjects: {list(sp_by_subj[sp_by_subj == 'all_fail'].index)}")

    # Measurement availability
    if "lh_thk_mean" in df.columns:
        print("\n" + "=" * 80)
        print("SURFACE MEASUREMENTS AVAILABILITY")
        print("=" * 80)
        measure_cols = [
            "lh_area_sum",
            "lh_depth_mean",
            "lh_curv_absmean",
            "lh_thk_mean",
            "lh_sp_area_sum",
            "rh_sp_area_sum",
            "total_sp_area",
        ]
        for col in measure_cols:
            if col in df.columns:
                n_valid = df[col].notna().sum()
                print(f"  {col:25s}: {n_valid}/{len(df)} available")

    # Failures detail
    failures = df[
        (df["cp_status"] != "complete")
        | (~df["sp_status"].isin(["complete", "complete_missing_optional"]))
    ]
    if not failures.empty:
        print("\n" + "=" * 80)
        print("FAILURE DETAILS")
        print("=" * 80)
        for _, row in failures.iterrows():
            if row["cp_status"] != "complete":
                missing = row.get("cp_critical_missing", [])
                print(
                    f"  CP FAIL: {row['subject_id']}/{row['split']} — {row['cp_status']}"
                    f"{' missing: ' + str(missing) if missing else ''}"
                )
            if row.get("sp_status") not in ("complete", "complete_missing_optional"):
                print(
                    f"  SP FAIL: {row['subject_id']}/{row['split']} — {row.get('sp_status', '?')}"
                    f"{' log: ' + str(row.get('sp_log_status', '')) if row.get('sp_log_status') else ''}"
                )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit surface pipeline results")
    parser.add_argument("--subject", type=str, help="Single subject ID")
    parser.add_argument(
        "--session", type=str, help="Session ID (required with --subject)"
    )
    parser.add_argument("--batch", action="store_true", help="Run batch audit")
    parser.add_argument(
        "--subjects", type=str, default="../data/subject.csv", help="Path to subjects CSV"
    )
    parser.add_argument(
        "--base-path", type=str, default=BASE_PATH, help="Base directory"
    )
    parser.add_argument(
        "--skip-measurements", action="store_true", help="Skip reading ADT/GI files"
    )
    parser.add_argument(
        "--skip-filesize", action="store_true", help="Skip file size computation"
    )
    parser.add_argument(
        "--skip-sp-area",
        action="store_true",
        help="Skip SP surface area computation from .innersp.gii meshes",
    )
    parser.add_argument(
        "--output", type=str, default="surface_audit.csv", help="Output CSV path"
    )

    args = parser.parse_args()

    if args.subject:
        if not args.session:
            parser.error("--session required with --subject")

        print(f"Auditing single subject: {args.subject}")
        records = []
        for split in SPLITS:
            r = audit_subject_split(
                args.subject,
                args.session,
                split,
                base_path=args.base_path,
                read_measurements=not args.skip_measurements,
                read_filesize=not args.skip_filesize,
                read_sp_area=not args.skip_sp_area,
            )
            records.append(r)
            print(f"  {split}: CP={r.get('cp_status')} SP={r.get('sp_status')}")

        df = pd.DataFrame(records)
        df.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")
        print_summary(df)

    elif args.batch:
        df = audit_all_subjects(
            args.subjects,
            base_path=args.base_path,
            read_measurements=not args.skip_measurements,
            read_filesize=not args.skip_filesize,
            read_sp_area=not args.skip_sp_area,
        )
        df.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")
        print_summary(df)

    else:
        parser.print_help()
