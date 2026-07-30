#!/usr/bin/env python3
"""
Image Quality Metrics for Fetal Brain Segmentation Reliability Analysis
========================================================================

Consolidated module combining the former image_quality_metrics.py (core SNR/CNR
computation), image_qam_v2.py (multi-split processing) and regenerate_iqm.py
(batch runner). Requires FNNDSC cluster access to --base_path.

Data Format: Long format (one row per subject-split combination)
- This follows tidy data principles and works well with pandas operations
- Derived metrics (split differences, means across splits) computed at analysis time

Usage (run from src/):
----------------------
    # Per-split SNR/CNR/volumes -> data/image_quality_metrics.csv
    uv run python image_quality_metrics.py \
        --subjects ../data/subject.csv \
        --base_path /neuro/labs/grantlab/research/MRI_processing/seungyoon.jeong/2025/Reliability/TEST/

    # Also rebuild the volume pair-diff columns of data/split_comparision_data.csv
    # (split_pair, abs_diff_*, rel_diff_*) from the metrics CSV
    uv run python image_quality_metrics.py ... --pair_diffs

    # Or import as module
    from image_quality_metrics import (
        compute_subject_split_metrics,
        batch_compute_metrics,
        add_derived_metrics,
        compute_independent_pair_diffs,
    )

See docs/data_dictionary.md for which column comes from where.
"""

import nibabel as nib
import numpy as np
import pandas as pd
import os
import argparse
from typing import Dict, List, Optional, Union


# =============================================================================
# CONFIGURATION
# =============================================================================

# Tissue label definitions (SP model)
TISSUE_LABELS = {
    "sp": [4, 5],        # Left/Right Subplate
    "cp": [1, 42],       # Left/Right Cortical Plate  
    "inner": [160, 161], # Left/Right Inner region
}

# Default voxel size in mm (template space)
VOXEL_SIZE_MM = 0.5

# Splits to process
SPLITS = ["S1", "S2", "S3", "S4"]


# =============================================================================
# CORE METRIC FUNCTIONS
# =============================================================================

def extract_ga_from_session(session_id: str) -> Optional[int]:
    """
    Extract gestational age (GA) in weeks from session_id.
    
    Session ID format: YYYY.MM.DD-XXXW-MR_EI_Fetal_Neuro-NNNNN
                       or YYYY.MM.DD-XXXY-MR_EI_Fetal_Neuro-NNNNN
    where XXX is the gestational age in weeks (e.g., 032W or 032Y).
    
    Parameters
    ----------
    session_id : str
        Session identifier
        
    Returns
    -------
    int or None
        Gestational age in weeks, or None if cannot be parsed
    """
    try:
        # Split by '-' and look for the GA component (e.g., '032Y' or '032W')
        parts = session_id.split('-')
        if len(parts) >= 2:
            ga_part = parts[1]  # Should be something like '032Y' or '032W'
            # Extract numeric part (first 3 characters)
            ga_str = ga_part[:3]
            if ga_str.isdigit():
                return int(ga_str)
    except Exception as e:
        print(f"  Warning: Could not extract GA from session_id '{session_id}': {e}")
    
    return None


def compute_snr(t2_data: np.ndarray, mask_data: np.ndarray, 
                tissue_labels: Union[int, List[int]]) -> float:
    """
    Compute Signal-to-Noise Ratio for a tissue type.
    
    SNR = mean(signal) / std(signal)
    
    Uses standard deviation within tissue as noise proxy (appropriate when
    no clean background region is available in fetal brain imaging).
    
    Parameters
    ----------
    t2_data : np.ndarray
        T2 image volume
    mask_data : np.ndarray
        Segmentation mask
    tissue_labels : int or list
        Label(s) for the tissue of interest
        
    Returns
    -------
    float
        Signal-to-noise ratio, or np.nan if computation fails
    """
    if isinstance(tissue_labels, int):
        tissue_labels = [tissue_labels]
    
    tissue_mask = np.isin(mask_data, tissue_labels)
    intensities = t2_data[tissue_mask]
    
    if len(intensities) == 0:
        return np.nan
    
    mean_signal = np.mean(intensities)
    std_signal = np.std(intensities)
    
    if std_signal == 0:
        return np.nan
    
    return mean_signal / std_signal


def compute_cnr(t2_data: np.ndarray, mask_data: np.ndarray,
                labels_a: Union[int, List[int]], 
                labels_b: Union[int, List[int]]) -> float:
    """
    Compute Contrast-to-Noise Ratio between two tissue types.
    
    CNR = |mean(A) - mean(B)| / pooled_std
    
    Parameters
    ----------
    t2_data : np.ndarray
        T2 image volume
    mask_data : np.ndarray
        Segmentation mask
    labels_a : int or list
        Label(s) for tissue A
    labels_b : int or list
        Label(s) for tissue B
        
    Returns
    -------
    float
        Contrast-to-noise ratio, or np.nan if computation fails
    """
    if isinstance(labels_a, int):
        labels_a = [labels_a]
    if isinstance(labels_b, int):
        labels_b = [labels_b]
    
    mask_a = np.isin(mask_data, labels_a)
    mask_b = np.isin(mask_data, labels_b)
    
    intensities_a = t2_data[mask_a]
    intensities_b = t2_data[mask_b]
    
    if len(intensities_a) == 0 or len(intensities_b) == 0:
        return np.nan
    
    mean_a = np.mean(intensities_a)
    mean_b = np.mean(intensities_b)
    
    # Pooled standard deviation
    pooled_std = np.sqrt((np.var(intensities_a) + np.var(intensities_b)) / 2)
    
    if pooled_std == 0:
        return np.nan
    
    return np.abs(mean_a - mean_b) / pooled_std


def compute_tissue_stats(t2_data: np.ndarray, mask_data: np.ndarray,
                         labels: Union[int, List[int]]) -> Dict:
    """
    Compute basic statistics for a tissue region.
    
    Parameters
    ----------
    t2_data : np.ndarray
        T2 image volume
    mask_data : np.ndarray
        Segmentation mask
    labels : int or list
        Label(s) for the tissue
        
    Returns
    -------
    dict
        Dictionary with mean, std, voxel_count
    """
    if isinstance(labels, int):
        labels = [labels]
    
    mask = np.isin(mask_data, labels)
    intensities = t2_data[mask]
    
    if len(intensities) == 0:
        return {"mean": np.nan, "std": np.nan, "voxel_count": 0}
    
    return {
        "mean": np.mean(intensities),
        "std": np.std(intensities),
        "voxel_count": int(np.sum(mask)),
    }


def compute_scaling_factor(xfm_inv_path: str) -> Optional[float]:
    """
    Extract scaling factor from inverse transformation matrix.
    
    The scaling factor is the absolute determinant of the 3x3 rotation/scaling
    submatrix, representing the volume ratio between native and template space.
    
    Parameters
    ----------
    xfm_inv_path : str
        Path to recon_to31_inv.xfm file
        
    Returns
    -------
    float or None
        Scaling factor, or None if file cannot be loaded
    """
    try:
        A = np.loadtxt(xfm_inv_path)
        return abs(np.linalg.det(A[:3, :3]))
    except Exception as e:
        print(f"  Warning: Could not load {xfm_inv_path}: {e}")
        return None


def compute_native_volume(voxel_count: int, scaling_factor: Optional[float],
                          voxel_size: float = VOXEL_SIZE_MM) -> Optional[float]:
    """
    Convert template-space voxel count to native anatomical volume (mm³).
    
    Formula: V_native = voxel_count × voxel_volume × scaling_factor
    
    Parameters
    ----------
    voxel_count : int
        Number of voxels in template space
    scaling_factor : float or None
        Scaling factor from transformation matrix
    voxel_size : float
        Voxel size in mm (default 0.5mm isotropic)
        
    Returns
    -------
    float or None
        Native volume in mm³
    """
    if scaling_factor is None or voxel_count is None:
        return None
    
    voxel_volume = voxel_size ** 3  # 0.125 mm³ for 0.5mm voxels
    return voxel_count * voxel_volume * scaling_factor


# =============================================================================
# SUBJECT/SPLIT PROCESSING
# =============================================================================

def compute_subject_split_metrics(t2_path: str, seg_path: str, 
                                   xfm_inv_path: str,
                                   subject_id: str, session_id: str, 
                                   split: str) -> Optional[Dict]:
    """
    Compute all image quality metrics for a single subject-split.
    
    Parameters
    ----------
    t2_path : str
        Path to T2 image (recon_to31_nuc.nii)
    seg_path : str
        Path to SP model segmentation
    xfm_inv_path : str
        Path to inverse transformation matrix
    subject_id : str
        Subject identifier
    session_id : str
        Session identifier
    split : str
        Split identifier (S1, S2, S3, S4)
        
    Returns
    -------
    dict or None
        Dictionary with all computed metrics, or None if loading fails
    """
    # Load data
    try:
        t2_data = nib.load(t2_path).get_fdata()
        seg_data = nib.load(seg_path).get_fdata()
    except Exception as e:
        print(f"  Error loading {subject_id}/{session_id}/{split}: {e}")
        return None
    
    # Get scaling factor
    scaling_factor = compute_scaling_factor(xfm_inv_path)
    
    # Compute SNR for each tissue
    snr_sp = compute_snr(t2_data, seg_data, TISSUE_LABELS["sp"])
    snr_cp = compute_snr(t2_data, seg_data, TISSUE_LABELS["cp"])
    snr_inner = compute_snr(t2_data, seg_data, TISSUE_LABELS["inner"])
    
    # Compute CNR between tissue pairs
    cnr_sp_cp = compute_cnr(t2_data, seg_data, TISSUE_LABELS["sp"], TISSUE_LABELS["cp"])
    cnr_sp_inner = compute_cnr(t2_data, seg_data, TISSUE_LABELS["sp"], TISSUE_LABELS["inner"])
    cnr_cp_inner = compute_cnr(t2_data, seg_data, TISSUE_LABELS["cp"], TISSUE_LABELS["inner"])
    
    # Get tissue statistics
    stats_sp = compute_tissue_stats(t2_data, seg_data, TISSUE_LABELS["sp"])
    stats_cp = compute_tissue_stats(t2_data, seg_data, TISSUE_LABELS["cp"])
    stats_inner = compute_tissue_stats(t2_data, seg_data, TISSUE_LABELS["inner"])
    
    # Compute native volumes
    native_vol_sp = compute_native_volume(stats_sp["voxel_count"], scaling_factor)
    native_vol_cp = compute_native_volume(stats_cp["voxel_count"], scaling_factor)
    native_vol_inner = compute_native_volume(stats_inner["voxel_count"], scaling_factor)
    
    # Extract gestational age
    ga = extract_ga_from_session(session_id)
    
    return {
        # Identifiers
        "subject_id": subject_id,
        "session_id": session_id,
        "ga": ga,
        "split": split,
        # SNR
        "snr_subplate": snr_sp,
        "snr_cortical_plate": snr_cp,
        "snr_inner": snr_inner,
        # CNR
        "cnr_sp_cp": cnr_sp_cp,
        "cnr_sp_inner": cnr_sp_inner,
        "cnr_cp_inner": cnr_cp_inner,
        # Tissue stats - Subplate
        "sp_mean_intensity": stats_sp["mean"],
        "sp_std_intensity": stats_sp["std"],
        "sp_volume_voxels": stats_sp["voxel_count"],
        # Tissue stats - Cortical Plate
        "cp_mean_intensity": stats_cp["mean"],
        "cp_std_intensity": stats_cp["std"],
        "cp_volume_voxels": stats_cp["voxel_count"],
        # Tissue stats - Inner
        "inner_mean_intensity": stats_inner["mean"],
        "inner_std_intensity": stats_inner["std"],
        "inner_volume_voxels": stats_inner["voxel_count"],
        # Scaling and native volumes
        "scaling_factor": scaling_factor,
        "native_vol_sp": native_vol_sp,
        "native_vol_cp": native_vol_cp,
        "native_vol_inner": native_vol_inner,
    }


def build_paths(base_path: str, subject_id: str, session_id: str, split: str) -> Dict[str, str]:
    """
    Build file paths for a subject-split.
    
    Parameters
    ----------
    base_path : str
        Base path to data directory
    subject_id : str
        Subject identifier
    session_id : str
        Session identifier
    split : str
        Split identifier
        
    Returns
    -------
    dict
        Dictionary with t2_path, seg_path, xfm_inv_path
    """
    split_path = os.path.join(base_path, str(subject_id), str(session_id), split)
    
    return {
        "t2_path": os.path.join(split_path, "recon_segmentation", "recon_to31_nuc.nii"),
        "seg_path": os.path.join(
            split_path, "segmentations",
            f"{subject_id}_{session_id}_nuc_deep_subplate_dilate_mc.nii"
        ),
        "xfm_inv_path": os.path.join(
            split_path, "recon_segmentation", "alignment_temp", "recon_to31_inv.xfm"
        ),
    }


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def batch_compute_metrics(subjects_df: pd.DataFrame, base_path: str,
                          splits: List[str] = SPLITS,
                          output_csv: Optional[str] = "../data/image_quality_metrics.csv",
                          verbose: bool = True) -> pd.DataFrame:
    """
    Compute image quality metrics for all subjects across all splits.
    
    Parameters
    ----------
    subjects_df : pd.DataFrame
        DataFrame with 'subject_id' and 'session_id' columns
    base_path : str
        Base path to data directory
    splits : list
        List of splits to process (default: S1-S4)
    output_csv : str or None
        Path to save output CSV (None to skip saving)
    verbose : bool
        Print progress messages
        
    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with one row per subject-split
    """
    results = []
    total = len(subjects_df) * len(splits)
    processed = 0
    
    for idx, row in subjects_df.iterrows():
        subject_id = str(row["subject_id"])
        session_id = str(row["session_id"])
        
        for split in splits:
            processed += 1
            
            if verbose:
                print(f"[{processed}/{total}] Processing {subject_id} - {split}...")
            
            paths = build_paths(base_path, subject_id, session_id, split)
            
            # Check files exist
            missing = [k for k, v in paths.items() if not os.path.exists(v)]
            if missing:
                if verbose:
                    print(f"  Warning: Missing files for {split}: {missing}")
                continue
            
            # Compute metrics
            metrics = compute_subject_split_metrics(
                paths["t2_path"], paths["seg_path"], paths["xfm_inv_path"],
                subject_id, session_id, split
            )
            
            if metrics is not None:
                results.append(metrics)
                if verbose:
                    print(f"  SNR(SP): {metrics['snr_subplate']:.2f}, "
                          f"SNR(CP): {metrics['snr_cortical_plate']:.2f}")
    
    df = pd.DataFrame(results)
    
    if output_csv and len(df) > 0:
        df.to_csv(output_csv, index=False)
        if verbose:
            print(f"\nSaved {len(df)} records to: {output_csv}")
    
    return df


def update_existing_csv(subjects_df: pd.DataFrame, base_path: str,
                        csv_path: str = "../data/image_quality_metrics.csv",
                        splits: List[str] = SPLITS) -> pd.DataFrame:
    """
    Update existing CSV with new subjects, avoiding recomputation.
    
    Parameters
    ----------
    subjects_df : pd.DataFrame
        DataFrame with 'subject_id' and 'session_id' columns
    base_path : str
        Base path to data directory
    csv_path : str
        Path to existing CSV
    splits : list
        List of splits to process
        
    Returns
    -------
    pd.DataFrame
        Updated DataFrame
    """
    # Load existing data if present
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        existing_keys = set(
            zip(
                existing_df["subject_id"].astype(str),
                existing_df["session_id"].astype(str),
                existing_df["split"].astype(str)
            )
        )
        print(f"Found {len(existing_df)} existing records in {csv_path}")
    else:
        existing_df = pd.DataFrame()
        existing_keys = set()
    
    # Find missing subject-splits
    new_records = []
    for _, row in subjects_df.iterrows():
        subject_id = str(row["subject_id"])
        session_id = str(row["session_id"])
        
        for split in splits:
            key = (subject_id, session_id, split)
            if key not in existing_keys:
                new_records.append({"subject_id": subject_id, 
                                    "session_id": session_id,
                                    "split": split})
    
    if not new_records:
        print("All subjects already processed. No updates needed.")
        return existing_df
    
    print(f"Found {len(new_records)} new subject-splits to process")
    
    # Process new records
    new_df = pd.DataFrame(new_records)
    new_results = batch_compute_metrics(
        new_df.drop_duplicates(subset=["subject_id", "session_id"]),
        base_path,
        splits=splits,
        output_csv=None,  # Don't save yet
        verbose=True
    )
    
    # Combine and save
    if len(existing_df) > 0:
        combined_df = pd.concat([existing_df, new_results], ignore_index=True)
    else:
        combined_df = new_results
    
    combined_df.to_csv(csv_path, index=False)
    print(f"Saved {len(combined_df)} total records to: {csv_path}")
    
    return combined_df


# =============================================================================
# DERIVED METRICS (for analysis)
# =============================================================================

def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived metrics for analysis (split differences, means).
    
    This converts long format to wide format with additional computed columns.
    Use this in summary_analysis.py for correlation analyses.
    
    Parameters
    ----------
    df : pd.DataFrame
        Long-format DataFrame from batch_compute_metrics
        
    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with derived metrics
    """
    # Pivot to wide format
    pivot_cols = [
        "snr_subplate", "snr_cortical_plate", "snr_inner",
        "sp_volume_voxels", "cp_volume_voxels", "inner_volume_voxels",
        "native_vol_sp", "native_vol_cp", "native_vol_inner",
        "scaling_factor"
    ]

    result_dfs = []

    for col in pivot_cols:
        if col not in df.columns:
            continue
        pivoted = df.pivot(
            index=["subject_id", "session_id"],
            columns="split",
            values=col
        )
        pivoted.columns = [f"{col}_{split}" for split in pivoted.columns]
        result_dfs.append(pivoted)

    if not result_dfs:
        return df

    wide_df = pd.concat(result_dfs, axis=1).reset_index()

    # Add SNR differences between independent split pairs
    for tissue in ["subplate", "cortical_plate"]:
        s1_col = f"snr_{tissue}_S1"
        s2_col = f"snr_{tissue}_S2"
        s3_col = f"snr_{tissue}_S3"
        s4_col = f"snr_{tissue}_S4"

        if s1_col in wide_df.columns and s2_col in wide_df.columns:
            wide_df[f"snr_{tissue}_diff_S1S2"] = abs(wide_df[s1_col] - wide_df[s2_col])
        if s3_col in wide_df.columns and s4_col in wide_df.columns:
            wide_df[f"snr_{tissue}_diff_S3S4"] = abs(wide_df[s3_col] - wide_df[s4_col])

    # Add voxel count relative differences
    for tissue in ["sp", "cp", "inner"]:
        for pair, (s1, s2) in [("S1S2", ("S1", "S2")), ("S3S4", ("S3", "S4"))]:
            col1 = f"{tissue}_volume_voxels_{s1}"
            col2 = f"{tissue}_volume_voxels_{s2}"

            if col1 in wide_df.columns and col2 in wide_df.columns:
                v1, v2 = wide_df[col1], wide_df[col2]
                mean_val = (v1 + v2) / 2
                wide_df[f"voxels_{tissue}_diff_{pair}"] = abs(v1 - v2)
                wide_df[f"voxels_{tissue}_reldiff_{pair}"] = np.where(
                    mean_val > 0,
                    abs(v1 - v2) / mean_val * 100,
                    0
                )

    # Add mean SNR across splits
    for tissue in ["subplate", "cortical_plate"]:
        snr_cols = [f"snr_{tissue}_{s}" for s in SPLITS if f"snr_{tissue}_{s}" in wide_df.columns]
        if snr_cols:
            wide_df[f"snr_{tissue}_mean"] = wide_df[snr_cols].mean(axis=1)
            wide_df[f"snr_{tissue}_std"] = wide_df[snr_cols].std(axis=1)

    return wide_df


# =============================================================================
# DERIVED METRICS (for analysis)
# =============================================================================


def compute_independent_pair_diffs(quality_df):
    """Compute native volume differences for S1-S2 and S3-S4 pairs."""
    records = []

    for (subj, sess), group in quality_df.groupby(["subject_id", "session_id"]):
        ga = group["GA"].iloc[0]

        for split1, split2 in [("S1", "S2"), ("S3", "S4")]:
            s1 = group[group["split"] == split1]
            s2 = group[group["split"] == split2]

            if len(s1) == 0 or len(s2) == 0:
                continue

            # Get volumes for each tissue
            sp1, sp2 = s1["native_vol_sp"].values[0], s2["native_vol_sp"].values[0]
            cp1, cp2 = s1["native_vol_cp"].values[0], s2["native_vol_cp"].values[0]
            inner1, inner2 = (
                s1["native_vol_inner"].values[0],
                s2["native_vol_inner"].values[0],
            )

            # Skip if any missing
            if any(pd.isna([sp1, sp2, cp1, cp2, inner1, inner2])):
                continue

            total1 = sp1 + cp1 + inner1
            total2 = sp2 + cp2 + inner2

            # Absolute differences
            abs_sp = abs(sp1 - sp2)
            abs_cp = abs(cp1 - cp2)
            abs_inner = abs(inner1 - inner2)
            abs_total = abs(total1 - total2)

            # Relative differences (%)
            rel_sp = abs_sp / ((sp1 + sp2) / 2) * 100
            rel_cp = abs_cp / ((cp1 + cp2) / 2) * 100
            rel_inner = abs_inner / ((inner1 + inner2) / 2) * 100
            rel_total = abs_total / ((total1 + total2) / 2) * 100

            records.append(
                {
                    "subject_id": subj,
                    "session_id": sess,
                    "GA": ga,
                    "split_pair": f"{split1}-{split2}",
                    "abs_diff_sp": abs_sp,
                    "abs_diff_cp": abs_cp,
                    "abs_diff_inner": abs_inner,
                    "abs_diff_total": abs_total,
                    "rel_diff_sp": rel_sp,
                    "rel_diff_cp": rel_cp,
                    "rel_diff_inner": rel_inner,
                    "rel_diff_total": rel_total,
                }
            )

    return pd.DataFrame(records)


def update_pair_diffs_csv(quality_df: pd.DataFrame, csv_path: str) -> pd.DataFrame:
    """Refresh the volume pair-diff columns of split_comparision_data.csv in place.

    Recomputes split_pair / abs_diff_* / rel_diff_* from `quality_df` and merges
    them back on (subject_id, session_id, split_pair), leaving every other column
    of the target CSV untouched. Columns owned by other producers -- cnr_diff and
    cnr_mean come from src/functions/CNR.py -- are preserved.

    Creates the CSV if it does not exist yet.
    """
    diffs = compute_independent_pair_diffs(quality_df)
    if diffs.empty:
        print(f"\nNo complete split pairs found; leaving {csv_path} untouched.")
        return diffs

    keys = ["subject_id", "session_id", "split_pair"]

    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        owned = [c for c in diffs.columns if c not in keys]
        existing = existing.drop(columns=owned, errors="ignore")
        for key in keys:
            existing[key] = existing[key].astype(str)
            diffs[key] = diffs[key].astype(str)
        merged = existing.merge(diffs, on=keys, how="outer")
    else:
        merged = diffs

    merged.to_csv(csv_path, index=False)
    print(f"\nWrote volume pair diffs for {len(diffs)} split pairs to {csv_path}")
    return merged


# =============================================================================
# SUMMARY/REPORTING
# =============================================================================

def print_summary(df: pd.DataFrame):
    """Print summary statistics for image quality metrics."""
    print("\n" + "=" * 70)
    print("IMAGE QUALITY METRICS SUMMARY")
    print("=" * 70)
    
    n_subjects = df[["subject_id", "session_id"]].drop_duplicates().shape[0]
    n_splits = df["split"].nunique() if "split" in df.columns else "N/A"
    n_records = len(df)
    
    print(f"\nDataset: {n_subjects} subjects, {n_splits} splits, {n_records} total records")
    
    # SNR summary
    print("\n--- SNR by Tissue Type ---")
    snr_cols = ["snr_subplate", "snr_cortical_plate", "snr_inner"]
    for col in snr_cols:
        if col in df.columns:
            data = df[col].dropna()
            print(f"  {col.replace('snr_', '').replace('_', ' ').title():20s}: "
                  f"Mean={data.mean():.2f}, SD={data.std():.2f}, "
                  f"Range=[{data.min():.2f}, {data.max():.2f}]")
    
    # CNR summary
    print("\n--- CNR Between Tissues ---")
    cnr_cols = ["cnr_sp_cp", "cnr_sp_inner", "cnr_cp_inner"]
    for col in cnr_cols:
        if col in df.columns:
            data = df[col].dropna()
            label = col.replace("cnr_", "").replace("_", " vs ").upper()
            print(f"  {label:20s}: "
                  f"Mean={data.mean():.2f}, SD={data.std():.2f}")
    
    # Volume summary (native)
    print("\n--- Native Volumes (mm³) ---")
    vol_cols = ["native_vol_sp", "native_vol_cp", "native_vol_inner"]
    for col in vol_cols:
        if col in df.columns:
            data = df[col].dropna()
            label = col.replace("native_vol_", "").upper()
            print(f"  {label:10s}: Mean={data.mean():.1f}, SD={data.std():.1f}")
    
    print("\n" + "=" * 70)


# =============================================================================
# MAIN CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute image quality metrics for fetal brain segmentation reliability study"
    )
    parser.add_argument(
        "--subjects", "-s", required=True,
        help="Path to subjects CSV (must have subject_id, session_id columns)"
    )
    parser.add_argument(
        "--base_path", "-b", required=True,
        help="Base path to data directory"
    )
    parser.add_argument(
        "--output", "-o", default="../data/image_quality_metrics.csv",
        help="Output CSV path (default: data/image_quality_metrics.csv)"
    )
    parser.add_argument(
        "--update", "-u", action="store_true",
        help="Update existing CSV (skip already processed subjects)"
    )
    parser.add_argument(
        "--splits", nargs="+", default=SPLITS,
        help="Splits to process (default: S1 S2 S3 S4)"
    )
    parser.add_argument(
        "--pair_diffs", action="store_true",
        help="Also refresh the volume pair-diff columns (split_pair, abs_diff_*, "
             "rel_diff_*) in --pair_diffs_output from the computed metrics"
    )
    parser.add_argument(
        "--pair_diffs_output", default="../data/split_comparision_data.csv",
        help="Target CSV for --pair_diffs (default: data/split_comparision_data.csv)"
    )

    args = parser.parse_args()
    
    # Load subjects
    subjects_df = pd.read_csv(args.subjects)
    print(f"Found {len(subjects_df)} subjects in {args.subjects}")
    
    # Process
    if args.update:
        df = update_existing_csv(
            subjects_df, args.base_path,
            csv_path=args.output,
            splits=args.splits
        )
    else:
        df = batch_compute_metrics(
            subjects_df, args.base_path,
            splits=args.splits,
            output_csv=args.output
        )
    
    # Print summary
    print_summary(df)

    if args.pair_diffs:
        update_pair_diffs_csv(df, args.pair_diffs_output)

    print("\nDone!")


if __name__ == "__main__":
    main()
