import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, generate_binary_structure
import os
import matplotlib.pyplot as plt
from helpers import get_middle_slice, normalize_intensity

TISSUE_LABELS = {
    "sp": [4, 5],
    "cp": [1, 42],
    "inner": [160, 161],
}

OPACITY = 0.4

def compute_cr(t2_data, seg_data, connectivity=3):
    """
    Compute boundary contrast ratio between SP and IZ.
    Returns dict with CR, intensities, counts, and masks for plotting.
    """
    struct = generate_binary_structure(3, connectivity)
    sp_mask = np.isin(seg_data, TISSUE_LABELS["sp"])
    iz_mask = np.isin(seg_data, TISSUE_LABELS["inner"])

    # SP boundary: SP voxels adjacent to IZ
    sp_boundary = sp_mask & binary_dilation(iz_mask, structure=struct, iterations=1)
    # IZ boundary: IZ voxels adjacent to SP
    iz_boundary = iz_mask & binary_dilation(sp_mask, structure=struct, iterations=1)

    sp_band = sp_boundary
    iz_band = iz_boundary

    sp_int = t2_data[sp_band]
    iz_int = t2_data[iz_band]
    mean_sp = np.mean(sp_int) if len(sp_int) > 0 else np.nan
    mean_iz = np.mean(iz_int) if len(iz_int) > 0 else np.nan
    denom = mean_sp + mean_iz
    cr = (mean_sp - mean_iz) / denom if denom != 0 else np.nan

    return {
        "cr": cr,
        "t2_data": t2_data,
        "sp_mask": sp_mask,
        "iz_mask": iz_mask,
        "sp_band": sp_band,
        "iz_band": iz_band,
    }

from matplotlib.patches import Patch


def plot_cr_bands(t2_data, cr_result, title="CR Bands", ax=None):

    mid_slice = get_middle_slice(t2_data, 2)
    t2_slice = t2_data[:, :, mid_slice]
    t2_norm = normalize_intensity(t2_slice)

    sp_mask_slice = cr_result["sp_mask"][:, :, mid_slice]
    iz_mask_slice = cr_result["iz_mask"][:, :, mid_slice]
    sp_band_slice = cr_result["sp_band"][:, :, mid_slice]
    iz_band_slice = cr_result["iz_band"][:, :, mid_slice]

    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    else:
        # If a single ax is provided, replace it with two side-by-side axes
        fig = ax.get_figure()
        pos = ax.get_position()
        ax.remove()
        half_w = pos.width / 2
        ax_left = fig.add_axes([pos.x0, pos.y0, half_w, pos.height])
        ax_right = fig.add_axes([pos.x0 + half_w, pos.y0, half_w, pos.height])
        axes = [ax_left, ax_right]

    # --- Left panel: original masks over T2 ---
    axes[0].imshow(t2_norm.T, cmap="gray", origin="lower")

    sp_mask_overlay = np.zeros((*t2_slice.shape, 4))
    sp_mask_overlay[sp_mask_slice, :] = [1, 0, 0, OPACITY]  # red
    axes[0].imshow(np.transpose(sp_mask_overlay, (1, 0, 2)), origin="lower")

    iz_mask_overlay = np.zeros((*t2_slice.shape, 4))
    iz_mask_overlay[iz_mask_slice, :] = [0, 0.4, 1, OPACITY]  # blue
    axes[0].imshow(np.transpose(iz_mask_overlay, (1, 0, 2)), origin="lower")

    axes[0].set_title(f"{title} — Original Masks")
    axes[0].axis("off")

    mask_legend = [
        Patch(facecolor=(1, 0, 0, OPACITY), edgecolor="red", label="SP mask"),
        Patch(facecolor=(0, 0.4, 1, OPACITY), edgecolor="blue", label="IZ mask"),
    ]
    axes[0].legend(handles=mask_legend, loc="upper right", fontsize=9)

    # --- Right panel: bands over T2 ---
    axes[1].imshow(t2_norm.T, cmap="gray", origin="lower")

    sp_overlay = np.zeros((*t2_slice.shape, 4))
    sp_overlay[sp_band_slice, :] = [1, 0, 0, OPACITY]  # red
    axes[1].imshow(np.transpose(sp_overlay, (1, 0, 2)), origin="lower")

    iz_overlay = np.zeros((*t2_slice.shape, 4))
    iz_overlay[iz_band_slice, :] = [0, 0.4, 1, OPACITY]  # blue
    axes[1].imshow(np.transpose(iz_overlay, (1, 0, 2)), origin="lower")

    axes[1].set_title(f"{title} — Boundary Bands")
    axes[1].axis("off")

    band_legend = [
        Patch(facecolor=(1, 0, 0, OPACITY), edgecolor="red", label="SP band"),
        Patch(facecolor=(0, 0.4, 1, OPACITY), edgecolor="blue", label="IZ band"),
    ]
    axes[1].legend(handles=band_legend, loc="upper right", fontsize=9)

    return fig


# === Batch ===
def batch_cnr(subjects_df, base_path, splits=None):
    """Compute boundary CR for all subjects/splits. Returns long-format DataFrame."""
    if splits is None:
        splits = ["S1", "S2", "S3", "S4"]
    records = []
    for _, row in subjects_df.iterrows():
        subj, sess = str(row["subject_id"]), str(row["session_id"])
        ga = row.get("GA", np.nan)
        for split in splits:
            sp = os.path.join(base_path, subj, sess, split)
            t2_p = os.path.join(sp, "recon_segmentation/recon_to31_nuc.nii")
            seg_p = os.path.join(
                sp, f"segmentations/{subj}_{sess}_nuc_deep_subplate_dilate_mc.nii"
            )
            if not os.path.exists(t2_p) or not os.path.exists(seg_p):
                continue
            try:
                t2 = nib.load(t2_p).get_fdata()
                seg = nib.load(seg_p).get_fdata()

                r = compute_cr(
                    t2, seg
                )
                records.append(
                    {
                        "subject_id": subj,
                        "session_id": sess,
                        "GA": ga,
                        "split": split,
                        "cr": r["cr"],
                    }
                )
            except Exception as e:
                print(f"  Error {subj}/{split}: {e}")

            if split == "S1":
                savepath = "cnr_batch_tests"
                os.makedirs(savepath, exist_ok=True)
                fig = plot_cr_bands(t2_data=t2, cr_result=r)
                fig.savefig(os.path.join(savepath, f"{subj}.png"))
                plt.close(fig)

            print(".", end="", flush=True)
        print("|", end="", flush=True)
    return pd.DataFrame(records)

if __name__ == "__main__":
    base_path = "/neuro/labs/grantlab/research/MRI_processing/seungyoon.jeong/2025/Reliability/TEST/"
    subjects_df = pd.read_csv("../../data/subject.csv")
    quality_df = pd.read_csv("../../data/image_quality_metrics.csv")
    split_pair_diffs = pd.read_csv("../../data/split_comparision_data.csv")

    # Compute CNR data
    cnr_data = batch_cnr(subjects_df, base_path)

    quality_df = quality_df.drop(columns=["sp_iz_cnr"])        

    # Merge CNR data into quality_df on subject_id, session_id, and split
    quality_df = quality_df.merge(
        cnr_data[["subject_id", "session_id", "split", "cr"]],
        on=["subject_id", "session_id", "split"],
        how="left",
    )

    # Rename cr to sp_iz_cnr
    quality_df.rename(columns={"cr": "sp_iz_cnr"}, inplace=True)

    # Save the updated dataframe
    quality_df.to_csv("../../data/image_quality_metrics.csv", index=False)

    print("\n")
    print("Successfully added sp_iz_cnr column to image_quality_metrics.csv")
    print(f"\nUpdated dataframe shape: {quality_df.shape}")
    print(f"\nFirst few rows of sp_iz_cnr:")
    print(quality_df[["subject_id", "session_id", "split", "sp_iz_cnr"]].head(10))

    # === Compute CNR diff and mean for split pairs (S1-S2, S3-S4) ===
    print("\n\nComputing CNR differences and means for split pairs...")
    
    # Create a pivot table for easy access to CNR values by split
    cnr_pivot = cnr_data.pivot_table(
        index=["subject_id", "session_id"],
        columns="split",
        values="cr",
        aggfunc="first"
    ).reset_index()
    
    # Compute CNR diff and mean for each split pair
    cnr_pair_records = []
    for _, row in cnr_pivot.iterrows():
        subj = row["subject_id"]
        sess = row["session_id"]
        
        # S1-S2 pair
        if "S1" in row.index and "S2" in row.index:
            s1_cnr = row.get("S1", np.nan)
            s2_cnr = row.get("S2", np.nan)
            if pd.notna(s1_cnr) and pd.notna(s2_cnr):
                cnr_pair_records.append({
                    "subject_id": subj,
                    "session_id": sess,
                    "split_pair": "S1-S2",
                    "cnr_diff": abs(s1_cnr - s2_cnr),
                    "cnr_mean": (s1_cnr + s2_cnr) / 2
                })
        
        # S3-S4 pair
        if "S3" in row.index and "S4" in row.index:
            s3_cnr = row.get("S3", np.nan)
            s4_cnr = row.get("S4", np.nan)
            if pd.notna(s3_cnr) and pd.notna(s4_cnr):
                cnr_pair_records.append({
                    "subject_id": subj,
                    "session_id": sess,
                    "split_pair": "S3-S4",
                    "cnr_diff": abs(s3_cnr - s4_cnr),
                    "cnr_mean": (s3_cnr + s4_cnr) / 2
                })
    
    cnr_pair_df = pd.DataFrame(cnr_pair_records)
    
    # Remove existing cnr_diff and cnr_mean columns if they exist
    for col in ["cnr_diff", "cnr_mean"]:
        if col in split_pair_diffs.columns:
            split_pair_diffs = split_pair_diffs.drop(columns=[col])
    
    # Merge CNR pair data into split_pair_diffs
    split_pair_diffs = split_pair_diffs.merge(
        cnr_pair_df,
        on=["subject_id", "session_id", "split_pair"],
        how="left"
    )
    
    # Save the updated split comparison dataframe
    split_pair_diffs.to_csv("../../data/split_comparision_data.csv", index=False)
    
    print("Successfully added cnr_diff and cnr_mean columns to split_comparision_data.csv")
    print(f"\nUpdated split comparison dataframe shape: {split_pair_diffs.shape}")
    print(f"\nFirst few rows with CNR columns:")
    print(split_pair_diffs[["subject_id", "session_id", "split_pair", "cnr_diff", "cnr_mean"]].head(10))
