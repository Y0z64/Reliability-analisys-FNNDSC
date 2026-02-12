import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import binary_dilation, generate_binary_structure
from helpers import get_middle_slice, normalize_intensity
import os

TISSUE_LABELS = {
    "sp": [4, 5],
    "cp": [1, 42],
    "inner": [160, 161],
}

OPACITY = 0.4


def compute_boundary_cr(t2_data, seg_data, connectivity=3):
    """
    Compute boundary contrast ratio between SP and IZ.
    Returns dict with CR, intensities, counts, and masks for plotting.
    """
    struct = generate_binary_structure(3, connectivity)
    sp_mask = np.isin(seg_data, TISSUE_LABELS["sp"])
    iz_mask = np.isin(seg_data, TISSUE_LABELS["inner"])

    # SP boundary: SP voxels adjacent to IZ
    sp_boundary = sp_mask & binary_dilation(iz_mask, structure=struct)
    # IZ boundary: IZ voxels adjacent to SP
    iz_boundary = iz_mask & binary_dilation(sp_mask, structure=struct)

    sp_int = t2_data[sp_boundary]
    iz_int = t2_data[iz_boundary]
    mean_sp = np.mean(sp_int) if len(sp_int) > 0 else np.nan
    mean_iz = np.mean(iz_int) if len(iz_int) > 0 else np.nan
    denom = mean_sp + mean_iz
    cr = (mean_sp - mean_iz) / denom if denom != 0 else np.nan

    return {
        "cr": cr,
        "mean_sp_boundary": mean_sp,
        "mean_iz_boundary": mean_iz,
        "n_sp_boundary": int(np.sum(sp_boundary)),
        "n_iz_boundary": int(np.sum(iz_boundary)),
        "sp_mask": sp_mask,
        "iz_mask": iz_mask,
        "sp_boundary": sp_boundary,
        "iz_boundary": iz_boundary,
    }


def plot_boundary_cr(t2_data, result, subject_id="", session_id="", split=""):
    """
    Plot boundary CR using same layout as Reliability.py:
    Row 0: T2 + SP/IZ masks (full)
    Row 1: T2 + boundary voxels only
    Col 3: Metrics panel
    """
    # Build overlay segmentations: 1=SP, 2=IZ
    full_seg = np.zeros_like(t2_data, dtype=int)
    full_seg[result["sp_mask"]] = 1
    full_seg[result["iz_mask"]] = 2

    boundary_seg = np.zeros_like(t2_data, dtype=int)
    boundary_seg[result["sp_boundary"]] = 1
    boundary_seg[result["iz_boundary"]] = 2

    # Colormaps: transparent, green(SP), blue(IZ)
    full_cmap = ListedColormap([(0, 0, 0, 0), (0, 0.8, 0.2, 1), (0.2, 0.2, 0.9, 1)])
    # transparent, orange(SP bnd), cyan(IZ bnd)
    bnd_cmap = ListedColormap([(0, 0, 0, 0), (1, 0.3, 0, 1), (0, 0.8, 1, 1)])

    slices = {
        "Axial": (2, get_middle_slice(t2_data, 2)),
        "Coronal": (1, get_middle_slice(t2_data, 1)),
        "Sagittal": (0, get_middle_slice(t2_data, 0)),
    }

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(
        2, 4, hspace=0.25, wspace=0.15, left=0.05, right=0.95, top=0.92, bottom=0.05
    )

    for col, (view_name, (axis, idx)) in enumerate(slices.items()):
        slc = [slice(None)] * 3
        slc[axis] = idx

        t2_slice = normalize_intensity(t2_data[tuple(slc)])
        full_slice = full_seg[tuple(slc)]
        bnd_slice = boundary_seg[tuple(slc)]

        # Row 0: full masks
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(t2_slice.T, cmap="gray", origin="lower")
        ax.imshow(
            full_slice.T,
            cmap=full_cmap,
            vmin=0,
            vmax=2,
            alpha=OPACITY,
            origin="lower",
            interpolation="nearest",
        )
        ax.set_title(view_name, fontsize=14, fontweight="bold", pad=10)
        ax.axis("off")

        # Row 1: boundary only
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(t2_slice.T, cmap="gray", origin="lower")
        ax.imshow(
            bnd_slice.T,
            cmap=bnd_cmap,
            vmin=0,
            vmax=2,
            alpha=0.8,
            origin="lower",
            interpolation="nearest",
        )
        ax.axis("off")

    fig.text(
        0.01,
        0.72,
        "Full Masks\n(SP=green, IZ=blue)",
        fontsize=11,
        fontweight="bold",
        rotation=90,
        va="center",
    )
    fig.text(
        0.01,
        0.28,
        "Boundary Only\n(SP=orange, IZ=cyan)",
        fontsize=11,
        fontweight="bold",
        rotation=90,
        va="center",
    )

    # Metrics panel (same style as Reliability.py)
    ax_m = fig.add_subplot(gs[:, 3])
    ax_m.axis("off")

    r = result
    metrics_lines = [
        f"{'Metric':<18} {'Value':>10}",
        "─" * 30,
        f"{'CR':<18} {r['cr']:>10.4f}",
        "",
        f"{'SP boundary':<18} {'':>10}",
        f"  Mean intensity   {r['mean_sp_boundary']:>10.1f}",
        f"  Voxels           {r['n_sp_boundary']:>10d}",
        "",
        f"{'IZ boundary':<18} {'':>10}",
        f"  Mean intensity   {r['mean_iz_boundary']:>10.1f}",
        f"  Voxels           {r['n_iz_boundary']:>10d}",
    ]

    ax_m.text(
        0.1,
        0.95,
        "\n".join(metrics_lines),
        fontsize=9,
        verticalalignment="top",
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=1",
            facecolor="#f0f0f0",
            edgecolor="#666666",
            linewidth=1.5,
        ),
    )

    title = f"{subject_id} | {session_id} | {split}" if subject_id else "Boundary CR"
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.97)

    return fig


def plot_boundary_3d(result, downsample=2):
    """Render 3D scatter of full masks + boundary voxels."""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    configs = [
        ("sp_mask", "SP", "green", 0.02, 1),
        ("iz_mask", "IZ", "blue", 0.02, 1),
        ("sp_boundary", "SP boundary", "orange", 0.6, 8),
        ("iz_boundary", "IZ boundary", "cyan", 0.6, 8),
    ]

    for key, label, color, alpha, size in configs:
        coords = np.argwhere(result[key])
        if len(coords) == 0:
            continue
        if "boundary" not in key:
            coords = coords[::downsample]
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            c=color,
            alpha=alpha,
            s=size,
            label=f"{label} ({np.sum(result[key])} vox)",
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(
        f"3D Boundary | CR = {result['cr']:.4f}", fontsize=14, fontweight="bold"
    )
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    return fig


# === Batch ===
def batch_boundary_cr(subjects_df, base_path, splits=None):
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
                r = compute_boundary_cr(
                    nib.load(t2_p).get_fdata(), nib.load(seg_p).get_fdata()
                )
                records.append(
                    {
                        "subject_id": subj,
                        "session_id": sess,
                        "GA": ga,
                        "split": split,
                        "boundary_cr": r["cr"],
                        "mean_sp_boundary": r["mean_sp_boundary"],
                        "mean_iz_boundary": r["mean_iz_boundary"],
                        "n_sp_boundary": r["n_sp_boundary"],
                        "n_iz_boundary": r["n_iz_boundary"],
                    }
                )
            except Exception as e:
                print(f"  Error {subj}/{split}: {e}")

            print(".", end="", flush=True)
        print("|", end="", flush=True)
    return pd.DataFrame(records)

if __name__ == "__main__":
    base_path = "/neuro/labs/grantlab/research/MRI_processing/seungyoon.jeong/2025/Reliability/TEST/"
    subjects_df = pd.read_csv("../../data/subject.csv")
    quality_df = pd.read_csv("../../data/image_quality_metrics.csv")

    # Compute CNR data
    cnr_data = batch_boundary_cr(subjects_df, base_path)

    if "sp_iz_cnr" in quality_df:
        quality_df = quality_df.drop(columns=["sp_iz_cnr"])

    # Merge CNR data into quality_df on subject_id, session_id, and split
    quality_df = quality_df.merge(
        cnr_data[["subject_id", "session_id", "split", "boundary_cr"]],
        on=["subject_id", "session_id", "split"],
        how="left",
    )

    # Rename boundary_cr to sp_iz_cnr
    quality_df.rename(columns={"boundary_cr": "sp_iz_cnr"}, inplace=True)

    # Save the updated dataframe
    quality_df.to_csv("../../data/image_quality_metrics.csv", index=False)

    print("\n")
    print("Successfully added sp_iz_cnr column to image_quality_metrics.csv")
    print(f"\nUpdated dataframe shape: {quality_df.shape}")
    print(f"\nFirst few rows of sp_iz_cnr column:")
    print(quality_df[["subject_id", "session_id", "split", "sp_iz_cnr"]].head(10))
