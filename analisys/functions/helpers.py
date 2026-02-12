FREESURFER_LUT_PATH = "/neuro/labs/grantlab/research/MRI_processing/yair.beltran/Reliability/FreeSurferColorLUT.txt"
# FREESURFER_LUT_PATH = "/home/yair/Projects/Reliability/FreeSurferColorLUT.txt"


import numpy as np
from scipy.spatial.distance import directed_hausdorff


def get_middle_slice(data, axis):
    return data.shape[axis] // 2


def normalize_intensity(img_data, lower_percentile=1, upper_percentile=99):
    lower, upper = np.percentile(img_data, [lower_percentile, upper_percentile])
    img_normalized = np.clip(img_data, lower, upper)
    return (img_normalized - lower) / (upper - lower)


def dice_coefficient(seg1, seg2):
    intersection = np.sum(seg1 * seg2)
    return (
        2.0 * intersection / (np.sum(seg1) + np.sum(seg2))
        if (np.sum(seg1) + np.sum(seg2)) > 0
        else 0
    )


def hausdorff_distance(seg1, seg2):
    coords1, coords2 = np.argwhere(seg1), np.argwhere(seg2)
    if len(coords1) == 0 or len(coords2) == 0:
        return np.inf
    return max(
        directed_hausdorff(coords1, coords2)[0], directed_hausdorff(coords2, coords1)[0]
    )


def relative_voxel_difference(seg1, seg2):
    count1 = np.sum(seg1)
    count2 = np.sum(seg2)

    if count1 == 0 and count2 == 0:
        return 0.0

    # Use a mean count of boxels instead of voxels of a single count of voxels of a segmentation
    mean_count = (count1 + count2) / 2.0
    if mean_count == 0:
        return 0.0

    return abs(count1 - count2) / mean_count * 100.0


def load_freesurfer_lut(path):
    lut_colors = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        label_idx = int(parts[0])
                        lut_colors[label_idx] = (
                            int(parts[2]) / 255.0,
                            int(parts[3]) / 255.0,
                            int(parts[4]) / 255.0,
                            1.0,
                        )
                    except (ValueError, IndexError):
                        continue
    lut_colors[0] = (0, 0, 0, 0)
    return lut_colors
