"""
Generate native-space subplate surfaces from recon-space `default_surfaces`.

The subplate surfaces (`{lh,rh}.innersp.obj`, `{lh,rh}.wm..obj`) live in each
split's `default_surfaces/` in *recon* (post-reconstruction) space. The cortical
plate surfaces in `surfaces/` already have `*.native_81920.obj` variants; this
script produces the equivalent native-space variants for the subplate surfaces.

A surface is a list of vertex XYZ coordinates plus normals and triangle
connectivity, so "registering" it means applying the transform to each vertex
(and the inverse-transpose to each normal) — no volume resampling. We apply
`recon_segmentation/recon_native.xfm` *forward*; this was verified against the CP
surfaces, where forward(recon-space) reproduces the shipped `*.native_81920.obj`
to ~0.02 mm (the inverse is off by ~14 mm).

Outputs (written next to the sources, in each `default_surfaces/`):
    {lh,rh}.innersp.native.obj   (inner SP boundary, native space)
    {lh,rh}.wm..native.obj       (outer SP boundary == WM surface, native space)
    {lh,rh}.innersp.native.thk   (per-vertex subplate thickness = |innersp - wm|,
                                   one value per vertex, native mm — colour scalar)

Vertex i of the innersp mesh corresponds to vertex i of the wm mesh (innersp is
deformed from wm), so the per-vertex Euclidean distance between the two native
meshes is the subplate thickness at that vertex.

Usage:
    uv run python src/processing/generate_native_surfaces.py            # all subjects
    uv run python src/processing/generate_native_surfaces.py --dry-run  # report only
    uv run python src/processing/generate_native_surfaces.py --limit 1  # first subject only
    uv run python src/processing/generate_native_surfaces.py --overwrite
"""

import argparse
import os
import re

import numpy as np
import pandas as pd

BASE_PATH = "/neuro/labs/grantlab/research/MRI_processing/seungyoon.jeong/2025/Reliability/TEST/"
SPLITS = ["S1", "S2", "S3", "S4"]
HEMIS = ["lh", "rh"]
# Base surface filenames in default_surfaces/ that we bring into native space.
SURFACES = ["innersp.obj", "wm..obj"]
SUBJECT_CSV = os.path.join(os.path.dirname(__file__), "..", "..", "data", "subject.csv")


def read_native_xfm(path: str) -> np.ndarray | None:
    """Parse an MNI Transform File and return the 3x4 linear matrix (R|t), or None."""
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


def read_obj_vertices(path: str) -> np.ndarray:
    """Read vertex coordinates (N,3) from an MNI `.obj` surface file."""
    with open(path) as f:
        n = int(f.readline().split()[-1])
        verts = np.empty((n, 3), dtype=np.float64)
        for i in range(n):
            verts[i] = [float(x) for x in f.readline().split()]
    return verts


def _fmt(rows: np.ndarray) -> list[str]:
    """Format an (N,3) array as MNI-obj coordinate lines (leading space, 6 sig figs)."""
    return [f" {x:.6g} {y:.6g} {z:.6g}" for x, y, z in rows]


def transform_obj(in_path: str, out_path: str, xfm: np.ndarray) -> np.ndarray:
    """Read an MNI `.obj`, transform vertices by `xfm` and normals by its
    inverse-transpose, copy connectivity verbatim, and write to `out_path`.

    Returns the (N,3) array of native-space vertices.
    """
    raw = open(in_path).read().split("\n")
    header = raw[0].split()
    if not header or header[0] != "P":
        raise ValueError(f"not a polygon .obj (header={raw[0]!r})")
    n = int(header[-1])

    # Layout: [0]=header, [1:1+n]=vertices, [1+n]=blank, then n normals, then
    # a blank line and the connectivity block. Parse defensively.
    v0 = 1
    verts = np.array([list(map(float, raw[v0 + i].split())) for i in range(n)],
                     dtype=np.float64)
    sep = v0 + n
    if raw[sep].strip() != "":
        raise ValueError(f"expected blank line after vertices at index {sep}")
    nrm0 = sep + 1
    normals = np.array([list(map(float, raw[nrm0 + i].split())) for i in range(n)],
                       dtype=np.float64)
    rest = raw[nrm0 + n:]  # begins with the blank line before the connectivity block

    # Vertices: full affine. Normals: inverse-transpose of the linear part, renormalized.
    L = xfm[:, :3]
    verts_n = verts @ L.T + xfm[:, 3]
    normals_n = normals @ np.linalg.inv(L)  # == normals @ (inv(L).T).T
    norms = np.linalg.norm(normals_n, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normals_n /= norms

    out_lines = [raw[0], *_fmt(verts_n), "", *_fmt(normals_n), *rest]
    with open(out_path, "w") as f:
        f.write("\n".join(out_lines))
    return verts_n


def native_name(base: str) -> str:
    """`innersp.obj` -> `innersp.native.obj` (insert `.native` before `.obj`)."""
    assert base.endswith(".obj")
    return base[:-len(".obj")] + ".native.obj"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report what would be done, write nothing")
    ap.add_argument("--overwrite", action="store_true", help="regenerate even if native file exists")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N subjects")
    args = ap.parse_args()

    subjects = pd.read_csv(SUBJECT_CSV)
    if args.limit:
        subjects = subjects.head(args.limit)

    n_obj = n_thk = n_skipped = n_missing_src = n_no_xfm = 0
    for _, r in subjects.iterrows():
        subject_id, session_id = str(r["subject_id"]), str(r["session_id"])
        for split in SPLITS:
            split_dir = os.path.join(BASE_PATH, subject_id, session_id, split)
            sp_dir = os.path.join(split_dir, "default_surfaces")
            xfm_path = os.path.join(split_dir, "recon_segmentation", "recon_native.xfm")
            xfm = read_native_xfm(xfm_path)

            for hemi in HEMIS:
                srcs = {b: os.path.join(sp_dir, f"{hemi}.{b}") for b in SURFACES}
                if not all(os.path.isfile(p) for p in srcs.values()):
                    n_missing_src += 1
                    continue
                if xfm is None:
                    n_no_xfm += 1
                    print(f"  NO XFM  {subject_id}/{session_id}/{split} {hemi}")
                    continue

                # Native vertices per surface: reuse if the .obj already exists,
                # else transform (and write) it.
                native_verts: dict[str, np.ndarray] = {}
                for base, src in srcs.items():
                    dst = os.path.join(sp_dir, f"{hemi}.{native_name(base)}")
                    try:
                        if os.path.isfile(dst) and not args.overwrite:
                            n_skipped += 1
                            native_verts[base] = read_obj_vertices(dst)
                        elif args.dry_run:
                            print(f"  WOULD WRITE {dst}")
                            n_obj += 1
                            native_verts[base] = read_obj_vertices(src) @ xfm[:, :3].T + xfm[:, 3]
                        else:
                            native_verts[base] = transform_obj(src, dst, xfm)
                            n_obj += 1
                            print(f"  wrote {dst}")
                    except Exception as e:
                        print(f"  FAILED {dst}: {e}")

                # Per-vertex subplate thickness = |innersp - wm| in native space.
                thk_dst = os.path.join(sp_dir, f"{hemi}.{native_name('innersp.obj')[:-4]}.thk")
                if "innersp.obj" in native_verts and "wm..obj" in native_verts:
                    inn, wm = native_verts["innersp.obj"], native_verts["wm..obj"]
                    if inn.shape == wm.shape:
                        if os.path.isfile(thk_dst) and not args.overwrite:
                            pass
                        elif args.dry_run:
                            print(f"  WOULD WRITE {thk_dst}")
                            n_thk += 1
                        else:
                            d = np.linalg.norm(inn - wm, axis=1)
                            np.savetxt(thk_dst, d, fmt="%.6f")
                            n_thk += 1
                            print(f"  wrote {thk_dst}  (mean {d.mean():.3f} mm)")
                    else:
                        print(f"  SHAPE MISMATCH {thk_dst}: {inn.shape} vs {wm.shape}")

    print("\n--- summary ---")
    print(f"native .obj written{'(dry-run)' if args.dry_run else ''} : {n_obj}")
    print(f"native .thk written{'(dry-run)' if args.dry_run else ''} : {n_thk}")
    print(f".obj skipped (exists)   : {n_skipped}")
    print(f"missing source .obj     : {n_missing_src}")
    print(f"missing/invalid xfm     : {n_no_xfm}")


if __name__ == "__main__":
    main()
