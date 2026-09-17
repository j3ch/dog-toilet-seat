"""Write each piece of a split seat to its own STL, ready to load in a slicer.

The split files hold all 8 pieces in one mesh, in assembled-and-exploded
positions with Y up.  Slicers want one solid per file, Z up, sitting on the bed.
Each piece is converted to Z up, turned about the vertical axis to its smallest
square footprint, centred on the origin and dropped onto Z=0.

    python tools/export_pieces.py [split.stl] [outdir]
"""

import math
import sys
from pathlib import Path

import numpy as np
import trimesh

import split8 as S

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "toiletv4_ridged_in_8.stl"
OUTDIR = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "pieces"


def best_spin(mesh):
    """Rotation about Z, in degrees, giving the smallest square footprint."""
    pts = mesh.vertices[:, :2]
    best, angle = np.inf, 0
    for d in range(180):
        t = math.radians(d)
        c, s = math.cos(t), math.sin(t)
        u = pts[:, 0] * c - pts[:, 1] * s
        v = pts[:, 0] * s + pts[:, 1] * c
        side = max(np.ptp(u), np.ptp(v))
        if side < best:
            best, angle = side, d
    return angle


def main():
    mesh = trimesh.load(str(SRC))
    mesh.merge_vertices()
    parts = [p for p in mesh.split(only_watertight=False) if p.volume > 1.0]
    if len(parts) != 8:
        raise SystemExit(f"expected 8 pieces in {SRC.name}, got {len(parts)}")
    # name pieces by the sector they came from, as the split and check scripts do
    parts.sort(key=lambda p: math.degrees(math.atan2(p.centroid[2], p.centroid[0])) % 360)

    OUTDIR.mkdir(exist_ok=True)
    print(f"{SRC.name} -> {OUTDIR.name}/\n")
    print(f"{'file':38s} {'footprint mm':>16s} {'tall':>7s} {'volume':>9s}")
    total = 0.0
    for si, p in enumerate(parts):
        p.apply_transform(trimesh.transformations.rotation_matrix(
            math.pi / 2, [1, 0, 0]))                       # model Y up -> Z up
        p.apply_transform(trimesh.transformations.rotation_matrix(
            math.radians(best_spin(p)), [0, 0, 1]))
        lo, hi = p.bounds
        p.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
        if not p.is_watertight:
            raise SystemExit(f"S{si} is not watertight")
        name = f"{SRC.stem.replace('_in_8', '')}_S{si}.stl"
        p.export(str(OUTDIR / name))
        w, d, h = p.extents
        total += p.volume
        print(f"{name:38s} {w:7.1f} x {d:6.1f} {h:7.1f} {p.volume / 1000:7.1f} cm3")
    print(f"\n8 files, {total / 1000:.1f} cm3 total")


if __name__ == "__main__":
    main()
