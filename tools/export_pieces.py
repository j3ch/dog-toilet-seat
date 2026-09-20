"""Write each piece of a split seat to its own STL, ready to load in a slicer.

The split files hold all 8 pieces in one mesh, in assembled-and-exploded
positions with Y up.  Slicers want one solid per file, Z up, sitting on the bed.
Each piece is converted to Z up, turned about the vertical axis to its smallest
square footprint, centred on the origin and dropped onto Z=0.

Pieces are turned over by default so the large ridged face is toward the bed and
the skirt points up: printed the other way up, the flange cantilevers out 80 mm
above the thin bottom edge of the skirt and needs support under all of it.  Pass
--upright to keep them the way the seat is used.

    python tools/export_pieces.py [split.stl] [outdir] [--upright]
"""

import math
import sys
from pathlib import Path

import numpy as np
import trimesh

import split8 as S

ROOT = Path(__file__).resolve().parent.parent
ARGV = [a for a in sys.argv[1:] if not a.startswith("--")]
FLIP = "--upright" not in sys.argv[1:]
SRC = Path(ARGV[0]) if len(ARGV) > 0 else ROOT / "toiletv4_ridged_in_8.stl"
OUTDIR = Path(ARGV[1]) if len(ARGV) > 1 else ROOT / "pieces"


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
    if len(parts) not in (4, 8):
        raise SystemExit(f"expected 4 or 8 pieces in {SRC.name}, got {len(parts)}")
    # name pieces by the sector they came from, as the split and check scripts do
    parts.sort(key=lambda p: math.degrees(math.atan2(p.centroid[2], p.centroid[0])) % 360)

    OUTDIR.mkdir(exist_ok=True)
    print(f"{SRC.name} -> {OUTDIR.name}/"
          f"   ({'flipped, ridged face down' if FLIP else 'upright, skirt down'})\n")
    print(f"{'file':38s} {'footprint mm':>16s} {'tall':>7s} {'volume':>9s} {'bed contact':>12s}")
    total = 0.0
    for si, p in enumerate(parts):
        # A 180 degree turn, never a mirror: mirroring would reverse every peg
        # and socket and the pieces would no longer mate.  Asserted below.
        before = p.volume
        placed = trimesh.transformations.rotation_matrix(
            -math.pi / 2 if FLIP else math.pi / 2, [1, 0, 0])
        p.apply_transform(placed)
        spin = trimesh.transformations.rotation_matrix(
            math.radians(best_spin(p)), [0, 0, 1])
        p.apply_transform(spin)
        det = np.linalg.det((spin @ placed)[:3, :3])
        if abs(det - 1.0) > 1e-9 or abs(p.volume - before) > 1e-6 * abs(before):
            raise SystemExit(f"S{si}: placement is not a pure rotation (det {det:.6f})")
        lo, hi = p.bounds
        p.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
        if not p.is_watertight:
            raise SystemExit(f"S{si} is not watertight")
        tag = "Q" if len(parts) == 4 else "S"
        stem = SRC.stem.replace("_in_8", "").replace("_in_4", "")
        name = f"{stem}_{tag}{si}.stl"
        p.export(str(OUTDIR / name))
        w, d, h = p.extents
        total += p.volume
        sec = p.section(plane_normal=[0, 0, 1], plane_origin=[0, 0, 0.2])
        contact = 0.0
        if sec is not None:
            planar, _ = sec.to_planar(normal=[0, 0, 1])
            contact = sum(q.area for q in planar.polygons_full)
        print(f"{name:38s} {w:7.1f} x {d:6.1f} {h:7.1f} {p.volume / 1000:7.1f} cm3"
              f" {contact:8.0f} mm2")
    print(f"\n{len(parts)} files, {total / 1000:.1f} cm3 total")


if __name__ == "__main__":
    main()
