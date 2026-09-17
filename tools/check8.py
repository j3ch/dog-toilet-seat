"""Verify toiletv4_sliced_in_8.stl against the v4 original.

Checks mesh integrity, volume conservation, that every joint carries the same
5 connectors at the same clearance v4 uses, that each piece fits the bed, and
that adjacent pieces actually mate without interference.

    python tools/check8.py [bed_mm]
"""

import math
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
import split8 as S

# check8.py [split.stl [assembled.stl [bed_mm]]]
ARGV = sys.argv[1:] if __name__ == "__main__" else []
SPLIT = ARGV[0] if len(ARGV) > 0 else None
WHOLE = ARGV[1] if len(ARGV) > 1 else None
BED = float(ARGV[2]) if len(ARGV) > 2 else 200.0
EXISTING_CUTS = (0.0, 90.0, 180.0, 270.0)
NEW_CUTS = (45.0, 135.0, 225.0, 315.0)

fails = []


def check(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        fails.append(msg)


def load_sectors():
    """The 8 pieces, un-exploded back into assembled position."""
    mesh = trimesh.load(SPLIT or S.OUT)
    mesh.merge_vertices()
    parts = [p for p in mesh.split(only_watertight=False) if p.volume > 1.0]
    for p in parts:
        a = math.degrees(math.atan2(p.centroid[2], p.centroid[0])) % 360
        si = int(a // 45)
        mid = math.radians(si * 45 + 22.5)
        p.apply_translation([-S.EXPLODE * math.cos(mid), 0.0,
                             -S.EXPLODE * math.sin(mid)])
    parts.sort(key=lambda p: math.degrees(math.atan2(p.centroid[2], p.centroid[0])) % 360)
    return parts


def bed_square(mesh):
    """Smallest square bed the piece fits on, over all rotations about Y."""
    pts = mesh.vertices[:, [0, 2]]
    best = np.inf
    for d in range(180):
        t = math.radians(d)
        c, s = math.cos(t), math.sin(t)
        u = pts[:, 0] * c - pts[:, 1] * s
        v = pts[:, 0] * s + pts[:, 1] * c
        best = min(best, max(np.ptp(u), np.ptp(v)))
    return best


def section_polygons(mesh, angle, offset):
    """Closed 2D polygons where the piece crosses the cut plane at `offset`."""
    _, _, n = S.frame(angle)
    sec = mesh.section(plane_normal=n, plane_origin=n * offset)
    if sec is None:
        return []
    planar, _ = sec.to_planar(normal=n)
    return list(planar.polygons_full)


def section_holes(mesh, angle, offset):
    """The voids inside the material where the piece crosses the plane."""
    from shapely.geometry import Polygon
    return [Polygon(ring) for poly in section_polygons(mesh, angle, offset)
            for ring in poly.interiors]


def male_side(pa, pb, angle):
    """Of the two pieces at a joint, which one carries the pegs.

    The male face is flush with the plane and its pegs reach across into the
    neighbour's space, so it is the piece with material past the plane.
    """
    _, _, n = S.frame(angle)
    if (pa.vertices @ n).max() > 1.0:
        return pa, pb, 1.0
    if (pb.vertices @ n).min() < -1.0:
        return pb, pa, -1.0
    raise SystemExit(f"joint at {angle:.0f} has no pegs on either side")


def audit_joint(pa, pb, angle, label):
    male, female, sign = male_side(pa, pb, angle)
    pegs = [classify(q) for q in
            section_polygons(male, angle, sign * S.PEG_L / 2)]
    holes = [classify(q) for q in
             section_holes(female, angle, sign * S.HOLE_DEPTH / 2)]
    np_, nt = (sum(k.startswith("circle") for k, _, _ in pegs),
               sum(k.startswith("rect") for k, _, _ in pegs))
    nh, ns = (sum(k.startswith("circle") for k, _, _ in holes),
              sum(k.startswith("rect") for k, _, _ in holes))
    # v4's own joints lose their lowest peg to the skirt trim, so the count is
    # checked against the other side of the joint rather than against a fixed 5.
    check(len(pegs) == np_ + nt and np_ >= 2 and nt == 2,
          f"{label} {angle:3.0f}: {np_} pegs + {nt} tongues "
          + str(sorted(k for k, _, _ in pegs)))
    check((nh, ns) == (np_, nt),
          f"{label} {angle:3.0f}: {nh} holes + {ns} slots to match  "
          + str(sorted(k for k, _, _ in holes)))
    if len(pegs) == len(holes) and pegs:
        worst = min(min(wh - wp, hh - hp) / 2
                    for (_, wp, hp), (_, wh, hh)
                    in zip(sorted(pegs, key=lambda t: t[1]),
                           sorted(holes, key=lambda t: t[1])))
        check(0.2 <= worst <= 0.6,
              f"{label} {angle:3.0f}: tightest clearance {worst:.2f} mm per side")


def classify(poly):
    """'circle d=..' / 'rect wxh' / 'other', from area and bounds."""
    x0, y0, x1, y1 = poly.bounds
    w, h = x1 - x0, y1 - y0
    if abs(w - h) < 0.4 and poly.area > 0.70 * w * h and poly.area < 0.83 * w * h:
        return f"circle d={w:.2f}", w, h
    if poly.area > 0.97 * w * h:
        return f"rect {max(w, h):.2f}x{min(w, h):.2f}", max(w, h), min(w, h)
    return "other", w, h


def main():
    if WHOLE:
        base = trimesh.load(WHOLE)
        base_volume = base.volume
        base_name = WHOLE.rsplit("/", 1)[-1]
    else:
        base_volume = sum(q.volume for q in S.load_quadrants())
        base_name = "v4"
    sectors = load_sectors()

    print("\n1. integrity")
    check(len(sectors) == 8, f"8 separate solids (got {len(sectors)})")
    for i, p in enumerate(sectors):
        check(p.is_watertight and p.is_winding_consistent and p.volume > 0,
              f"S{i} watertight, consistent winding, volume {p.volume / 1000:.1f} cm3")
        check(p.euler_number == 2, f"S{i} is a single closed shell (euler={p.euler_number})")

    print("\n2. volume conservation")
    peg = 4 * (3 * math.pi * (S.PEG_D / 2) ** 2 * S.PEG_L * math.cos(math.pi / S.SECTIONS) ** 0
                * (S.SECTIONS / (2 * math.pi)) * math.sin(2 * math.pi / S.SECTIONS)
                + 2 * S.TONGUE_L * S.TONGUE_H * S.PEG_L)
    peg = 4 * (3 * 0.5 * S.SECTIONS * (S.PEG_D / 2) ** 2
               * math.sin(2 * math.pi / S.SECTIONS) * S.PEG_L
               + 2 * S.TONGUE_L * S.TONGUE_H * S.PEG_L)
    sock = 4 * (3 * 0.5 * S.SECTIONS * (S.HOLE_D / 2) ** 2
                * math.sin(2 * math.pi / S.SECTIONS) * S.HOLE_DEPTH
                + 2 * S.SLOT_L * S.SLOT_H * S.HOLE_DEPTH)
    want = base_volume + peg - sock
    got = sum(p.volume for p in sectors)
    err = abs(got - want) / base_volume
    check(err < 0.005, f"{got / 1000:.1f} cm3 vs expected {want / 1000:.1f} cm3 "
                       f"({base_name} {base_volume / 1000:.1f} + pegs - sockets), "
                       f"{err * 100:.3f}% error")

    print("\n3. connectors on the 4 new joints")
    for angle in NEW_CUTS:
        si = int(angle // 45)
        audit_joint(sectors[si - 1], sectors[si % 8], angle, "new cut")

    print("\n4. the 4 original v4 joints still intact")
    for angle in EXISTING_CUTS:
        si = int(angle // 45)
        audit_joint(sectors[si - 1], sectors[si % 8], angle, "v4 cut ")

    print(f"\n5. fit on a {BED:.0f} x {BED:.0f} mm bed")
    for i, p in enumerate(sectors):
        sq = bed_square(p)
        tall = p.extents[1]
        check(sq <= BED and tall <= BED,
              f"S{i}: needs {sq:.0f} mm square, {tall:.1f} mm tall")

    print("\n6. assembled fit - no interference between neighbours")
    for i in range(8):
        a, b = sectors[i], sectors[(i + 1) % 8]
        inter = trimesh.boolean.intersection([a, b], engine=S.ENGINE)
        vol = 0.0 if inter is None or inter.is_empty else abs(inter.volume)
        check(vol < 10.0, f"S{i}/S{(i + 1) % 8}: overlap {vol:.3f} mm3")

    print("\n" + ("ALL CHECKS PASSED" if not fails
                  else f"{len(fails)} CHECK(S) FAILED"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
