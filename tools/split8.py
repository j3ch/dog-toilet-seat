"""Re-split the v4 seat from 4 printable pieces into 8.

toiletv4_sliced.stl holds the seat already cut into quadrants by the planes
X=0 and Z=0, exported exploded 30 mm outward.  This reassembles it, adds a
radial cut through each quadrant at 45/135/225/315 degrees, and rebuilds on the
new faces the same 5-connector joint the existing cuts use -- 3 round pegs down
the skirt and 2 flat tongues in the flange, 0.5 mm clearance -- then re-explodes
the 8 pieces for export.

    python tools/split8.py
"""

import math
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent.parent
SRC = str(ROOT / "toiletv4_sliced.stl")
OUT = str(ROOT / "toiletv4_sliced_in_8.stl")
ENGINE = "manifold"

# --- joint geometry, measured off the existing v4 cut faces -----------------
PEG_D, PEG_L = 5.0, 6.0          # male round peg
HOLE_D, HOLE_DEPTH = 6.0, 7.0    # female socket
TONGUE_L, TONGUE_H = 35.0, 4.0   # male flange tongue
SLOT_L, SLOT_H = 36.0, 5.0       # female flange slot
TONGUE_Y = 6.0                   # tongue mid-height (spans y = 4..8)
PEG_YS = (-25.0, -47.0, -70.0)   # peg heights down the skirt
SECTIONS = 24                    # facets per peg, matching v4

# v4's own sockets leave 1.5-2.0 mm of wall.  The diagonal cuts meet the ~10.5 mm
# skirt obliquely, so the bore drifts across the wall as it goes in and a little
# less is available; 1.0 mm is still 2-3 perimeters.  Actual values are printed.
MIN_WALL = 1.0
TONGUE_GAP = 13.0                # material between the two tongues, as v4
BACKING = 2.0                    # male material required behind a peg root
EXPLODE = 40.0                   # radial explode distance for the export


def frame(angle_deg):
    """(u, v, n) for the radial cut plane at `angle_deg`.

    u runs outward along the plane, v is up, n is the plane normal pointing
    toward increasing angle.  Points with n.p > 0 are on the high-angle side.
    """
    a = math.radians(angle_deg)
    return (np.array([math.cos(a), 0.0, math.sin(a)]),
            np.array([0.0, 1.0, 0.0]),
            np.array([-math.sin(a), 0.0, math.cos(a)]))


def placement_transform(u, v, n, origin):
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2] = u, v, n
    m[:3, 3] = origin
    return m


def load_quadrants():
    """The 4 v4 quadrants, translated back into assembled position.

    Returned ordered by bisector angle, so index i covers i*90 .. i*90+90.
    """
    mesh = trimesh.load(SRC)
    mesh.merge_vertices()
    parts = [p for p in mesh.split(only_watertight=False) if p.volume > 1.0]
    if len(parts) != 4:
        raise SystemExit(f"expected 4 solid components in {SRC}, got {len(parts)}")
    for p in parts:
        cx, _, cz = p.centroid
        p.apply_translation([-30.0 if cx > 0 else 30.0, 0.0,
                             -30.0 if cz > 0 else 30.0])
        if not p.is_watertight:
            raise SystemExit("quadrant is not watertight")
    parts.sort(key=lambda p: math.degrees(math.atan2(p.centroid[2], p.centroid[0])) % 360)
    return parts


def spans(mesh, u, v, y, offset, r0=60.0, r1=300.0, step=0.25):
    """Radial intervals of solid material at height `y`, `offset` off the plane.

    Never sample at offset 0: points exactly on the cap plane sit on the mesh
    boundary, where an inside/outside test is ambiguous.
    """
    rs = np.arange(r0, r1, step)
    hit = mesh.contains(np.outer(rs, u) + y * v + offset)
    out, start = [], None
    for i, h in enumerate(hit):
        if h and start is None:
            start = rs[i]
        elif not h and start is not None:
            out.append((start, rs[i]))
            start = None
    if start is not None:
        out.append((start, rs[-1]))
    return out


def common_interval(mesh, u, v, n, y, offsets):
    """Radial band of material shared by every one of `offsets` along n."""
    lo, hi = -np.inf, np.inf
    for off in offsets:
        cand = spans(mesh, u, v, y, off * n)
        if not cand:
            return None
        a, b = max(cand, key=lambda sp: sp[1] - sp[0])
        lo, hi = max(lo, a), min(hi, b)
    return (lo, hi) if hi > lo else None


def centre_range(mesh_low, mesh_high, u, v, n, y, half_male, half_female):
    """Radial positions where a connector of this width fits both halves.

    The male peg has to sit inside the cut face (and a little material behind
    it); the female socket has to stay buried over its whole depth, which is the
    binding constraint -- the wall curves away from the cut as it runs on.
    """
    face = common_interval(mesh_low, u, v, n, y, (-0.1, -BACKING))
    sock = common_interval(mesh_high, u, v, n, y, (0.1, HOLE_DEPTH / 2, HOLE_DEPTH))
    if face is None or sock is None:
        return None
    lo = max(face[0] + half_male, sock[0] + half_female)
    hi = min(face[1] - half_male, sock[1] - half_female)
    return (lo, hi) if hi >= lo else None


def joint_features(mesh_low, mesh_high, angle):
    """Peg and tongue placements for the new cut at `angle`.

    Positions are measured from the material actually present rather than
    hard-coded, so each connector stays centred in its wall.
    """
    u, v, n = frame(angle)
    feats = []

    for y in PEG_YS:
        rng = centre_range(mesh_low, mesh_high, u, v, n, y,
                           PEG_D / 2 + MIN_WALL, HOLE_D / 2 + MIN_WALL)
        if rng is None:
            raise SystemExit(f"cut {angle:.0f}: skirt at y={y:.0f} is too thin or "
                             f"drifts too far to hold a peg with {MIN_WALL} mm wall")
        lo, hi = rng
        feats.append(("peg", (lo + hi) / 2.0, y, (hi - lo) / 2.0 + MIN_WALL))

    rng = centre_range(mesh_low, mesh_high, u, v, n, TONGUE_Y,
                       TONGUE_L / 2 + MIN_WALL, SLOT_L / 2 + MIN_WALL)
    if rng is None:
        raise SystemExit(f"cut {angle:.0f}: flange cannot hold a tongue")
    lo, hi = rng
    # Two tongues, spread across whatever span the flange gives so the wide
    # corner flanges are held near both edges rather than only in the middle,
    # but never closer than v4's own slot spacing.
    separation = min(hi - lo, max(SLOT_L + TONGUE_GAP, (hi - lo) * 0.6))
    if hi - lo < SLOT_L + TONGUE_GAP:
        raise SystemExit(f"cut {angle:.0f}: flange has room for one tongue, not two "
                         f"(usable span {hi - lo:.1f} mm, "
                         f"need {SLOT_L + TONGUE_GAP:.1f})")
    mid = (lo + hi) / 2.0
    for r in (mid - separation / 2.0, mid + separation / 2.0):
        feats.append(("tongue", r, TONGUE_Y, (hi - lo - separation) / 2.0 + MIN_WALL))
    return feats


def check_buried(mesh, u, v, n, r, y, half_len, half_h, depth, label, round_=False):
    """Assert the body really holds this connector over its whole depth.

    `round_` samples a disc rather than its enclosing rectangle, so the corners
    a cylinder never occupies do not fail the check.
    """
    rr = np.linspace(r - half_len, r + half_len, 9)
    yy = np.linspace(y - half_h, y + half_h, 9)
    face = [(a, b) for a in rr for b in yy
            if not round_ or
            ((a - r) / half_len) ** 2 + ((b - y) / half_h) ** 2 <= 1.0 + 1e-9]
    dd = np.linspace(0.2, depth, 6)
    grid = np.array([a * u + b * v + d * n for a, b in face for d in dd])
    missing = int((~mesh.contains(grid)).sum())
    if missing:
        raise SystemExit(f"{label}: {missing}/{len(grid)} sample points outside the "
                         "body - connector would break through")


def verify_joint(mesh_low, mesh_high, angle, feats):
    u, v, n = frame(angle)
    for kind, r, y, _ in feats:
        rnd = kind == "peg"
        hl, hh = ((PEG_D / 2, PEG_D / 2) if rnd else (TONGUE_L / 2, TONGUE_H / 2))
        check_buried(mesh_low, u, v, -n, r, y, hl, hh, BACKING,
                     f"cut {angle:.0f} {kind} r={r:.1f} y={y:.0f} (male root)", rnd)
        hl, hh = ((HOLE_D / 2, HOLE_D / 2) if rnd else (SLOT_L / 2, SLOT_H / 2))
        check_buried(mesh_high, u, v, n, r, y, hl, hh, HOLE_DEPTH,
                     f"cut {angle:.0f} {kind} r={r:.1f} y={y:.0f} (female socket)", rnd)


def connector_solids(angle, feats, male):
    """Peg solids (male=True) or socket solids (male=False) for one joint.

    Both grow from the plane in the +n direction: the pegs reach across into the
    neighbour's space, the sockets bore into the female body.
    """
    u, v, n = frame(angle)
    out = []
    for kind, r, y, _ in feats:
        if kind == "peg":
            d, h = (PEG_D, PEG_L) if male else (HOLE_D, HOLE_DEPTH)
            out.append(trimesh.creation.cylinder(
                radius=d / 2, height=h, sections=SECTIONS,
                transform=placement_transform(u, v, n, r * u + y * v + n * (h / 2))))
        else:
            le, hi, h = ((TONGUE_L, TONGUE_H, PEG_L) if male
                         else (SLOT_L, SLOT_H, HOLE_DEPTH))
            out.append(trimesh.creation.box(
                extents=[le, hi, h],
                transform=placement_transform(u, v, n, r * u + y * v + n * (h / 2))))
    return out


def main():
    quads = load_quadrants()
    print(f"loaded 4 quadrants, {sum(q.volume for q in quads) / 1000:.1f} cm3 total")

    pieces = {}
    for qi, quad in enumerate(quads):
        angle = qi * 90 + 45.0          # bisector of this quadrant = the new cut
        _, _, n = frame(angle)
        high = trimesh.intersections.slice_mesh_plane(quad, n, [0, 0, 0], cap=True)
        low = trimesh.intersections.slice_mesh_plane(quad, -n, [0, 0, 0], cap=True)
        for half, name in ((low, "low"), (high, "high")):
            if not half.is_watertight:
                raise SystemExit(f"cut {angle:.0f}: {name} half is not watertight")

        feats = joint_features(low, high, angle)
        verify_joint(low, high, angle, feats)
        print(f"cut {angle:5.0f} deg  " + "  ".join(
            f"{k}(r={r:.1f},y={y:.0f},wall>={w:.2f})" for k, r, y, w in feats))

        # lower-angle side is male, higher-angle side is female
        low = trimesh.boolean.union([low] + connector_solids(angle, feats, True),
                                    engine=ENGINE)
        high = trimesh.boolean.difference([high] + connector_solids(angle, feats, False),
                                          engine=ENGINE)
        pieces[2 * qi], pieces[2 * qi + 1] = low, high

    out = []
    for si in sorted(pieces):
        p = pieces[si]
        mid = math.radians(si * 45 + 22.5)
        p.apply_translation([EXPLODE * math.cos(mid), 0.0, EXPLODE * math.sin(mid)])
        if not p.is_watertight:
            raise SystemExit(f"sector {si} is not watertight after the booleans")
        out.append(p)
        print(f"  S{si}: {len(p.faces):6d} faces, {p.volume / 1000:7.1f} cm3")

    trimesh.util.concatenate(out).export(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
