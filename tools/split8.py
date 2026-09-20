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
import shapely
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
PEG_YS = (-22.0, -45.0, -60.0)   # nominal peg heights, as v4's top two plus one
PEG_Y_SEARCH = 6.0               # how far a peg may be nudged to find sound wall
SECTIONS = 24                    # facets per peg, matching v4

# v4's own sockets leave 1.5-2.0 mm of wall.  The diagonal cuts meet the ~10.5 mm
# skirt obliquely, so the bore drifts across the wall as it goes in and a little
# less is available; 1.0 mm is still 2-3 perimeters.  Actual values are printed.
MIN_WALL = 1.0
TONGUE_GAP = 13.0                # material between the two tongues, as v4
BACKING = 2.0                    # male material required behind a peg root
# Depths the cross-section is taken at, on each side of the cut.
MALE_PROBES = (-0.1, -1.0, -BACKING)
FEMALE_PROBES = (0.1, 1.5, 3.0, 4.5, 6.0, 7.0)
EXPLODE = 40.0                   # radial explode distance for the export

# The bowl this is for takes at most 70 mm of skirt; v4's is 80 mm.  The lowest
# connector is not at the same height on every original mating face - the four
# sockets top out at -68.0, -67.0, -67.0 and -65.905 - so the deepest cut that
# bisects none of them is 0.5 mm above the highest, leaving 65.4 mm of skirt.
SKIRT_TRIM_Y = -65.4

# v4's top is a shallow dish: it rises about 2.7 mm from the inner edge out to
# the rim, so ridges laid on it are not coplanar.  Filling it up to its own high
# point makes the standing surface a true plane and takes nothing away.  The
# slab starts above the 8.5 mm flange slots so the joints are untouched, and its
# footprint is read below them, so they do not punch holes in it.
FLAT_TOP = True
FLAT_FLOOR = 8.6
FLAT_FOOTPRINT_YS = (2.0, 9.0)
# The section outline comes back with a few zero-length segments, which split
# the slab's triangulation into separate shells; this drops them without moving
# the outline (volume is identical to a tenth of a mm3).
FLAT_SIMPLIFY = 0.001
# Hold the slab a hair inside the outline.  Flush, its wall lands exactly on the
# part's own outer wall, and that coincident pair survives the booleans only to
# come apart in the ridge union - 48 non-manifold edges along it.  The cost is a
# 0.25 mm strip of the original rounded rim left at the very edge.
FLAT_INSET = 0.25


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
    if SKIRT_TRIM_Y is not None:
        # Boolean against a box rather than a capped plane slice: the slicer's
        # cap triangulation leaves hair-thin slivers elsewhere on the part.
        keep = trimesh.creation.box(
            extents=[2000.0, 400.0, 2000.0],
            transform=trimesh.transformations.translation_matrix(
                [0.0, SKIRT_TRIM_Y + 200.0, 0.0]))
        parts = [trimesh.boolean.intersection([p, keep], engine=ENGINE) for p in parts]
        for p in parts:
            if not p.is_watertight:
                raise SystemExit("quadrant is not watertight after the skirt trim")
    if FLAT_TOP:
        level = max(p.bounds[1][1] for p in parts)
        parts = [flatten_top(p, level) for p in parts]
        for p in parts:
            if not p.is_watertight:
                raise SystemExit("quadrant is not watertight after flattening")
    parts.sort(key=lambda p: math.degrees(math.atan2(p.centroid[2], p.centroid[0])) % 360)
    return parts


def profile(mesh, angle, offset):
    """Cross-section of `mesh` at `offset` from the cut plane, in (r, y).

    Taken as an exact section polygon rather than by sampling `contains`: once
    the skirt trim adds a cap face, ray casting misclassifies enough points to
    make the measured wall jump around by several mm between adjacent heights.
    """
    u, v, n = frame(angle)
    sec = mesh.section(plane_normal=n, plane_origin=n * offset)
    if sec is None:
        return None
    planar, to_3d = sec.to_planar(normal=n)

    def conv(coords):
        a = np.asarray(coords)
        w = np.column_stack([a[:, 0], a[:, 1], np.zeros(len(a)), np.ones(len(a))]) @ to_3d.T
        return np.column_stack([w[:, :3] @ u, w[:, :3] @ v])

    polys = [shapely.Polygon(conv(p.exterior.coords),
                             [conv(r.coords) for r in p.interiors])
             for p in planar.polygons_full]
    return shapely.union_all(polys) if polys else None


def xz_outline(mesh, y):
    """The solid's cross-section at height `y`, as a polygon in world X/Z."""
    sec = mesh.section(plane_normal=[0, 1, 0], plane_origin=[0, y, 0])
    if sec is None:
        return None
    planar, to_3d = sec.to_planar(normal=[0, 1, 0])

    def conv(coords):
        a = np.asarray(coords)
        w = np.column_stack([a[:, 0], a[:, 1], np.zeros(len(a)), np.ones(len(a))]) @ to_3d.T
        return w[:, [0, 2]]

    polys = [shapely.Polygon(conv(p.exterior.coords),
                             [conv(r.coords) for r in p.interiors])
             for p in planar.polygons_full]
    return shapely.union_all(polys) if polys else None


def flatten_top(mesh, level):
    """Fill the dished top up to a flat plane at `level`.

    The footprint is the union of cross-sections taken below the flange slots
    and just above them: below, the outline is clean but slightly drafted in;
    above, it is full width.  Taken at slot height it would come back with
    slot-shaped holes and notch the new surface.
    """
    foot = shapely.union_all([o for o in
                              (xz_outline(mesh, y) for y in FLAT_FOOTPRINT_YS)
                              if o is not None])
    foot = foot.buffer(-FLAT_INSET)
    slabs = []
    for part in getattr(foot, "geoms", [foot]):
        part = part.simplify(FLAT_SIMPLIFY)
        p = trimesh.creation.extrude_polygon(part, height=level - FLAT_FLOOR)
        if not p.is_watertight:
            raise SystemExit("flattening slab is not a closed volume")
        p.apply_transform(trimesh.transformations.rotation_matrix(
            math.pi / 2, [1, 0, 0]))
        p.apply_translation([0.0, level, 0.0])
        slabs.append(p)
    return trimesh.boolean.union([mesh] + slabs, engine=ENGINE)


def spans(region, y):
    """Radial intervals of material at height `y`."""
    cut = shapely.LineString([(-1e4, y), (1e4, y)]).intersection(region)
    out = []
    for g in getattr(cut, "geoms", [cut]):
        if g.is_empty or g.geom_type != "LineString":
            continue
        xs = [c[0] for c in g.coords]
        out.append((min(xs), max(xs)))
    return out


def common_interval(regions, y):
    """Radial band of material shared by every one of these cross-sections."""
    lo, hi = -np.inf, np.inf
    for region in regions:
        cand = spans(region, y)
        if not cand:
            return None
        a, b = max(cand, key=lambda sp: sp[1] - sp[0])
        lo, hi = max(lo, a), min(hi, b)
    return (lo, hi) if hi > lo else None


def centre_range(male, female, y, half_male, half_female):
    """Radial positions where a connector of this width fits both halves.

    The male peg has to sit inside the cut face (and a little material behind
    it); the female socket has to stay buried over its whole depth, which is the
    binding constraint - the wall curves away from the cut as it runs on.
    """
    face = common_interval(male, y)
    sock = common_interval(female, y)
    if face is None or sock is None:
        return None
    lo = max(face[0] + half_male, sock[0] + half_female)
    hi = min(face[1] - half_male, sock[1] - half_female)
    return (lo, hi) if hi >= lo else None


def footprint(r, y, half_len, half_h, round_):
    if round_:
        return shapely.Point(r, y).buffer(half_len, quad_segs=SECTIONS // 2)
    return shapely.box(r - half_len, y - half_h, r + half_len, y + half_h)


def joint_features(male, female, angle):
    """Peg and tongue placements for the new cut at `angle`.

    Positions are measured from the material actually present rather than
    hard-coded, so each connector stays centred in its wall.
    """
    feats = []

    for target in PEG_YS:
        # The wall thins and drifts differently around the ring, so take the
        # height near the nominal one that leaves the most material.
        best = None
        for dy in np.arange(-PEG_Y_SEARCH, PEG_Y_SEARCH + 0.5, 1.0):
            y = target + dy
            if y - HOLE_D / 2 < SKIRT_TRIM_Y + 1.0 or y + HOLE_D / 2 > -1.0:
                continue
            rng = centre_range(male, female, y,
                               PEG_D / 2 + MIN_WALL, HOLE_D / 2 + MIN_WALL)
            if rng is None:
                continue
            lo, hi = rng
            wall = (hi - lo) / 2.0 + MIN_WALL
            if best is None or wall > best[0]:
                best = (wall, (lo + hi) / 2.0, y)
        if best is None:
            raise SystemExit(f"cut {angle:.0f}: no height within {PEG_Y_SEARCH:.0f} mm "
                             f"of y={target:.0f} where the skirt can hold a peg with "
                             f"{MIN_WALL} mm wall")
        feats.append(("peg", best[1], best[2], best[0]))

    rng = centre_range(male, female, TONGUE_Y,
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


def verify_joint(male, female, angle, feats):
    """Assert every connector is fully buried over its whole depth."""
    for kind, r, y, _ in feats:
        rnd = kind == "peg"
        peg = footprint(r, y, *((PEG_D / 2, PEG_D / 2) if rnd
                                else (TONGUE_L / 2, TONGUE_H / 2)), rnd)
        sock = footprint(r, y, *((HOLE_D / 2, HOLE_D / 2) if rnd
                                 else (SLOT_L / 2, SLOT_H / 2)), rnd)
        for region, shape, side in ((male, peg, "male root"),
                                    (female, sock, "female socket")):
            for i, cross in enumerate(region):
                if not cross.contains(shape):
                    raise SystemExit(
                        f"cut {angle:.0f} {kind} r={r:.1f} y={y:.0f} ({side}): "
                        f"breaks through at probe {i}")


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

        male = [profile(low, angle, d) for d in MALE_PROBES]
        female = [profile(high, angle, d) for d in FEMALE_PROBES]
        if any(x is None for x in male + female):
            raise SystemExit(f"cut {angle:.0f}: empty cross-section")
        feats = joint_features(male, female, angle)
        verify_joint(male, female, angle, feats)
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
