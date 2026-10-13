"""The v4 seat, remodelled: the geometry every output is built from.

toiletv4_sliced.stl holds the seat already cut into quadrants by the planes
X=0 and Z=0, exported exploded 30 mm outward, with v4's own joints (pegs down
the skirt, tongues in the flange, 0.5 mm clearance).  load_quadrants()
reassembles it, trims the skirt, turns the teardrop hole round and moves its
supports to the front, flattens the top and breaks the hole edge.  ridges.py
adds the ridges and writes the files.

The seat used to be re-split into 8 smaller pieces here; it no longer is - it
is printed in v4's 4 quadrants.
"""

import io
import math
from pathlib import Path

import manifold3d
import numpy as np
import shapely
import shapely.affinity
import trimesh

ROOT = Path(__file__).resolve().parent.parent
SRC = str(ROOT / "toiletv4_sliced.stl")
ENGINE = "manifold"

# --- joint geometry, measured off the existing v4 cut faces -----------------
PEG_D, PEG_L = 5.0, 6.0          # male round peg
HOLE_D, HOLE_DEPTH = 6.0, 7.0    # female socket
TONGUE_L, TONGUE_H = 35.0, 4.0   # male flange tongue
SLOT_L, SLOT_H = 36.0, 5.0       # female flange slot
TONGUE_Y = 6.0                   # tongue mid-height (spans y = 4..8)
SECTIONS = 24                    # facets per peg, matching v4

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
#
# The inset is taken on the assembled seat, so it applies to the outer wall and
# the hole but not to the cut planes.  Applied per quadrant it also pulled each
# slab back from its own seams, leaving a 0.5 mm groove down all four of them.
FLAT_INSET = 0.25
# Flattening squares off the rounded lip v4 had where the top meets the hole,
# and that edge is right where Lala's paws sit.  This breaks it back with a
# 2 mm chamfer, cut as a staircase of 0.2 mm steps - finer than a layer, so the
# printed edge is rounded by the layer lines anyway.  A true quarter-round was
# tried first and is not usable: its profile ends in 0.01-0.09 mm offsets from
# the hole wall, slivers that do not survive STL's float32 and left one
# quadrant non-watertight on reload.
HOLE_ROUND_R = 2.0
HOLE_ROUND_STEPS = 10

# The back is -Z: the skirt is squarer at the back and tapers to the front, as
# the bowl in measurements.jpg does.  v4's teardrop hole had its wide end at the
# front, flush with the skirt, and its narrow end at the back, where three
# gusset ribs (230, 270, 314 degrees, ~50 mm reach) carried the flange.  It is
# turned round: mirrored in Z, slid back until its wide end sits on the back
# wall, and its front half stretched (scaled in Z about the widest line, so the
# sides stay tangent there) until the tip is RIB_REACH + RIB_LAND inside the
# front wall.  The skirt is not touched.
REVERSE_HOLE = True
RIB_REACH = 30.0                 # how far a support reaches in from the skirt wall
RIB_LAND = 3.0                   # flange left past a rib tip, as v4 (3-4 mm)
RIB_DEPTH = 27.0                 # how far down the skirt it runs; v4's 45 mm, scaled
RIB_THICK = 9.0                  # as v4 (25.5 px in toiletcuts_v2.svg)
RIB_ANGLES = (46.0, 90.0, 134.0)     # v4's 314, 270 and 230, mirrored to the front
RIB_SINK = 1.0                   # into the wall and the flange, for the union to weld
LOFT_STEP = 0.5                  # the cutter that clears the old ribs follows the wall,
LOFT_ANGLE = 0.5                 # sampled this often up (mm) and round (degrees),
LOFT_SHY = 0.02                  # and held this far off it
SLIT = 0.05                      # seam slits narrower than twice this are closed
FILL_TOP = 9.5                   # the old hole is refilled up into the flattening slab
FILL_OVERLAP = 2.0               # and reaches past its edge so the union welds
# v4's own joint at 0 degrees has a flange tongue starting ~106 mm out, which
# the reversed hole runs 1.5 mm past the end of.  Rather than dent the hole,
# the tongue is shortened and the end of its slot filled, so the slot keeps
# JOINT_WALL of material between it and the hole.  The tongue at 270 lies
# almost wholly inside the new hole, so it is removed and its slot filled.
JOINT_WALL = 2.0
SLOT_FILL = 1.5


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
    parts.sort(key=lambda p: math.degrees(math.atan2(p.centroid[2], p.centroid[0])) % 360)
    if REVERSE_HOLE:
        parts = reverse_hole(parts)
    if FLAT_TOP:
        level = max(p.bounds[1][1] for p in parts)
        foot = flat_footprint(parts)
        parts = [flatten_top(p, level, foot.intersection(sector_polygon(i)))
                 for i, p in enumerate(parts)]
        for p in parts:
            if not p.is_watertight:
                raise SystemExit("quadrant is not watertight after flattening")
        if HOLE_ROUND_R:
            # Built once from the shared footprint so the four quadrants get
            # the same edge, then subtracted from each.
            rings = [shapely.Polygon(r) for g in getattr(foot, "geoms", [foot])
                     for r in g.interiors]
            if not rings:
                raise SystemExit("no hole in the flattening footprint to round")
            cutter = hole_round_cutter(max(rings, key=lambda r: r.area), level)
            parts = [trimesh.boolean.difference([p, cutter], engine=ENGINE)
                     for p in parts]
            for p in parts:
                if not p.is_watertight:
                    raise SystemExit("quadrant is not watertight after rounding")
    return parts


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


def flat_footprint(parts):
    """Where the flattening slab goes, as a polygon in world X/Z.

    The union of cross-sections taken below the flange slots and just above
    them: below, the outline is clean but slightly drafted in; above, it is
    full width.  Taken at slot height it would come back with slot-shaped holes
    and notch the new surface.

    Read piece by piece (assembled_outline): the quadrants booleaned into one
    solid tear along a seam once the hole is reversed, and the section loses
    the hole, so the slab would fill it.
    """
    foot = shapely.union_all([o for o in
                              (assembled_outline(parts, y) for y in FLAT_FOOTPRINT_YS)
                              if o is not None])
    return foot.buffer(-FLAT_INSET)


def sector_polygon(index, span=90.0, reach=1000.0):
    a, b = math.radians(index * span), math.radians((index + 1) * span)
    return shapely.Polygon([(0.0, 0.0),
                            (reach * math.cos(a), reach * math.sin(a)),
                            (reach * math.cos(b), reach * math.sin(b))])


def flatten_top(mesh, level, foot):
    """Fill the dished top up to a flat plane at `level`."""
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


def hole_round_cutter(hole, level, radius=HOLE_ROUND_R, steps=HOLE_ROUND_STEPS):
    """A 45 degree break along the top edge of the hole, as a cutting solid.

    Stepped rather than swept: at 2 mm over 10 steps each facet is 0.2 mm,
    finer than a layer.  The staircase circumscribes the chamfer, so it never
    leaves material proud of it.

    `hole` comes from the flattening footprint rather than from a section of
    the assembled seat: a hairline groove at one seam opens the annulus there,
    so the hole does not come back as a ring and the largest one found is a
    sliver.
    """
    solids = []
    for i in range(steps):
        d_lo = radius * i / steps            # depth of the shallow edge of this step
        d_hi = radius * (i + 1) / steps
        off = radius - d_lo
        region = hole.buffer(off) if off > 0 else hole
        p = trimesh.creation.extrude_polygon(region.simplify(FLAT_SIMPLIFY),
                                             height=d_hi + 1.0)
        p.apply_transform(trimesh.transformations.rotation_matrix(
            math.pi / 2, [1, 0, 0]))
        p.apply_translation([0.0, level + 1.0, 0.0])
        solids.append(p)
    return trimesh.boolean.union(solids, engine=ENGINE)


def sector_wedge(sector, span=45.0, reach=600.0, half_height=200.0):
    """Solid covering one sector, on the planes the pieces are cut on."""
    a = math.radians(sector * span)
    b = math.radians((sector + 1) * span)
    tri = shapely.Polygon([(0.0, 0.0),
                           (reach * math.cos(a), reach * math.sin(a)),
                           (reach * math.cos(b), reach * math.sin(b))])
    prism = trimesh.creation.extrude_polygon(tri, height=2 * half_height)
    # +90 about X sends the polygon's (x, y) to world (x, z) and extrudes along
    # -Y; the -90 rotation would mirror the sector in Z.
    prism.apply_transform(trimesh.transformations.rotation_matrix(
        math.pi / 2, [1, 0, 0]))
    prism.apply_translation([0.0, half_height, 0.0])
    mid = math.radians(sector * span + span / 2)
    probe = [reach / 2 * math.cos(mid), 0.0, reach / 2 * math.sin(mid)]
    if not prism.contains([probe])[0]:
        raise SystemExit(f"sector {sector} wedge does not cover its own bisector")
    return prism


def prism(region, y0, y1):
    """`region`, a polygon in world X/Z, as a solid between heights y0 and y1."""
    solids = []
    for part in getattr(region, "geoms", [region]):
        if part.is_empty or part.area < 0.01:
            continue
        p = trimesh.creation.extrude_polygon(part.simplify(FLAT_SIMPLIFY),
                                             height=y1 - y0)
        p.apply_transform(trimesh.transformations.rotation_matrix(
            math.pi / 2, [1, 0, 0]))
        p.apply_translation([0.0, y1, 0.0])
        if not p.is_watertight:
            raise SystemExit("a prism is not a closed volume")
        solids.append(p)
    if len(solids) == 1:
        return solids[0]
    return trimesh.boolean.union(solids, engine=ENGINE)


def assembled_outline(parts, y):
    """The assembled seat's section at `y`: each piece's, unioned in 2D.

    Where two pieces' faces meet on a seam without sharing vertices the union
    leaves a zero-width ring along it; those are dropped.
    """
    merged = shapely.union_all([o for o in (xz_outline(p, y) for p in parts)
                                if o is not None])
    return shapely.union_all([
        shapely.Polygon(g.exterior, [r for r in g.interiors
                                     if shapely.Polygon(r).area > 1.0])
        for g in getattr(merged, "geoms", [merged])])


def opening(parts, y):
    """The largest hole through the assembled seat at height `y`.

    A plane through a row of vertices (v4 has some at round heights, y=-10
    among them) sections into open loops and loses part of the outline, so the
    height is nudged a hair until it comes back closed.
    """
    for dy in (0.0, 0.05, -0.05, 0.1, -0.1):
        sec = assembled_outline(parts, y + dy)
        body = max(getattr(sec, "geoms", [sec]), key=lambda g: g.area)
        if body.interiors:
            return shapely.Polygon(max(body.interiors,
                                       key=lambda r: shapely.Polygon(r).area))
    raise SystemExit(f"no closed opening through the seat at y={y}")


def skirt_inner(parts, y):
    """The skirt's inner opening at `y`, without the ribs.

    The opening is convex and the ribs are notches in it, so the hull is the
    bare wall.  At some heights the two pieces' sections do not quite meet on
    a seam and leave a zero-width slit running from the opening into the
    wall; the hull would reach out to the slit's tip and the rib cutter bite
    up to 7 mm into the wall there, so slits are closed up first.
    """
    return opening(parts, y).buffer(-SLIT).buffer(SLIT).convex_hull


def ray_hit(region, angle, lateral=0.0):
    """How far out along the ray at `angle` the edge of `region` is.

    `lateral` shifts the ray sideways; the distance is still measured along it.
    """
    a = math.radians(angle)
    u = np.array([math.cos(a), math.sin(a)])
    start = np.array([-math.sin(a), math.cos(a)]) * lateral
    hit = shapely.LineString([tuple(start), tuple(start + u * 1000.0)]).intersection(
        region.exterior)
    pts = [np.asarray(c) for g in getattr(hit, "geoms", [hit]) for c in g.coords]
    if not pts:
        raise SystemExit(f"nothing on the ray at {angle:.0f} degrees")
    return min(float(p @ u) for p in pts)


def flange_tongues(parts):
    """v4's flange tongues, as (piece, footprint at tongue height).

    A tongue is whatever a piece has outside its own sector.
    """
    out = []
    for i, p in enumerate(parts):
        stray = xz_outline(p, TONGUE_Y).difference(sector_polygon(i).buffer(0.01))
        out += [(i, g) for g in getattr(stray, "geoms", [stray]) if g.area > 5.0]
    return out


def reversed_hole(parts):
    """v4's hole turned round, as a polygon in world X/Z."""
    old = opening(parts, 9.0)
    wall = skirt_inner(parts, -1.0)
    room = wall.buffer(-FLAT_INSET)
    hole = shapely.affinity.scale(old, 1.0, -1.0, origin=(0.0, 0.0))
    # Slid back as far as it will go, so the wide end comes to rest on the wall.
    for dz in np.arange(-100.0, 100.0, 0.1):
        if room.contains(shapely.affinity.translate(hole, 0.0, dz)):
            break
    else:
        raise SystemExit("the mirrored hole does not fit inside the skirt")
    hole = shapely.affinity.translate(hole, 0.0, dz)
    x0, z0, x1, z1 = hole.bounds
    z_w = max(np.arange(z0 + 1.0, z1 - 1.0, 0.25),
              key=lambda z: shapely.LineString([(x0 - 1, z), (x1 + 1, z)])
              .intersection(hole).length)
    # The sides are parallel on the widest line, so scaling the front half in Z
    # about it keeps them tangent there.
    tip = ray_hit(wall, 90.0) - RIB_REACH - RIB_LAND
    back = hole.intersection(shapely.box(x0 - 1, z0 - 1, x1 + 1, z_w))
    front = hole.intersection(shapely.box(x0 - 1, z_w, x1 + 1, z1 + 1))
    front = shapely.affinity.scale(front, 1.0, (tip - z_w) / (z1 - z_w),
                                   origin=(0.0, z_w))
    hole = shapely.union_all([back, front]).buffer(0)
    if hole.geom_type != "Polygon":
        raise SystemExit("the reversed hole came apart")
    if hole.difference(room).area > 1.0:
        raise SystemExit("the reversed hole runs into the skirt")
    return old, hole.intersection(room)


def polar(region, angles):
    """Distance from the origin to the edge of the convex `region` along each angle."""
    c = np.asarray(region.exterior.coords)[:-1]
    th = np.arctan2(c[:, 1], c[:, 0])
    order = np.argsort(th)
    c, th = c[order], th[order]
    d = np.column_stack([np.cos(angles), np.sin(angles)])
    i = np.searchsorted(th, np.arctan2(d[:, 1], d[:, 0]))
    p0, p1 = c[i - 1], c[i % len(c)]
    e = p1 - p0
    cross = lambda u, w: u[:, 0] * w[:, 1] - u[:, 1] * w[:, 0]
    return cross(p0, e) / cross(d, e)


def old_rib_cutter(parts):
    """Everything inside the skirt below the flange, as a solid lofted on the wall.

    Below the flange the only thing inside the wall is v4's ribs.  The bare
    wall is the hull of the opening at each height (the ribs are notches in
    it); it is sampled every LOFT_STEP mm up and LOFT_ANGLE degrees round,
    held LOFT_SHY off the wall so the cut never touches it, and the rings
    joined into one ruled solid that runs up to the flange's underside.  A
    stack of prisms left a staircase of rib up to 0.5 mm proud, and a 0.3 mm
    plate of each rib top under the flange.
    """
    angles = np.radians(np.arange(0.0, 360.0, LOFT_ANGLE))
    ys = list(np.arange(SKIRT_TRIM_Y + 0.5, -0.1, LOFT_STEP)) + [-LOFT_SHY]
    rings = []
    for y in ys:
        # A section exactly on the underside is degenerate; 0.1 mm below it
        # the wall is 0.01 mm further in.
        wall = skirt_inner(parts, min(y, -0.1))
        r = polar(wall, angles) - LOFT_SHY
        ring = shapely.Polygon(np.column_stack([r * np.cos(angles), r * np.sin(angles)]))
        if not wall.buffer(-LOFT_SHY / 2).contains(ring):
            raise SystemExit(f"the loft ring at y={y:.1f} reaches the wall")
        rings.append(np.column_stack([r * np.cos(angles), np.full(len(r), y),
                                      r * np.sin(angles)]))
    k, m = len(rings), len(angles)
    verts = np.vstack(rings + [[[0.0, ys[0], 0.0]], [[0.0, ys[-1], 0.0]]])
    bottom, top = k * m, k * m + 1
    faces = []
    for i in range(k - 1):
        for j in range(m):
            a0, a1 = i * m + j, i * m + (j + 1) % m
            b0, b1 = a0 + m, a1 + m
            faces += [(a0, a1, b1), (a0, b1, b0)]
    for j in range(m):
        faces.append((bottom, (j + 1) % m, j))
        faces.append((top, (k - 1) * m + j, (k - 1) * m + (j + 1) % m))
    loft = trimesh.Trimesh(verts, np.array(faces), process=False)
    trimesh.repair.fix_normals(loft)
    if not loft.is_volume:
        raise SystemExit("the cutter lofted on the skirt wall is not a closed volume")
    return loft


def front_ribs(parts, hole):
    """The supports under the front flange, one gusset per RIB_ANGLES.

    In side view each is a concave quarter-ellipse, like v4's: RIB_REACH in
    from the wall at the flange, running RIB_DEPTH down the skirt.  Its outer
    edge follows the wall down, sunk RIB_SINK into it.  A rib whose reach would
    bring it within RIB_LAND of the hole is shortened.
    """
    ys = np.linspace(0.0, -RIB_DEPTH, int(RIB_DEPTH) + 1)
    walls = [skirt_inner(parts, min(y, -0.5)) for y in ys]
    ribs = []
    for angle in RIB_ANGLES:
        a = math.radians(angle)
        u = np.array([math.cos(a), 0.0, math.sin(a)])
        v = np.array([0.0, 1.0, 0.0])
        n = np.array([-math.sin(a), 0.0, math.cos(a)])
        centre = [ray_hit(w, angle) for w in walls]
        outer = [max(ray_hit(w, angle, s * RIB_THICK / 2) for s in (-1, 0, 1)) + RIB_SINK
                 for w in walls]
        def top(reach):
            """The rib's footprint where it meets the flange."""
            return shapely.Polygon([tuple(p[[0, 2]]) for p in (
                u * (centre[0] - reach) + n * RIB_THICK / 2,
                u * outer[0] + n * RIB_THICK / 2,
                u * outer[0] - n * RIB_THICK / 2,
                u * (centre[0] - reach) - n * RIB_THICK / 2)])

        # Measured on the footprint rather than the centre line: where the hole
        # edge runs across the rib at a slant, a corner gets there first.
        reach = RIB_REACH
        while top(reach).distance(hole) < RIB_LAND:
            reach -= 0.25
        if reach < RIB_REACH / 2:
            raise SystemExit(f"rib at {angle:.0f}: only {reach:.1f} mm between wall and hole")
        fillet = [1.0 - math.sqrt(max(0.0, 1.0 - (1.0 + y / RIB_DEPTH) ** 2)) for y in ys]
        profile = shapely.Polygon(
            [(outer[0], RIB_SINK)] + list(zip(outer, ys))
            + [(c - reach * f, y) for c, f, y in reversed(list(zip(centre, fillet, ys)))]
            + [(centre[0] - reach, RIB_SINK)])
        if not profile.is_valid:
            raise SystemExit(f"rib at {angle:.0f}: profile is not a simple polygon")
        rib = trimesh.creation.extrude_polygon(profile, height=RIB_THICK)
        rib.apply_transform(placement_transform(u, v, n, -n * RIB_THICK / 2))
        print(f"  rib at {angle:5.1f} deg: reach {reach:.1f} mm, "
              f"{top(reach).distance(hole):.1f} mm short of the hole")
        ribs.append(rib)
    return trimesh.util.concatenate(ribs)


def reverse_hole(parts):
    """Turn the teardrop round and move its supports to the front.

    In order: v4's ribs come out, the old hole is filled back up to the
    flange, the new hole is cut, and the new ribs go in.  The skirt is not
    touched.
    """
    old, hole = reversed_hole(parts)
    tongues = flange_tongues(parts)
    dead = [(i, g) for i, g in tongues if g.intersection(hole).area > g.area / 2]
    # A surviving tongue whose slot comes within JOINT_WALL of the hole loses
    # its inner end, and the slot is filled to match, 0.5 mm short of it.
    near = [(i, g) for i, g in tongues if (i, g) not in dead
            and g.distance(hole) < JOINT_WALL + 0.5]
    for i, g in near:
        cut = g.intersection(hole.buffer(JOINT_WALL + 0.5))
        width = min(g.bounds[2] - g.bounds[0], g.bounds[3] - g.bounds[1])
        print(f"  tongue at {math.degrees(math.atan2(g.centroid.y, g.centroid.x)) % 360:.0f} deg "
              f"shortened by {cut.area / width:.1f} mm to keep its slot off the hole")
    print(f"hole reversed: was z {old.bounds[1]:.1f}..{old.bounds[3]:.1f}, "
          f"{old.area / 100:.0f} cm2; now z {hole.bounds[1]:.1f}..{hole.bounds[3]:.1f}, "
          f"{hole.area / 100:.0f} cm2; {len(dead)} of v4's tongues inside it removed")

    level = max(p.bounds[1][1] for p in parts)
    cutter = old_rib_cutter(parts)
    through = prism(hole, -1.0, level + 5.0)
    ribs = front_ribs(parts, hole)
    out = []
    for i, p in enumerate(parts):
        sector = sector_polygon(i)
        p = trimesh.boolean.difference([p, cutter], engine=ENGINE)
        stubs = ([g.buffer(0.3).difference(sector) for j, g in dead if j == i]
                 + [g.buffer(0.3).intersection(hole.buffer(JOINT_WALL + 0.5))
                    .difference(sector) for j, g in near if j == i])
        if stubs:
            p = trimesh.boolean.difference(
                [p, prism(shapely.union_all(stubs), TONGUE_Y - TONGUE_H,
                          TONGUE_Y + TONGUE_H)], engine=ENGINE)
        # The old hole, the slot of any tongue that was removed, and the inner
        # end of any slot whose tongue was shortened.
        # A slot is 1 mm deeper than its tongue is long, so the fill reaches
        # SLOT_FILL past the tongue to meet the slot's floor; short of it, the
        # slot wraps round the fill and leaves a tunnel through the piece.
        fill = shapely.union_all([old.buffer(FILL_OVERLAP)]
                                 + [g.buffer(SLOT_FILL) for _, g in dead]
                                 + [g.buffer(SLOT_FILL).intersection(hole.buffer(JOINT_WALL))
                                    for _, g in near]).intersection(sector)
        p = trimesh.boolean.union([p, prism(fill, 0.0, FILL_TOP)], engine=ENGINE)
        p = trimesh.boolean.difference([p, through], engine=ENGINE)
        mine = trimesh.boolean.intersection([ribs, sector_wedge(i, 90.0)], engine=ENGINE)
        if len(mine.faces):
            p = trimesh.boolean.union([p, mine], engine=ENGINE)
        if not p.is_watertight:
            raise SystemExit(f"quadrant {i} is not watertight after reversing the hole")
        out.append(p)
    return out


def edges_ok(mesh):
    """Closed, one shell, every edge used by exactly two faces."""
    _, uses = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    return mesh.is_watertight and mesh.euler_number == 2 and bool((uses == 2).all())


def stl_sound(mesh, name, tol=1e-4):
    """`mesh`, made sure to read back from an STL as the same closed solid.

    STL stores float32.  Where a boolean has left a near-degenerate sliver -
    the reversed hole leaves one on the 270 degree seam face, running from the
    slab floor at the hole out to the rim - rounding can collapse it and merge
    vertices that were distinct, so the file reads back with an edge shared
    by four faces.  Such a piece goes back through manifold's simplify, which
    collapses edges shorter than `tol` (0.1 micron) and so removes the
    sliver; it is refused if the volume moves by more than 0.05 mm3.
    """
    def reload(m):
        back = trimesh.load(io.BytesIO(m.export(file_type="stl")), file_type="stl")
        back.merge_vertices()
        return back

    if edges_ok(reload(mesh)):
        return mesh
    man = manifold3d.Manifold(manifold3d.Mesh(
        vert_properties=np.asarray(mesh.vertices, dtype=np.float32),
        tri_verts=np.asarray(mesh.faces, dtype=np.uint32)))
    out = man.simplify(tol).to_mesh()
    fixed = trimesh.Trimesh(np.asarray(out.vert_properties)[:, :3], out.tri_verts)
    change = abs(fixed.volume - mesh.volume)
    if not edges_ok(reload(fixed)) or change > 0.05:
        raise SystemExit(f"{name} does not survive STL's float32 and could not be repaired")
    print(f"  ({name}: collapsed a sliver float32 would pinch, "
          f"volume change {change:.4f} mm3)")
    return fixed
