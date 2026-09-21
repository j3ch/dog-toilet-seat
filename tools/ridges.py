"""Add the v5 ridge pattern to the v4 seat.

The ridged model exists only as Fusion .f3d archives, whose geometry is a
proprietary ShapeManager BREP nothing outside Fusion can read.  What is readable
is toiletv5_rings.svg: 64 closed curves that are uniform outward offsets of the
hole outline, spaced 7.086 px apart, the innermost sitting two steps out from the
hole.  Registering the v4 sketch hole (toiletcuts_v2.svg, 617.74 x 690.63 px)
against the v4 model's own hole (219.1 x 242.9 mm) gives 0.3532 mm/px, which puts
that spacing at 2.503 mm -- the pattern was laid out on a 2.5 mm offset step.

So rather than mapping the v5 curves onto a model whose hole is a different
shape, this rebuilds the same construction on v4's own hole: concentric outward
offsets at 2.5 mm, raised on alternating bands.

The ridges are cut from a lifted copy of the part rather than stamped on at a
fixed height, so they follow the sloping top face exactly.  The stock starts at
FLOOR, which is below the top surface everywhere (it dips to ~9.2 mm near the
hole) but above the 8.5 mm flange slots, so the joints are never touched.

    python tools/ridges.py
"""

import math
from pathlib import Path

import numpy as np
import shapely
import trimesh

import split8 as S

ROOT = Path(__file__).resolve().parent.parent
OUT_WHOLE = str(ROOT / "toiletv4_ridged.stl")
OUT_SPLIT = str(ROOT / "toiletv4_ridged_in_8.stl")
OUT_SPLIT4 = str(ROOT / "toiletv4_ridged_in_4.stl")

# --- ridge pattern ---------------------------------------------------------
PITCH = 2.5         # ring spacing read off toiletv5_rings.svg
RIDGE_W = PITCH     # width of a raised band
PERIOD = 2 * PITCH  # raised, then flat
START = 2 * PITCH   # first ridge sits where the SVG's innermost ring does
N_RINGS = 64        # as drawn in the SVG - a floor, not the count used
RIDGE_H = 1.5       # how far the ridges stand above the top surface
HOLE_Y = 9.0        # height to read the hole outline at, below the top chamfer
SIMPLIFY = 0.15     # outline simplification, mm

RIDGE_SINK = 1.0    # how far the ridge stock reaches into the part below the top

# Laid on the bed the concentric bands are ~25 unconnected ribbons, 2.5 mm wide
# and up to 250 mm long, each free to curl at its ends - which is what lifted
# the first print.  A radial bar across them ties them into one network.  The
# bars sit on the cut planes, where they fall on a seam rather than in the
# middle of a face: the four 4-piece borders, plus the four diagonals, because
# the outermost bands exist only in the corners and never reach a 4-piece
# border - with only those four, 12373 mm2 stayed loose in 56 pieces.
BRIDGE_W = 10.0
BRIDGE_ANGLES = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)
# A plain band right round the outer edge, no ridges in it, closing the pattern
# into one enclosed loop and catching the outer end of every arc.
BORDER_W = 10.0
EDGE_GAP = 0.5      # keep the bands off the outer wall (see band_prisms)


def outlines(mesh, y=HOLE_Y):
    """The hole and the body footprint at height `y`, plus the plane transform."""
    sec = mesh.section(plane_normal=[0, 1, 0], plane_origin=[0, y, 0])
    planar, to_3d = sec.to_planar(normal=[0, 1, 0])
    body = max(planar.polygons_full, key=lambda p: p.area)
    ring = max(body.interiors, key=lambda r: shapely.Polygon(r).area)
    return shapely.Polygon(ring).simplify(SIMPLIFY), body, to_3d


def band_count(hole, clip):
    """How many bands it takes to reach the far corner of the flange.

    The SVG draws 64 rings, which is 32 bands and reaches 162.5 mm from the
    hole.  v4's back corners are up to 185 mm out, so a fixed 32 leaves a flat
    band 20-23 mm wide there; the count is taken from the geometry instead.
    """
    reach = max(hole.exterior.distance(shapely.Point(p))
                for p in clip.exterior.coords)
    return max(N_RINGS // 2,
               int(math.ceil((reach - START) / PERIOD)) + 1)


def flat_top_level(mesh, region, tol=0.01):
    """The height of the part's flat top, asserting that it really is flat.

    Sampled over `region`, the band footprint: the flattening slab is held just
    inside the outline, so a thin strip of the original rounded rim survives at
    the very edge, and it is the ground under the ridges that has to be flat.
    """
    top = mesh.bounds[1][1]
    inner = region
    x0, z0, x1, z1 = inner.bounds
    gx, gz = np.meshgrid(np.arange(x0, x1, 3.0), np.arange(z0, z1, 3.0))
    pts = [p for p in np.column_stack([gx.ravel(), gz.ravel()])
           if inner.contains(shapely.Point(p))]
    pts = np.array(pts)
    hit, _, _ = mesh.ray.intersects_location(
        np.column_stack([pts[:, 0], np.full(len(pts), top + 40.0), pts[:, 1]]),
        np.tile([0.0, -1.0, 0.0], (len(pts), 1)), multiple_hits=False)
    y = hit[hit[:, 1] > 5.0][:, 1]
    off = top - y
    share = float((off <= tol).mean())
    # 98%, not 100%: a hairline groove (<=0.2 mm) runs down each seam where two
    # quadrants' flattening slabs meet, and the sampling grid clips it.  The
    # bridges cover it everywhere except the innermost 5 mm.
    if share < 0.98:
        raise SystemExit(f"top is not flat: only {share * 100:.1f}% of {len(y)} samples "
                         f"sit at {top:.3f} mm; split8.FLAT_TOP must be on")
    print(f"  flat over {share * 100:.2f}% of the band footprint; "
          f"worst dip {off.max():.2f} mm, at the hole edge")
    return top


def band_shapes(hole, body):
    """The raised pattern as 2D polygons: concentric bands plus the bridges.

    Bands are clipped to the body footprint pulled in by EDGE_GAP.  Left flush,
    a band's side face lands exactly on the part's outer wall, and that
    coincident pair comes apart in the ridge union.

    Bands and bridges come back as two lists, each internally non-overlapping.
    Merging them in 2D first gives one polygon with 1200+ points and 56 holes
    that the triangulator cannot close, so they are extruded apart and merged
    as solids instead.
    """
    clip = body.buffer(-EDGE_GAP)
    inner = hole.buffer(START)
    shapes, bars = [], []
    for k in range(band_count(hole, clip)):
        off = START + k * PERIOD
        band = (hole.buffer(off + RIDGE_W, quad_segs=16)
                .difference(hole.buffer(off, quad_segs=16))
                .intersection(clip))
        if not band.is_empty:
            shapes.append(band)
    for angle in BRIDGE_ANGLES:
        t = math.radians(angle)
        d = np.array([math.cos(t), math.sin(t)])
        n = np.array([-math.sin(t), math.cos(t)]) * (BRIDGE_W / 2.0)
        far = d * 400.0
        bar = shapely.Polygon([tuple(-n), tuple(far - n), tuple(far + n), tuple(n)])
        bar = bar.intersection(clip).difference(inner)
        if not bar.is_empty:
            bars.append(bar)
    border = clip.difference(clip.buffer(-BORDER_W))
    if not border.is_empty:
        bars.append(border)
    # The spokes run into the border, so merge them before they are extruded;
    # the result is one ring with fingers, which triangulates cleanly.
    merged = shapely.union_all(bars)
    return shapes, list(getattr(merged, "geoms", [merged]))


def band_prisms(bands, bars, to_3d, top):
    """The raised pattern as a solid: bands and bridges, merged."""

    def solids(shapes):
        out = []
        for shape in shapes:
            for part in getattr(shape, "geoms", [shape]):
                p = trimesh.creation.extrude_polygon(part.simplify(S.FLAT_SIMPLIFY),
                                                     height=RIDGE_SINK + RIDGE_H)
                p.apply_transform(to_3d)
                p.apply_translation([0.0, top - RIDGE_SINK - HOLE_Y, 0.0])
                if not p.is_watertight:
                    raise SystemExit("a ridge prism is not a closed volume")
                out.append(p)
        return trimesh.util.concatenate(out)

    return trimesh.boolean.union([solids(bands), solids(bars)], engine=S.ENGINE)


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


def add_ridges(mesh, prisms, sector=None, span=45.0):
    """Stand the banded parts of the flat top up by RIDGE_H.

    On a flat top the bands are just prisms sitting on the plane, so they are
    unioned straight on.  They reach RIDGE_SINK into the part so the union has
    real overlap to weld through rather than a coplanar touch, and a piece's
    stock is clipped to its own sector so nothing lands over a peg protruding
    past a cut plane.
    """
    stock = prisms
    if sector is not None:
        stock = trimesh.boolean.intersection([prisms, sector_wedge(sector, span)],
                                             engine=S.ENGINE)
    return trimesh.boolean.union([mesh, stock], engine=S.ENGINE)


def load_pieces():
    """The 8 split pieces, un-exploded, ordered by sector."""
    mesh = trimesh.load(S.OUT)
    mesh.merge_vertices()
    parts = [p for p in mesh.split(only_watertight=False) if p.volume > 1.0]
    for p in parts:
        a = math.degrees(math.atan2(p.centroid[2], p.centroid[0])) % 360
        mid = math.radians(int(a // 45) * 45 + 22.5)
        p.apply_translation([-S.EXPLODE * math.cos(mid), 0.0,
                             -S.EXPLODE * math.sin(mid)])
    parts.sort(key=lambda p: math.degrees(math.atan2(p.centroid[2], p.centroid[0])) % 360)
    return parts


def report(name, mesh):
    """Print the shell breakdown and return the count of loose ridge fragments.

    Shells with negative volume are internal voids, not stray solids: an
    assembled seat keeps the 0.5 mm clearance gap around each connector of the
    joints it was cut on, and those read back as inward-facing shells.
    """
    parts = mesh.split(only_watertight=False)
    solids = [p for p in parts if p.volume > 0]
    voids = [p for p in parts if p.volume < 0]
    print(f"  {name}: {len(mesh.faces):6d} faces, {mesh.volume / 1000:7.1f} cm3, "
          f"{len(solids)} solid(s), {len(voids)} internal void(s)"
          + (f" totalling {-sum(p.volume for p in voids):.1f} mm3" if voids else ""))
    return len(solids) - 1


def main():
    solid = trimesh.boolean.union(S.load_quadrants(), engine=S.ENGINE)
    print(f"v4 assembled: watertight={solid.is_watertight}, "
          f"{solid.volume / 1000:.1f} cm3")

    hole, body, to_3d = outlines(solid)
    print(f"hole outline: {len(hole.exterior.coords)} pts after {SIMPLIFY} mm simplify")

    bands, bars = band_shapes(hole, body)
    top = flat_top_level(solid, shapely.union_all(bands + bars))
    print(f"top is flat at y = {top:.3f} mm")
    prisms = band_prisms(bands, bars, to_3d, top)
    loose = prisms.split(only_watertight=False)
    print(f"raised pattern: {len(bands)} bands {RIDGE_W:.1f} mm wide at "
          f"{PERIOD:.1f} mm period, {RIDGE_H:.1f} mm tall, tied by "
          f"{len(BRIDGE_ANGLES)} bridges {BRIDGE_W:.0f} mm wide and a "
          f"{BORDER_W:.0f} mm border -> {len(loose)} connected "
          f"group(s), largest {max(p.volume for p in loose) / prisms.volume * 100:.0f}%")

    # Four pieces, on v4's own cuts at X=0 and Z=0 and carrying its own
    # connectors: the quadrants are already exactly that, so they only need the
    # ridges adding, each clipped to its own 90 degree sector.
    quads = S.load_quadrants()
    assembled, out4 = [], []
    for qi, quad in enumerate(quads):
        quad = add_ridges(quad, prisms, qi, span=90.0)
        assembled.append(quad.copy())
        if report(f"Q{qi}", quad) or not quad.is_watertight:
            raise SystemExit(f"quadrant {qi} is unsound after adding ridges")
        mid = math.radians(qi * 90 + 45)
        quad.apply_translation([S.EXPLODE * math.cos(mid), 0.0,
                                S.EXPLODE * math.sin(mid)])
        out4.append(quad)
    trimesh.util.concatenate(out4).export(OUT_SPLIT4)
    print(f"wrote {OUT_SPLIT4}")

    # The whole seat, as the four quadrants in assembled position rather than
    # booleaned into one solid.  Fusing them leaves vertex pairs along the cut
    # planes too close together to survive STL's float32, which reloads as a
    # torn mesh; kept separate the file is clean, and the seat is four glued
    # pieces anyway.
    whole = trimesh.util.concatenate(assembled)
    print(f"  whole seat: {len(whole.faces)} faces, {whole.volume / 1000:.1f} cm3, "
          f"{whole.extents[1]:.1f} mm tall, ridges {(whole.volume - solid.volume) / 1000:.1f} cm3")
    whole.export(OUT_WHOLE)
    print(f"wrote {OUT_WHOLE}")

    out = []
    for si, piece in enumerate(load_pieces()):
        piece = add_ridges(piece, prisms, si)
        if report(f"S{si}", piece):
            raise SystemExit(f"sector {si} has loose ridge fragments")
        if not piece.is_watertight:
            raise SystemExit(f"sector {si} is not watertight after adding ridges")
        mid = math.radians(si * 45 + 22.5)
        piece.apply_translation([S.EXPLODE * math.cos(mid), 0.0,
                                 S.EXPLODE * math.sin(mid)])
        out.append(piece)
    trimesh.util.concatenate(out).export(OUT_SPLIT)
    print(f"wrote {OUT_SPLIT}")


if __name__ == "__main__":
    main()
