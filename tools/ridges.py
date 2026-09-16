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

import shapely
import trimesh

import split8 as S

ROOT = Path(__file__).resolve().parent.parent
OUT_WHOLE = str(ROOT / "toiletv4_ridged.stl")
OUT_SPLIT = str(ROOT / "toiletv4_ridged_in_8.stl")

# --- ridge pattern ---------------------------------------------------------
PITCH = 2.5         # ring spacing read off toiletv5_rings.svg
RIDGE_W = PITCH     # width of a raised band
PERIOD = 2 * PITCH  # raised, then flat
START = 2 * PITCH   # first ridge sits where the SVG's innermost ring does
N_RINGS = 64        # as drawn in the SVG
RIDGE_H = 1.5       # how far the ridges stand above the top surface
HOLE_Y = 9.0        # height to read the hole outline at, below the top chamfer
SIMPLIFY = 0.15     # outline simplification, mm

FLOOR = 8.6         # bottom of the ridge stock: under the top face, over the slots
CEILING = 40.0
EDGE_GAP = 0.5      # keep the bands off the outer wall (see band_prisms)


def outlines(mesh, y=HOLE_Y):
    """The hole and the body footprint at height `y`, plus the plane transform."""
    sec = mesh.section(plane_normal=[0, 1, 0], plane_origin=[0, y, 0])
    planar, to_3d = sec.to_planar(normal=[0, 1, 0])
    body = max(planar.polygons_full, key=lambda p: p.area)
    ring = max(body.interiors, key=lambda r: shapely.Polygon(r).area)
    return shapely.Polygon(ring).simplify(SIMPLIFY), body, to_3d


def band_prisms(hole, body, to_3d):
    """One tall prism per raised band, in model space.

    Bands are clipped to the body footprint pulled in by EDGE_GAP.  Left flush,
    a band's side face lands exactly on the part's outer wall, and where that
    wall meets the top face the two surfaces touch at a point - a pinch vertex
    that keeps the mesh watertight but not a closed shell.
    """
    clip = body.buffer(-EDGE_GAP)
    prisms = []
    for k in range(N_RINGS // 2):
        off = START + k * PERIOD
        band = (hole.buffer(off + RIDGE_W, quad_segs=16)
                .difference(hole.buffer(off, quad_segs=16))
                .intersection(clip))
        if band.is_empty:
            continue
        for part in getattr(band, "geoms", [band]):
            p = trimesh.creation.extrude_polygon(part, height=CEILING - FLOOR)
            p.apply_transform(to_3d)
            p.apply_translation([0.0, FLOOR - HOLE_Y, 0.0])
            prisms.append(p)
    return trimesh.util.concatenate(prisms)


def sector_wedge(sector, reach=600.0, half_height=200.0):
    """Solid covering one 45 degree sector, on the planes the pieces are cut on."""
    a = math.radians(sector * 45.0)
    b = math.radians((sector + 1) * 45.0)
    tri = shapely.Polygon([(0.0, 0.0),
                           (reach * math.cos(a), reach * math.sin(a)),
                           (reach * math.cos(b), reach * math.sin(b))])
    prism = trimesh.creation.extrude_polygon(tri, height=2 * half_height)
    # +90 about X sends the polygon's (x, y) to world (x, z) and extrudes along
    # -Y; the -90 rotation would mirror the sector in Z.
    prism.apply_transform(trimesh.transformations.rotation_matrix(
        math.pi / 2, [1, 0, 0]))
    prism.apply_translation([0.0, half_height, 0.0])
    mid = math.radians(sector * 45 + 22.5)
    probe = [reach / 2 * math.cos(mid), 0.0, reach / 2 * math.sin(mid)]
    if not prism.contains([probe])[0]:
        raise SystemExit(f"sector {sector} wedge does not cover its own bisector")
    return prism


def add_ridges(mesh, prisms, sector=None):
    """Raise the banded parts of this part's top face by RIDGE_H.

    The stock is the part itself lifted and clipped to the bands, so it overlaps
    the body and welds to it; clipping a piece to its own sector also trims the
    sliver that would otherwise sit over a peg protruding past the cut plane.
    """
    lifted = mesh.copy()
    lifted.apply_translation([0.0, RIDGE_H, 0.0])
    stock = trimesh.boolean.intersection([lifted, prisms], engine=S.ENGINE)
    if sector is not None:
        stock = trimesh.boolean.intersection([stock, sector_wedge(sector)],
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

    prisms = band_prisms(hole, body, to_3d)
    print(f"{N_RINGS // 2} raised bands at {PERIOD:.1f} mm period, "
          f"{RIDGE_W:.1f} mm wide, {RIDGE_H:.1f} mm tall, "
          f"first at {START:.1f} mm from the hole")

    whole = add_ridges(solid, prisms)
    loose = report("whole seat", whole)
    print(f"  ridge material: {(whole.volume - solid.volume) / 1000:.1f} cm3, "
          f"height {solid.extents[1]:.1f} -> {whole.extents[1]:.1f} mm")
    if loose:
        raise SystemExit("ridges did not weld to the body")
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
