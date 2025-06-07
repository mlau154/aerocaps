import numpy as np

import aerocaps
import aerocaps.iges.curves
import aerocaps.iges.surfaces
import aerocaps.iges.iges_generator
from aerocaps.geom.geometry_container import GeometryContainer


def test_fill_surface_xy_plane():
    # Create a triangle in the X-Y plane
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    p0 = aerocaps.Point3D.from_array(points[0, :])
    p1 = aerocaps.Point3D.from_array(points[1, :])
    p2 = aerocaps.Point3D.from_array(points[2, :])
    line_1 = aerocaps.Line3D(p0=p0, p1=p1)
    line_2 = aerocaps.Line3D(p0=p1, p1=p2)
    line_3 = aerocaps.Line3D(p0=p2, p1=p0)
    composite = aerocaps.CompositeCurve3D([line_1, line_2, line_3])

    # Create a surface that fully encloses the triangle in the X-Y plane
    corners = np.array([
        [-2.0, -2.0, 0.0],
        [2.0, -2.0, 0.0],
        [2.0, 2.0, 0.0],
        [-2.0, 2.0, 0.0]
    ])
    pa = aerocaps.Point3D.from_array(corners[0, :])
    pb = aerocaps.Point3D.from_array(corners[1, :])
    pc = aerocaps.Point3D.from_array(corners[2, :])
    pd = aerocaps.Point3D.from_array(corners[3, :])
    surf = aerocaps.BezierSurface([[pa, pd], [pb, pc]])

    # Create the parametric space versions of the triangle lines
    parametric_points = np.array([
        [0.5, 0.5, 0.0],
        [0.75, 0.75, 0.0],
        [0.5, 0.75, 0.0]
    ])
    p0_para = aerocaps.Point3D.from_array(parametric_points[0, :])
    p1_para = aerocaps.Point3D.from_array(parametric_points[1, :])
    p2_para = aerocaps.Point3D.from_array(parametric_points[2, :])
    line_1_para = aerocaps.Line3D(p0=p0_para, p1=p1_para)
    line_2_para = aerocaps.Line3D(p0=p1_para, p1=p2_para)
    line_3_para = aerocaps.Line3D(p0=p2_para, p1=p0_para)
    composite_para = aerocaps.CompositeCurve3D([line_1_para, line_2_para, line_3_para])

    # Create the definition for the parametric curve
    curve_on_parametric_surface = aerocaps.CurveOnParametricSurface(
        surf,
        composite_para,
        composite
    )

    # Create the trimmed surface object
    trimmed_surf = aerocaps.TrimmedSurface(
        untrimmed_surface=surf,
        outer_boundary=composite,
        outer_boundary_para=composite_para,
        outer_curve_on_parametric_surf_para=curve_on_parametric_surface
    )

    # Set up the IGES generator and generate the IGES file
    container = GeometryContainer()
    container.add_geometry(trimmed_surf)
    container.export_iges("fill_surface_xy_plane.igs")
