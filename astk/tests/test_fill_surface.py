import numpy as np

import astk
import astk.iges.curves
import astk.iges.surfaces
import astk.iges.iges_generator


def test_fill_surface_xy_plane():
    # Create a triangle in the X-Y plane
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    p0 = astk.Point3D.from_array(points[0, :])
    p1 = astk.Point3D.from_array(points[1, :])
    p2 = astk.Point3D.from_array(points[2, :])
    line_1 = astk.Line3D(p0=p0, p1=p1)
    line_2 = astk.Line3D(p0=p1, p1=p2)
    line_3 = astk.Line3D(p0=p2, p1=p0)
    composite = astk.CompositeCurve3D([line_1, line_2, line_3])

    # Create a surface that fully encloses the triangle in the X-Y plane
    corners = np.array([
        [-2.0, -2.0, 0.0],
        [2.0, -2.0, 0.0],
        [2.0, 2.0, 0.0],
        [-2.0, 2.0, 0.0]
    ])
    pa = astk.Point3D.from_array(corners[0, :])
    pb = astk.Point3D.from_array(corners[1, :])
    pc = astk.Point3D.from_array(corners[2, :])
    pd = astk.Point3D.from_array(corners[3, :])
    surf = astk.BezierSurface([[pa, pd], [pb, pc]])

    # Create the parametric space versions of the triangle lines
    parametric_points = np.array([
        [0.5, 0.5, 0.0],
        [0.75, 0.75, 0.0],
        [0.5, 0.75, 0.0]
    ])
    p0_para = astk.Point3D.from_array(parametric_points[0, :])
    p1_para = astk.Point3D.from_array(parametric_points[1, :])
    p2_para = astk.Point3D.from_array(parametric_points[2, :])
    line_1_para = astk.Line3D(p0=p0_para, p1=p1_para)
    line_2_para = astk.Line3D(p0=p1_para, p1=p2_para)
    line_3_para = astk.Line3D(p0=p2_para, p1=p0_para)
    composite_para = astk.CompositeCurve3D([line_1_para, line_2_para, line_3_para])

    # Create the definition for the parametric curve
    curve_on_parametric_surface = astk.CurveOnParametricSurface(
        surf,
        composite_para,
        composite
    )

    # Create the trimmed surface object
    trimmed_surf = astk.TrimmedSurface(surf, curve_on_parametric_surface)

    # Set up the IGES generator and generate the IGES file
    entities = [line.to_iges() for line in [line_1, line_2, line_3]]
    entities.extend([line.to_iges() for line in [line_1_para, line_2_para, line_3_para]])
    entities.append(composite.to_iges(entities[0:3]))
    entities.append(composite_para.to_iges(entities[3:6]))
    entities.append(surf.to_iges())
    entities.append(curve_on_parametric_surface.to_iges(entities[8], entities[7], entities[6]))
    entities.append(trimmed_surf.to_iges(entities[8], entities[9]))
    iges_generator = astk.iges.iges_generator.IGESGenerator(entities, units="meters")
    iges_generator.generate("fill_surface_xy_plane.igs")


def test_fill_surface_xy_plane_new_builtin():
    # Create a triangle in the X-Y plane
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    p0 = astk.Point3D.from_array(points[0, :])
    p1 = astk.Point3D.from_array(points[1, :])
    p2 = astk.Point3D.from_array(points[2, :])
    line_1 = astk.Line3D(p0=p0, p1=p1)
    line_2 = astk.Line3D(p0=p1, p1=p2)
    line_3 = astk.Line3D(p0=p2, p1=p0)

    fill = astk.PlanarFillSurfaceCreator([line_1, line_2, line_3])
    iges_generator = astk.iges.iges_generator.IGESGenerator(fill.to_iges(), units="meters")
    iges_generator.generate("fill_surface_xy_plane_new_builtin.igs")
