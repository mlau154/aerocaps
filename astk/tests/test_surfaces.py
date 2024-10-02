import os

import numpy as np

print(os.getcwd())


#from astk import DATA_DIR
from astk.geom.point import Point3D
from astk.geom.surfaces import NURBSSurface, BezierSurface, SurfaceEdge
from astk.geom.curves import Bezier3D,Line3D
from astk.units.angle import Angle
from astk.iges.iges_generator import IGESGenerator
from astk import TEST_DIR



def test_nurbs_revolve():
    axis = Line3D(p0=Point3D.from_array(np.array([0.0, 0.0, 0.0])),
                  p1=Point3D.from_array(np.array([0.0, 0.0, 1.0])))
    cubic_bezier_cps = np.array([
        [0.0, -1.0, 0.0],
        [0.0, -1.2, 0.5],
        [0.0, -1.3, 1.0],
        [0.0, -0.8, 1.5]
    ])
    bezier = Bezier3D([Point3D.from_array(p) for p in cubic_bezier_cps])
    nurbs_surface = NURBSSurface.from_bezier_revolve(bezier, axis, Angle(deg=15.0), Angle(deg=130.0))

    iges_entities = [nurbs_surface.to_iges()]
    cp_net_points, cp_net_lines = nurbs_surface.generate_control_point_net()
    iges_entities.extend([cp_net_point.to_iges() for cp_net_point in cp_net_points])
    iges_entities.extend([cp_net_line.to_iges() for cp_net_line in cp_net_lines])

    iges_file = os.path.join(TEST_DIR, "nurbs_test.igs")
    iges_generator = IGESGenerator(iges_entities, "meters")
    iges_generator.generate(iges_file)

    point_array = nurbs_surface.evaluate(30, 30)
    for point in point_array[:, 0, :]:
        radius = np.sqrt(point[0] ** 2 + point[1] ** 2)
        assert np.isclose(radius, 1.0, 1e-10)
    for point in point_array[:, -1, :]:
        radius = np.sqrt(point[0] ** 2 + point[1] ** 2)
        assert np.isclose(radius, 0.8, 1e-10)


def test_bezier_surface_1():
    """
    Tests the continuity enforcement method across many random pairs of 4x4 ``BezierSurface``s.
    """
    # FOR TESTING 4x4 and 4x4 first
    n = 4
    m = 4
    num_samples = 50
    rng = np.random.default_rng(seed=42)

    cp_sets_1 = rng.random((num_samples, n+1, m+1, 3))
    cp_sets_2 = rng.random((num_samples, n+1, m+1, 3))

    #Loop through different sides of the 4x4
    
    for i in range(4):
        for j in range(4):
            side_self=SurfaceEdge(i)
            side_other=SurfaceEdge(j)



            # Loop through each pair of control point meshes
            for cp_set1, cp_set2 in zip(cp_sets_1, cp_sets_2):
                bez_surf_1 = BezierSurface(cp_set1)
                bez_surf_2 = BezierSurface(cp_set2)

                # Enforce G0, G1, and G2 continuity
                bez_surf_1.enforce_g0g1g2(bez_surf_2, 1.0, side_self, side_other)

                # Verify G0, G1, and G2 continuity
                bez_surf_1.verify_g0(bez_surf_2, side_self, side_other)
                bez_surf_1.verify_g1(bez_surf_2, side_self, side_other)
                bez_surf_1.verify_g2(bez_surf_2, side_self, side_other)
