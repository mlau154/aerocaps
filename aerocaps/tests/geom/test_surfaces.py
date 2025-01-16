from unittest import TestCase

import numpy as np
import pyvista as pv

import aerocaps as ac
from build.lib.aerocaps.iges.iges_generator import IGESGenerator

_SHOW_PLOTS = False


class TestBezierSurface(TestCase):
    def test_enforce_g0g1_multiface(self):
        bez_surf_u0 = ac.BezierSurface(np.array([
            [
                [0.0, 0.0, 0.0],
                [0.3, 0.1, 0.1],
                [0.6, -0.1, 0.0],
                [1.0, 0.0, 0.0]
            ],
            [
                [0.0, 0.5, 0.0],
                [0.3, 0.7, 0.1],
                [0.6, 0.4, 0.1],
                [1.0, 0.5, 0.0]
            ],
            [
                [0.0, 1.0, 0.0],
                [0.3, 1.1, 0.2],
                [0.6, 0.9, 0.1],
                [1.0, 1.0, 0.0]
            ]
        ]))
        bez_surf_u1 = ac.BezierSurface(np.array([
            [
                [0.0, -1.0, 0.0],
                [0.3, -0.7, 0.1],
                [0.6, -1.0, 0.0],
                [1.0, -1.0, 0.0]
            ],
            [
                [0.0, -1.5, 0.0],
                [0.3, -1.3, 0.2],
                [0.6, -1.4, 0.2],
                [1.0, -1.5, 0.0]
            ],
            [
                [0.0, -2.0, 0.0],
                [0.3, -1.9, 0.2],
                [0.6, -2.1, 0.0],
                [1.0, -2.0, 0.0]
            ]
        ]))
        bez_surf_v0 = ac.BezierSurface(np.array([
            [
                [0.0, 0.0, 0.0],
                [-0.1, -0.3, 0.0],
                [0.1, -0.8, 0.1],
                [0.0, -1.0, 0.0]
            ],
            [
                [-1.0, 0.0, 0.0],
                [-1.1, -0.4, 0.1],
                [-0.9, -0.8, 0.0],
                [-1.0, -1.1, 0.0]
            ],
            [
                [-2.0, 0.0, 0.0],
                [-2.0, -0.3, 0.0],
                [-2.0, -0.7, 0.0],
                [-2.0, -1.0, 0.0]
            ]
        ]))
        bez_surf_v1 = ac.BezierSurface(np.array([
            [
                [1.0, 0.0, 0.0],
                [0.9, -0.3, 0.0],
                [0.8, -0.7, 0.1],
                [1.0, -1.0, 0.0]
            ],
            [
                [2.0, 0.0, 0.0],
                [2.1, -0.3, 0.1],
                [1.9, -0.7, 0.0],
                [2.0, -1.1, 0.0]
            ],
            [
                [2.5, 0.0, 0.0],
                [2.5, -0.4, 0.0],
                [2.5, -0.6, 0.0],
                [2.5, -1.0, 0.0]
            ]
        ]))
        bez_surf_target = ac.BezierSurface(np.zeros((4, 4, 3)))
        bez_surf_target.enforce_g0g1_multiface(
            bez_surf_target,
            f_u0=1.0,
            f_u1=1.0,
            f_v0=1.0,
            f_v1=1.0,
            adjacent_surf_u0=bez_surf_u0,
            adjacent_surf_u1=bez_surf_u1,
            adjacent_surf_v0=bez_surf_v0,
            adjacent_surf_v1=bez_surf_v1,
            other_edge_u0=ac.SurfaceEdge.u0,
            other_edge_u1=ac.SurfaceEdge.u0,
            other_edge_v0=ac.SurfaceEdge.u0,
            other_edge_v1=ac.SurfaceEdge.u0
        )

        if _SHOW_PLOTS:
            surfs = [bez_surf_u0, bez_surf_u1, bez_surf_v0, bez_surf_v1, bez_surf_target]
            iges = IGESGenerator([surf.to_iges() for surf in surfs], units="meters")
            iges.generate("surf_opt_test.igs")
            colors = ["blue", "red", "yellow", "purple", "green"]
            plot = pv.Plotter()
            for idx, (surf, color) in enumerate(zip(surfs, colors)):
                surf.plot_surface(
                    plot,
                    50,
                    50,
                    color=color,
                    show_edges=True if idx == 4 else False
                )
                surf.plot_control_points(
                    plot,
                    render_points_as_spheres=True,
                    point_size=16,
                    color="lime" if idx == 4 else "black"
                )
                surf.plot_control_point_mesh_lines(
                    plot,
                    color="gray"
                )
            plot.add_axes()
            plot.show()
