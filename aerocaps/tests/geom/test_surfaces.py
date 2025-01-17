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
            plot = pv.Plotter(off_screen=True)
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

    def test_enforce_g0g1g2_multiface(self):
        bez_surf_u0 = ac.BezierSurface(np.array([
            [
                [0.0, 0.0, 0.0],
                [0.2, 0.05, 0.1],
                [0.3, 0.1, 0.1],
                [0.5, 0.0, 0.1],
                [0.7, -0.1, 0.0],
                [1.0, 0.0, 0.0]
            ],
            [
                [0.0, 0.5, 0.0],
                [0.2, 0.6, 0.0],
                [0.3, 0.7, 0.1],
                [0.5, 0.6, 0.0],
                [0.6, 0.4, 0.1],
                [1.0, 0.5, 0.0]
            ],
            [
                [0.0, 1.0, 0.0],
                [0.15, 1.0, 0.1],
                [0.3, 1.1, 0.2],
                [0.6, 1.0, 0.1],
                [0.6, 0.9, 0.1],
                [1.0, 1.0, 0.0]
            ]
        ]))
        bez_surf_u1 = ac.BezierSurface(np.array([
            [
                [0.0, -1.0, 0.0],
                [0.2, -0.9, 0.0],
                [0.3, -0.7, 0.1],
                [0.45, -0.9, 0.0],
                [0.6, -1.0, 0.0],
                [1.0, -1.0, 0.0]
            ],
            [
                [0.0, -1.5, 0.0],
                [0.15, -1.4, 0.0],
                [0.3, -1.3, 0.2],
                [0.5, -1.5, 0.1],
                [0.6, -1.4, 0.2],
                [1.0, -1.5, 0.0]
            ],
            [
                [0.0, -2.0, 0.0],
                [0.2, -2.0, 0.1],
                [0.3, -1.9, 0.2],
                [0.5, -2.0, 0.1],
                [0.6, -2.1, 0.0],
                [1.0, -2.0, 0.0]
            ]
        ]))
        bez_surf_v0 = ac.BezierSurface(np.array([
            [
                [0.0, 0.0, 0.0],
                [0.0, -0.4, 0.1],
                [-0.1, -0.3, 0.0],
                [0.0, -0.6, 0.0],
                [0.1, -0.8, 0.1],
                [0.0, -1.0, 0.0]
            ],
            [
                [-1.0, 0.0, 0.0],
                [-1.0, -0.2, 0.0],
                [-1.1, -0.4, 0.1],
                [-1.0, -0.6, 0.1],
                [-0.9, -0.8, 0.0],
                [-1.0, -1.1, 0.0]
            ],
            [
                [-2.0, 0.0, 0.0],
                [-2.0, -0.2, 0.0],
                [-2.0, -0.3, 0.0],
                [-2.1, -0.5, 0.1],
                [-2.0, -0.7, 0.0],
                [-2.0, -1.0, 0.0]
            ]
        ]))
        bez_surf_v1 = ac.BezierSurface(np.array([
            [
                [1.0, 0.0, 0.0],
                [1.0, -0.1, 0.0],
                [0.9, -0.3, 0.0],
                [1.0, -0.5, 0.0],
                [0.8, -0.7, 0.1],
                [1.0, -1.0, 0.0]
            ],
            [
                [2.0, 0.0, 0.0],
                [2.0, -0.1, 0.0],
                [2.1, -0.3, 0.1],
                [2.0, -0.5, 0.05],
                [1.9, -0.7, 0.0],
                [2.0, -1.1, 0.0]
            ],
            [
                [2.5, 0.0, 0.0],
                [2.5, -0.2, 0.0],
                [2.5, -0.4, 0.0],
                [2.6, -0.5, 0.1],
                [2.5, -0.6, 0.0],
                [2.5, -1.0, 0.0]
            ]
        ]))
        bez_surf_target = ac.BezierSurface(np.zeros((6, 6, 3)))
        res = bez_surf_target.enforce_g0g1g2_multiface(
            adjacent_surf_u0=bez_surf_u0,
            adjacent_surf_u1=bez_surf_u1,
            adjacent_surf_v0=bez_surf_v0,
            adjacent_surf_v1=bez_surf_v1,
            other_edge_u0=ac.SurfaceEdge.u0,
            other_edge_u1=ac.SurfaceEdge.u0,
            other_edge_v0=ac.SurfaceEdge.u0,
            other_edge_v1=ac.SurfaceEdge.u0,
            f_u0_initial=0.2,
            f_u1_initial=0.2,
            f_v0_initial=0.2,
            f_v1_initial=0.2
        )
        print(f"{res = }")

        if _SHOW_PLOTS:
            surfs = [bez_surf_u0, bez_surf_u1, bez_surf_v0, bez_surf_v1, bez_surf_target]

            container = ac.GeometryContainer()
            for surf in surfs:
                container.add_geometry(surf)
            container.plot()
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


def test_dsdmu_sens_to_row_3():
    """
    Test if the perpendicular first derivative along a boundary of a Bezier surface is affected by the
    position of the control points in row 3
    """
    surf = ac.BezierSurface(np.array([
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
    for j in range(4):
        edge_derivs = surf.get_first_deriv_cp_sens_along_edge(ac.SurfaceEdge.u0, 2, j, perp=True)
        assert np.all(edge_derivs == 0.0)
