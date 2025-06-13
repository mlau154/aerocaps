import numpy as np
from scipy.optimize import minimize

import aerocaps as ac
from aerocaps.examples.bezier_surface import bezier_surface_2x3
from aerocaps.iges.iges_generator import IGESGenerator


def test_one_edge():
    surf_1 = bezier_surface_2x3()
    points = np.array([
        [
            [0.0, 0.0, 0.0],
            [0.3, 0.0, 0.1],
            [0.6, 0.0, -0.1],
            [1.0, 0.0, 0.0]
        ],
        [
            [0.0, -0.2, 0.1],
            [0.3, -0.3, 0.3],
            [0.6, -0.2, -0.1],
            [1.0, -0.075, -0.075]
        ],
        [
            [0.0, -0.4, 0.2],
            [0.3, -0.5, 0.2],
            [0.6, -0.3, 0.1],
            [1.0, -0.15625, -0.125]
        ]
    ])
    surf_2 = ac.BezierSurface(points)

    points = np.array([
        [
            [1.0, 0.0, 0.0],
            [1.0, -0.075, -0.075],
            [1.0, -0.15625, -0.125]
        ],
        [
            [1.2, 0.0, -0.1],
            [1.2, -0.075, -0.1],
            [1.25, -0.15625, -0.2]
        ],
        [
            [1.4, 0.0, -0.15],
            [1.4, -0.075, -0.15],
            [1.35, -0.15625, -0.25]
        ]
    ])
    surf_3 = ac.BezierSurface(points)

    n_points = 10
    d_g1_1 = surf_1.get_first_derivs_along_edge(ac.SurfaceEdge.u0, n_points=n_points)
    d_g2_1 = surf_1.get_second_derivs_along_edge(ac.SurfaceEdge.u0, n_points=n_points)
    d_g1_3 = surf_3.get_first_derivs_along_edge(ac.SurfaceEdge.u0, n_points=n_points)
    d_g2_3 = surf_3.get_second_derivs_along_edge(ac.SurfaceEdge.u0, n_points=n_points)

    def optimize():
        f1 = 0.3
        f2 = 1.0
        # x0_g1 = surf_2.get_control_point_array()[1, :-1, :].flatten()
        # x0_g1_b = surf_2.get_control_point_array()[2, 2, :].flatten()
        # x0_g1 = np.append(x0_g1, x0_g1_b)
        # x0_g2 = surf_2.get_control_point_array()[2, :-2, :].flatten()

        x0 = surf_2.get_control_point_array()[1:, :-1, :].flatten()

        # def obj_fun_g1(x):
        #     x_reshaped = x.reshape((4, 3))
        #     surf_2.points[1][0] = ac.Point3D.from_array(x_reshaped[0])
        #     surf_2.points[1][1] = ac.Point3D.from_array(x_reshaped[1])
        #     surf_2.points[1][2] = ac.Point3D.from_array(x_reshaped[2])
        #     surf_2.points[2][2] = ac.Point3D.from_array(x_reshaped[3])
        #     d_g1_2 = surf_2.get_first_derivs_along_edge(ac.SurfaceEdge.u0, n_points=n_points)
        #     d_g1_2b = surf_2.get_first_derivs_along_edge(ac.SurfaceEdge.v1, n_points=n_points)
        #     return np.sum((d_g1_1 + 1/f1 * d_g1_2)**2) + np.sum((d_g1_3 - 1/f2 * d_g1_2b)**2)
        #
        # def obj_fun_g2(x):
        #     x_reshaped = x.reshape((2, 3))
        #     surf_2.points[2][0] = ac.Point3D.from_array(x_reshaped[0])
        #     surf_2.points[2][1] = ac.Point3D.from_array(x_reshaped[1])
        #     d_g2_2 = surf_2.get_second_derivs_along_edge(ac.SurfaceEdge.u0, n_points=n_points)
        #     d_g2_2b = surf_2.get_second_derivs_along_edge(ac.SurfaceEdge.v1, n_points=n_points)
        #     return np.sum((d_g2_1 + 1/f1**2 * d_g2_2)**2) + np.sum((d_g2_3 - 1/f2**2 * d_g2_2b)**2)

        # res_g1 = minimize(obj_fun_g1, x0_g1, method="SLSQP")
        # res_g2 = minimize(obj_fun_g2, x0_g2, method="SLSQP")
        # print(f"{res_g1 = }")
        # print(f"{res_g2 = }")

        def obj_fun(x):
            x_reshaped = x.reshape((2, 3, 3))
            for i in range(x_reshaped.shape[0]):
                for j in range(x_reshaped.shape[1]):
                    surf_2.points[i + 1][j] = ac.Point3D.from_array(x_reshaped[i, j])
            d_g1_2 = surf_2.get_first_derivs_along_edge(ac.SurfaceEdge.u0, n_points=n_points)
            d_g1_2b = surf_2.get_first_derivs_along_edge(ac.SurfaceEdge.v1, n_points=n_points)
            d_g2_2 = surf_2.get_second_derivs_along_edge(ac.SurfaceEdge.u0, n_points=n_points)
            d_g2_2b = surf_2.get_second_derivs_along_edge(ac.SurfaceEdge.v1, n_points=n_points)
            J1 = np.sum((d_g1_1 + 1/f1 * d_g1_2)**2)
            J2 = np.sum((d_g1_3 - 1/f2 * d_g1_2b)**2)
            J3 = np.sum((d_g2_1 + 1/f1**2 * d_g2_2)**2)
            J4 = np.sum((d_g2_3 - 1/f2**2 * d_g2_2b)**2)
            return J1 + J2 + J3 + J4

        res = minimize(obj_fun, x0)
        print(f"{res = }")

    optimize()

    # iges_generator = IGESGenerator([surf_1.to_iges(), surf_2.to_iges(), surf_3.to_iges()], units="meters")
    # iges_generator.generate("surface_optimizer.igs")

    # import pyvista as pv
    # plot = pv.Plotter()
    # surf_1.plot_surface(plot, color="blue")
    # surf_2.plot_surface(plot, color="red")
    # surf_2.plot_control_points(plot, color="black", point_size=20, render_points_as_spheres=True)
    # surf_2.plot_control_point_mesh_lines(plot, color="orange")
    # surf_3.plot_surface(plot, color="green")
    # plot.show()


if __name__ == "__main__":
    test_one_edge()
