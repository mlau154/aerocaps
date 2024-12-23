import typing
from enum import Enum

import numpy as np
import pyvista as pv

import astk.iges.entity
import astk.iges.surfaces
from astk.geom import Surface, InvalidGeometryError, NegativeWeightError
from astk.geom.point import Point3D
from astk.geom.curves import Bezier3D, Line3D
from astk.geom.tools import project_point_onto_line, measure_distance_point_line, rotate_point_about_axis
from astk.geom.vector import Vector3D
from astk.units.angle import Angle
from astk.units.length import Length
from astk.utils.math import bernstein_poly


__all__ = [
    "SurfaceEdge",
    "SurfaceCorner",
    "BezierSurface",
    "RationalBezierSurface",
    "NURBSSurface"
]


class SurfaceEdge(Enum):
    """
    North
    ^
    |
    |
    |________> u
    South
    """
    North = 0
    South = 1
    East = 2
    West = 3


class SurfaceCorner(Enum):
    Northeast = 0
    Northwest = 1
    Southwest = 2
    Southeast = 3


class BezierSurface(Surface):
    def __init__(self, points: typing.List[typing.List[Point3D]] or np.ndarray):
        if isinstance(points, np.ndarray):
            points = [[Point3D.from_array(pt_row) for pt_row in pt_mat] for pt_mat in points]
        self.points = points
        self.degree_u = len(points) - 1
        self.degree_v = len(points[0]) - 1
        self.Nu = self.degree_u + 1
        self.Nv = self.degree_v + 1

    def to_iges(self, *args, **kwargs) -> astk.iges.entity.IGESEntity:
        return astk.iges.surfaces.BezierSurfaceIGES(self.get_control_point_array())

    def get_control_point_array(self) -> np.ndarray:
        return np.array([np.array([p.as_array() for p in p_arr]) for p_arr in self.points])

    @classmethod
    def generate_from_array(cls, P: np.ndarray):
        return cls([
            [Point3D(x=Length(m=xyz[0]), y=Length(m=xyz[1]), z=Length(m=xyz[2])) for xyz in point_arr]
            for point_arr in P])

    def evaluate_ndarray(self, u: float, v: float):
        P = self.get_control_point_array()

        # Evaluate the surface
        point = np.zeros(P.shape[2])
        for i in range(self.degree_u + 1):
            for j in range(self.degree_v + 1):
                Bu = bernstein_poly(self.degree_u, i, u)
                Bv = bernstein_poly(self.degree_v, j, v)
                BuBv = Bu * Bv
                point += P[i, j, :] * BuBv

        return point
    
    def dSdu(self, u: float, v: float):
        P = self.get_control_point_array()
        deriv_u = np.zeros(P.shape[2])
        for i in range(self.degree_u + 1):
            for j in range(self.degree_v + 1):                
                dbudu=self.degree_u*(bernstein_poly(self.degree_u-1, i-1, u)-bernstein_poly(self.degree_u-1, i, u))
                bv=bernstein_poly(self.degree_v, j, v)
                dbudubv=dbudu*bv
                deriv_u += P[i, j, :] * dbudubv
        return deriv_u
    
    def dSdv(self, u: float, v: float):
        P = self.get_control_point_array()
        deriv_v = np.zeros(P.shape[2])
        for i in range(self.degree_u + 1):
            for j in range(self.degree_v + 1):
                dbvdv=self.degree_v*(bernstein_poly(self.degree_v-1, j-1, v)-bernstein_poly(self.degree_v-1, j, v))
                bu=bernstein_poly(self.degree_u, i, u)
                budbvdv=bu*dbvdv
                deriv_v +=P[i, j, :] * budbvdv
        return deriv_v
    
    def d2Sdu2(self, u: float, v: float):
        P = self.get_control_point_array()
        deriv_u_2 = np.zeros(P.shape[2])
        for i in range(self.degree_u + 1):
            for j in range(self.degree_v + 1): 
                term1=self.degree_u*(self.degree_u-1)*(bernstein_poly(self.degree_u-2,i-2,u)-bernstein_poly(self.degree_u-2,i-1,u))
                term2=self.degree_u*(self.degree_u-1)*(bernstein_poly(self.degree_u-2,i-1,u)-bernstein_poly(self.degree_u-2,i,u))
                d2budu2=term1-term2
                Bv = bernstein_poly(self.degree_v, j, v)
                d2budu2_Bv=d2budu2*Bv
                deriv_u_2 +=P[i, j, :] * d2budu2_Bv
        return deriv_u_2
    
    def d2Sdv2(self, u: float, v: float):
        P = self.get_control_point_array()
        deriv_v_2 = np.zeros(P.shape[2])
        for i in range(self.degree_u + 1):
            for j in range(self.degree_v + 1): 
                term1=self.degree_v*(self.degree_v-1)*(bernstein_poly(self.degree_v-2,j-2,v)-bernstein_poly(self.degree_v-2,j-1,v))
                term2=self.degree_v*(self.degree_v-1)*(bernstein_poly(self.degree_v-2,j-1,v)-bernstein_poly(self.degree_v-2,j,v))
                d2bvdv2=term1-term2
                Bu=bernstein_poly(self.degree_u, i, u)
                Bu_d2bvdv2=Bu*d2bvdv2
                deriv_v_2 +=P[i, j, :] * Bu_d2bvdv2
        return deriv_v_2

    def get_edge(self, edge: SurfaceEdge, n_points: int = 10) -> np.ndarray:
        if edge == SurfaceEdge.North:
            return np.array([self.evaluate_ndarray(u, 1) for u in np.linspace(0.0, 1.0, n_points)])
        elif edge == SurfaceEdge.South:
            return np.array([self.evaluate_ndarray(u, 0) for u in np.linspace(0.0, 1.0, n_points)])
        elif edge == SurfaceEdge.East:
            return np.array([self.evaluate_ndarray(1, v) for v in np.linspace(0.0, 1.0, n_points)])
        elif edge == SurfaceEdge.West:
            return np.array([self.evaluate_ndarray(0, v) for v in np.linspace(0.0, 1.0, n_points)])
        else:
            raise ValueError(f"No edge called {edge}")

    def get_first_derivs_along_edge(self, edge: SurfaceEdge, n_points: int = 10, perp=True) -> np.ndarray:
        if edge == SurfaceEdge.North:
            return np.array([(self.dSdv(u, 1.0) if perp else
                              self.dSdu(u, 1.0)) for u in np.linspace(0.0, 1.0, n_points)])
        elif edge == SurfaceEdge.South:
            return np.array([(self.dSdv(u, 0.0) if perp else
                              self.dSdu(u, 0.0)) for u in np.linspace(0.0, 1.0, n_points)])
        elif edge == SurfaceEdge.East:
            return np.array([(self.dSdu(1.0, v) if perp else
                              self.dSdv(1.0, v)) for v in np.linspace(0.0, 1.0, n_points)])
        elif edge == SurfaceEdge.West:
            return np.array([(self.dSdu(0.0, v) if perp else
                              self.dSdv(0.0, v)) for v in np.linspace(0.0, 1.0, n_points)])
        else:
            raise ValueError(f"No edge called {edge}")

    def get_second_derivs_along_edge(self, edge: SurfaceEdge, n_points: int = 10, perp=True) -> np.ndarray:
        if edge == SurfaceEdge.North:
            return np.array([(self.d2Sdv2(u, 1.0) if perp else
                              self.d2Sdu2(u, 1.0)) for u in np.linspace(0.0, 1.0, n_points)])
        elif edge == SurfaceEdge.South:
            return np.array([(self.d2Sdv2(u, 0.0) if perp else
                              self.d2Sdu2(u, 0.0)) for u in np.linspace(0.0, 1.0, n_points)])
        elif edge == SurfaceEdge.East:
            return np.array([(self.d2Sdu2(1.0, v) if perp else
                              self.d2Sdv2(1.0, v)) for v in np.linspace(0.0, 1.0, n_points)])
        elif edge == SurfaceEdge.West:
            return np.array([(self.d2Sdu2(0.0, v) if perp else
                              self.d2Sdv2(0.0, v)) for v in np.linspace(0.0, 1.0, n_points)])
        else:
            raise ValueError(f"No edge called {edge}")

    def verify_g0(self, other: "BezierSurface", surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge,
                  n_points: int = 10):
        """
        Verifies that two BezierSurfaces are G0 continuous along their shared edge
        """
        self_edge = self.get_edge(surface_edge, n_points=n_points)
        other_edge = other.get_edge(other_surface_edge, n_points=n_points)
        assert np.array_equal(self_edge, other_edge)

    def verify_g1(self, other: "BezierSurface", surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge,
                  n_points: int = 10):
        """
        Verifies that two BezierSurfaces are G1 continuous along their shared edge
        """
        # Get the first derivatives at the boundary and perpendicular to the boundary for each surface,
        # evaluated at "n_points" locations along the boundary
        self_perp_edge_derivs = self.get_first_derivs_along_edge(surface_edge, n_points=n_points, perp=True)
        other_perp_edge_derivs = other.get_first_derivs_along_edge(other_surface_edge, n_points=n_points, perp=True)

        # Initialize an array of ratios of magnitude of the derivative values at each point for both sides
        # of the boundary
        magnitude_ratios = []

        # Loop over each pair of cross-derivatives evaluated along the boundary
        for point_idx, (self_perp_edge_deriv, other_perp_edge_deriv) in enumerate(zip(
                self_perp_edge_derivs, other_perp_edge_derivs)):

            # Ensure that each derivative vector has the same direction along the boundary for each surface
            try:
                assert np.allclose(
                    np.nan_to_num(self_perp_edge_deriv / np.linalg.norm(self_perp_edge_deriv)),
                    np.nan_to_num(other_perp_edge_deriv / np.linalg.norm(other_perp_edge_deriv))
                )
            except AssertionError:
                assert np.allclose(
                    np.nan_to_num(self_perp_edge_deriv / np.linalg.norm(self_perp_edge_deriv)),
                    np.nan_to_num(-other_perp_edge_deriv / np.linalg.norm(other_perp_edge_deriv))
                )


            # Compute the ratio of the magnitudes for each derivative vector along the boundary for each surface.
            # These will be compared at the end.
            with np.errstate(divide="ignore"):
                magnitude_ratios.append(self_perp_edge_deriv / other_perp_edge_deriv)


        # Assert that the first derivatives along each boundary are proportional
        current_f = None
        for magnitude_ratio in magnitude_ratios:
            for dxdydz_ratio in magnitude_ratio:
                if np.isinf(dxdydz_ratio) or dxdydz_ratio == 0.0:
                    continue
                if current_f is None:
                    current_f = dxdydz_ratio
                    continue
                assert np.isclose(dxdydz_ratio, current_f)

    def verify_g2(self, other: "BezierSurface", surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge,
                  n_points: int = 10):
        """
        Verifies that two BezierSurfaces are G2 continuous along their shared edge
        """
        # Get the first derivatives at the boundary and perpendicular to the boundary for each surface,
        # evaluated at "n_points" locations along the boundary
        self_perp_edge_derivs = self.get_second_derivs_along_edge(surface_edge, n_points=n_points, perp=True)
        other_perp_edge_derivs = other.get_second_derivs_along_edge(other_surface_edge, n_points=n_points, perp=True)

        # Initialize an array of ratios of magnitude of the derivative values at each point for both sides
        # of the boundary
        magnitude_ratios = []

        # Loop over each pair of cross-derivatives evaluated along the boundary
        for point_idx, (self_perp_edge_deriv, other_perp_edge_deriv) in enumerate(zip(
                self_perp_edge_derivs, other_perp_edge_derivs)):

            # Ensure that each derivative vector has the same direction along the boundary for each surface
            assert np.allclose(
                np.nan_to_num(self_perp_edge_deriv / np.linalg.norm(self_perp_edge_deriv)),
                np.nan_to_num(other_perp_edge_deriv / np.linalg.norm(other_perp_edge_deriv))
            )

            # Compute the ratio of the magnitudes for each derivative vector along the boundary for each surface.
            # These will be compared at the end.
            with np.errstate(divide="ignore"):
                magnitude_ratios.append(self_perp_edge_deriv / other_perp_edge_deriv)

        # Assert that the second derivatives along each boundary are proportional
        current_f = None
        for magnitude_ratio in magnitude_ratios:
            for dxdydz_ratio in magnitude_ratio:
                if np.isinf(dxdydz_ratio) or dxdydz_ratio == 0.0:
                    continue
                if current_f is None:
                    current_f = dxdydz_ratio
                    continue
                assert np.isclose(dxdydz_ratio, current_f)

    def evaluate_simple(self, u: float, v: float):
        return Point3D.from_array(self.evaluate_ndarray(u, v))

    def evaluate(self, Nu: int, Nv: int) -> np.ndarray:
        U, V = np.meshgrid(np.linspace(0.0, 1.0, Nu), np.linspace(0.0, 1.0, Nv))
        return np.array(
            [[self.evaluate_ndarray(U[i, j], V[i, j]) for j in range(U.shape[1])] for i in range(U.shape[0])]
        )

    def extract_edge_curve(self, surface_edge: SurfaceEdge):
        P = self.get_control_point_array()

        if surface_edge == SurfaceEdge.West:
            return Bezier3D.generate_from_array(P[0, :, :])
        if surface_edge == SurfaceEdge.East:
            return Bezier3D.generate_from_array(P[-1, :, :])
        if surface_edge == SurfaceEdge.South:
            return Bezier3D.generate_from_array(P[:, 0, :])
        if surface_edge == SurfaceEdge.North:
            return Bezier3D.generate_from_array(P[:, -1, :])

    def extract_isoparametric_curve_u(self, Nu: int, v: float):
        u_vec = np.linspace(0.0, 1.0, Nu)
        return np.array([self.evaluate_ndarray(u, v) for u in u_vec])

    def extract_isoparametric_curve_v(self, Nv: int, u: float):
        v_vec = np.linspace(0.0, 1.0, Nv)
        return np.array([self.evaluate_ndarray(u, v) for v in v_vec])

    def get_parallel_degree(self, surface_edge: SurfaceEdge):
        if surface_edge in [SurfaceEdge.North, SurfaceEdge.South]:
            return self.degree_u
        return self.degree_v

    def get_perpendicular_degree(self, surface_edge: SurfaceEdge):
        if surface_edge in [SurfaceEdge.North, SurfaceEdge.South]:
            return self.degree_v
        return self.degree_u

    def get_point(self, row_index: int, continuity_index: int, surface_edge: SurfaceEdge):
        if surface_edge == SurfaceEdge.North:
            return self.points[row_index][-(continuity_index + 1)]
        elif surface_edge == SurfaceEdge.South:
            return self.points[row_index][continuity_index]
        elif surface_edge == SurfaceEdge.East:
            return self.points[-(continuity_index + 1)][row_index]
        elif surface_edge == SurfaceEdge.West:
            return self.points[continuity_index][row_index]
        else:
            raise ValueError("Invalid surface_edge value")

    def set_point(self, point: Point3D, row_index: int, continuity_index: int, surface_edge: SurfaceEdge):
        if surface_edge == SurfaceEdge.North:
            self.points[row_index][-(continuity_index + 1)] = point
        elif surface_edge == SurfaceEdge.South:
            self.points[row_index][continuity_index] = point
        elif surface_edge == SurfaceEdge.East:
            self.points[-(continuity_index + 1)][row_index] = point
        elif surface_edge == SurfaceEdge.West:
            self.points[continuity_index][row_index] = point
        else:
            raise ValueError("Invalid surface_edge value")

    def enforce_g0(self, other: "BezierSurface",
                   surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge):

        assert self.get_parallel_degree(surface_edge) == other.get_parallel_degree(other_surface_edge)
        for row_index in range(self.get_parallel_degree(surface_edge) + 1):
            self.set_point(other.get_point(row_index, 0, other_surface_edge), row_index, 0, surface_edge)

    def enforce_c0(self, other: "BezierSurface", surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge):
        self.enforce_g0(other, surface_edge, other_surface_edge)

    def enforce_g0g1(self, other: "BezierSurface", f: float,
                     surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge):
        self.enforce_g0(other, surface_edge, other_surface_edge)
        n_ratio = other.get_perpendicular_degree(other_surface_edge) / self.get_perpendicular_degree(surface_edge)
        for row_index in range(self.get_parallel_degree(surface_edge) + 1):

            P_i0_b = self.get_point(row_index, 0, surface_edge)
            P_im_a = other.get_point(row_index, 0, other_surface_edge)
            P_im1_a = other.get_point(row_index, 1, other_surface_edge)

            P_i1_b = P_i0_b + f * n_ratio * (P_im_a - P_im1_a)
            self.set_point(P_i1_b, row_index, 1, surface_edge)

    def enforce_c0c1(self, other: "BezierSurface",
                     surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge):
        self.enforce_g0g1(other, 1.0, surface_edge, other_surface_edge)

    def enforce_g0g1g2(self, other: "BezierSurface", f: float,
                       surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge):
        self.enforce_g0g1(other, f, surface_edge, other_surface_edge)
        p_perp_a = other.get_perpendicular_degree(other_surface_edge)
        p_perp_b = self.get_perpendicular_degree(surface_edge)
        n_ratio = (p_perp_a**2 - p_perp_a) / (p_perp_b**2 - p_perp_b)
        for row_index in range(self.get_parallel_degree(surface_edge) + 1):

            P_i0_b = self.get_point(row_index, 0, surface_edge)
            P_i1_b = self.get_point(row_index, 1, surface_edge)
            P_im_a = other.get_point(row_index, 0, other_surface_edge)
            P_im1_a = other.get_point(row_index, 1, other_surface_edge)
            P_im2_a = other.get_point(row_index, 2, other_surface_edge)

            P_i2_b = (2.0 * P_i1_b - P_i0_b) + f**2 * n_ratio * (P_im_a - 2.0 * P_im1_a + P_im2_a)
            self.set_point(P_i2_b, row_index, 2, surface_edge)

    def enforce_c0c1c2(self, other: "BezierSurface",

                       surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge):
        self.enforce_g0g1g2(other, 1.0, surface_edge, other_surface_edge)

    def generate_control_point_net(self) -> (typing.List[Point3D], typing.List[Line3D]):

        points = []
        lines = []
        control_points = self.get_control_point_array()

        for i in range(self.Nu):
            for j in range(self.Nv):
                points.append(Point3D.from_array(control_points[i, j, :]))

        for i in range(self.Nu - 1):
            for j in range(self.Nv - 1):
                point_obj_1 = Point3D.from_array(control_points[i, j, :])
                point_obj_2 = Point3D.from_array(control_points[i + 1, j, :])
                point_obj_3 = Point3D.from_array(control_points[i, j + 1, :])

                line_1 = Line3D(p0=point_obj_1, p1=point_obj_2)
                line_2 = Line3D(p0=point_obj_1, p1=point_obj_3)
                lines.extend([line_1, line_2])

                if i < self.Nu - 2 and j < self.Nv - 2:
                    continue

                point_obj_4 = Point3D.from_array(control_points[i + 1, j + 1, :])
                line_3 = Line3D(p0=point_obj_3, p1=point_obj_4)
                line_4 = Line3D(p0=point_obj_2, p1=point_obj_4)
                lines.extend([line_3, line_4])

        return points, lines

    def plot_surface(self, plot: pv.Plotter, **mesh_kwargs):
        XYZ = self.evaluate(50, 50)
        grid = pv.StructuredGrid(XYZ[:, :, 0], XYZ[:, :, 1], XYZ[:, :, 2])
        plot.add_mesh(grid, **mesh_kwargs)

        return grid

    def plot_control_point_mesh_lines(self, plot: pv.Plotter, **line_kwargs):
        _, line_objs = self.generate_control_point_net()
        line_arr = np.array([[line_obj.p0.as_array(), line_obj.p1.as_array()] for line_obj in line_objs])
        line_arr = line_arr.reshape((len(line_objs) * 2, 3))
        plot.add_lines(line_arr, **line_kwargs)

    def plot_control_points(self,  plot: pv.Plotter, **point_kwargs):
        point_objs, _ = self.generate_control_point_net()
        point_arr = np.array([point_obj.as_array() for point_obj in point_objs])
        plot.add_points(point_arr, **point_kwargs)


class RationalBezierSurface(Surface):
    def __init__(self,
                 points: typing.List[typing.List[Point3D]] or np.ndarray,
                 weights: np.ndarray,
                 ):
        if isinstance(points, np.ndarray):
            points = [[Point3D.from_array(pt_row) for pt_row in pt_mat] for pt_mat in points]
        self.points = points
        knots_u = np.zeros(2 * len(points))
        knots_v = np.zeros(2 * len(points[0]))
        knots_u[len(points):] = 1.0
        knots_v[len(points[0]):] = 1.0
        degree_u = len(points) - 1
        degree_v = len(points[0]) - 1
        assert knots_u.ndim == 1
        assert knots_v.ndim == 1
        assert weights.ndim == 2
        assert len(knots_u) == len(points) + degree_u + 1
        assert len(knots_v) == len(points[0]) + degree_v + 1
        assert len(points) == weights.shape[0]
        assert len(points[0]) == weights.shape[1]

        # Negative weight check
        for weight_row in weights:
            for weight in weight_row:
                if weight < 0:
                    raise NegativeWeightError("All weights must be non-negative")

        self.knots_u = knots_u
        self.knots_v = knots_v
        self.weights = weights
        self.degree_u = degree_u
        self.degree_v = degree_v
        self.Nu, self.Nv = len(points), len(points[0])

    def to_iges(self, *args, **kwargs) -> astk.iges.entity.IGESEntity:
        return astk.iges.surfaces.RationalBSplineSurfaceIGES(
            control_points=self.get_control_point_array(),
            knots_u=self.knots_u,
            knots_v=self.knots_v,
            weights=self.weights,
            degree_u=self.degree_u,
            degree_v=self.degree_v
        )

    def get_control_point_array(self) -> np.ndarray:
        return np.array([np.array([p.as_array() for p in p_arr]) for p_arr in self.points])

    @classmethod
    def generate_from_array(cls, P: np.ndarray, weights: np.ndarray):
        return cls([
            [Point3D(x=Length(m=xyz[0]), y=Length(m=xyz[1]), z=Length(m=xyz[2])) for xyz in point_arr]
            for point_arr in P], weights)

    @classmethod
    def from_bezier_revolve(cls, bezier: Bezier3D, axis: Line3D, start_angle: Angle, end_angle: Angle):

        # if abs(end_angle.rad - start_angle.rad) > np.pi / 2:
        #     raise ValueError("Angle difference must be less than or equal to 90 degrees for a rational Bezier surface"
        #                      " creation from Bezier revolve. For angle differences larger than 90 degrees, use"
        #                      " NURBSSurface.from_bezier_revolve.")

        def _determine_angle_distribution() -> typing.List[Angle]:
            angle_diff = abs(end_angle.rad - start_angle.rad)

            if angle_diff == 0.0:
                raise InvalidGeometryError("Starting and ending angles cannot be the same for a "
                                           "NURBSSurface from revolved Bezier curve")

            if angle_diff % (0.5 * np.pi) == 0.0:  # If angle difference is a multiple of 90 degrees
                N_angles = 2 * int(angle_diff // (0.5 * np.pi)) + 1
            else:
                N_angles = 2 * int(angle_diff // (0.5 * np.pi)) + 3

            rad_dist = np.linspace(start_angle.rad, end_angle.rad, N_angles)
            return [Angle(rad=r) for r in rad_dist]

        control_points = []
        weights = []
        angles = _determine_angle_distribution()

        for point in bezier.points:

            axis_projection = project_point_onto_line(point, axis)
            radius = measure_distance_point_line(point, axis)
            if radius == 0.0:
                new_points = [point for _ in angles]
            else:
                new_points = [rotate_point_about_axis(point, axis, angle) for angle in angles]

            for idx, rotated_point in enumerate(new_points):
                if idx == 0:
                    weights.append([])
                if not idx % 2:  # Skip even indices (these represent the "through" control points)
                    weights[-1].append(1.0)
                    continue
                sine_half_angle = np.sin(0.5 * np.pi - 0.5 * (angles[idx + 1].rad - angles[idx - 1].rad))

                if radius != 0.0:
                    distance = radius / sine_half_angle  # radius / sin(half angle)
                    vector = Vector3D(p0=axis_projection, p1=rotated_point)
                    new_points[idx] = axis_projection + Point3D.from_array(
                        distance * np.array(vector.normalized_value()))

                weights[-1].append(sine_half_angle)

            control_points.append(np.array([new_point.as_array() for new_point in new_points]))

        control_points = np.array(control_points)
        weights = np.array(weights)

        return cls(control_points, weights)

    def evaluate_ndarray(self, u: float, v: float):
        P = self.get_control_point_array()

        # Evaluate the surface
        point = np.zeros(P.shape[2])
        wBuBv_sum = 0.0
        for i in range(self.Nu):
            for j in range(self.Nv):
                Bu = bernstein_poly(self.degree_u, i, u)
                Bv = bernstein_poly(self.degree_v, j, v)
                wBuBv = Bu * Bv * self.weights[i, j]
                wBuBv_sum += wBuBv
                point += P[i, j, :] * wBuBv

        return point / wBuBv_sum

    def evaluate_simple(self, u: float, v: float):
        return Point3D.from_array(self.evaluate_ndarray(u, v))

    def evaluate(self, Nu: int, Nv: int) -> np.ndarray:
        U, V = np.meshgrid(np.linspace(0.0, 1.0, Nu), np.linspace(0.0, 1.0, Nv))
        return np.array(
            [[self.evaluate_ndarray(U[i, j], V[i, j]) for j in range(U.shape[1])] for i in range(U.shape[0])]
        )

    def get_parallel_degree(self, surface_edge: SurfaceEdge):
        if surface_edge in [SurfaceEdge.North, SurfaceEdge.South]:
            return self.degree_u
        return self.degree_v

    def get_perpendicular_degree(self, surface_edge: SurfaceEdge):
        if surface_edge in [SurfaceEdge.North, SurfaceEdge.South]:
            return self.degree_v
        return self.degree_u

    def get_corner_index(self, surface_corner: SurfaceCorner) -> (int, int):
        if surface_corner == SurfaceCorner.Northeast:
            return self.degree_u, self.degree_v
        elif surface_corner == SurfaceCorner.Northwest:
            return 0, self.degree_v
        elif surface_corner == SurfaceCorner.Southwest:
            return 0, 0
        elif surface_corner == SurfaceCorner.Southeast:
            return self.degree_u, 1
        else:
            raise ValueError("Invalid surface_corner value")

    def get_point(self, row_index: int, continuity_index: int, surface_edge: SurfaceEdge):
        if surface_edge == SurfaceEdge.North:
            return self.points[row_index][-(continuity_index + 1)]
        elif surface_edge == SurfaceEdge.South:
            return self.points[row_index][continuity_index]
        elif surface_edge == SurfaceEdge.East:
            return self.points[-(continuity_index + 1)][row_index]
        elif surface_edge == SurfaceEdge.West:
            return self.points[continuity_index][row_index]
        else:
            raise ValueError("Invalid surface_edge value")

    def set_point(self, point: Point3D, row_index: int, continuity_index: int, surface_edge: SurfaceEdge):
        if surface_edge == SurfaceEdge.North:
            self.points[row_index][-(continuity_index + 1)] = point
        elif surface_edge == SurfaceEdge.South:
            self.points[row_index][continuity_index] = point
        elif surface_edge == SurfaceEdge.East:
            self.points[-(continuity_index + 1)][row_index] = point
        elif surface_edge == SurfaceEdge.West:
            self.points[continuity_index][row_index] = point
        else:
            raise ValueError("Invalid surface_edge value")

    def get_weight(self, row_index: int, continuity_index: int, surface_edge: SurfaceEdge):
        if surface_edge == SurfaceEdge.North:
            return self.weights[row_index][-(continuity_index + 1)]
        elif surface_edge == SurfaceEdge.South:
            return self.weights[row_index][continuity_index]
        elif surface_edge == SurfaceEdge.East:
            return self.weights[-(continuity_index + 1)][row_index]
        elif surface_edge == SurfaceEdge.West:
            return self.weights[continuity_index][row_index]
        else:
            raise ValueError("Invalid surface_edge value")

    def set_weight(self, weight: float, row_index: int, continuity_index: int, surface_edge: SurfaceEdge):
        if surface_edge == SurfaceEdge.North:
            self.weights[row_index][-(continuity_index + 1)] = weight
        elif surface_edge == SurfaceEdge.South:
            self.weights[row_index][continuity_index] = weight
        elif surface_edge == SurfaceEdge.East:
            self.weights[-(continuity_index + 1)][row_index] = weight
        elif surface_edge == SurfaceEdge.West:
            self.weights[continuity_index][row_index] = weight
        else:
            raise ValueError("Invalid surface_edge value")

    def enforce_g0(self, other: "RationalBezierSurface",
                   surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge):
        # P^b[:, 0] = P^a[:, -1]
        assert self.get_parallel_degree(surface_edge) == other.get_parallel_degree(other_surface_edge)
        for row_index in range(self.get_parallel_degree(surface_edge) + 1):
            self.set_point(other.get_point(row_index, 0, other_surface_edge), row_index, 0, surface_edge)
            self.set_weight(other.get_weight(row_index, 0, other_surface_edge), row_index, 0, surface_edge)

    def enforce_c0(self, other: "RationalBezierSurface", surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge):
        self.enforce_g0(other, surface_edge, other_surface_edge)

    def enforce_g0g1(self, other: "RationalBezierSurface", f: float or np.ndarray,
                     surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge):

        if isinstance(f, np.ndarray):
            assert len(f) == self.get_parallel_degree(surface_edge) + 1

        self.enforce_g0(other, surface_edge, other_surface_edge)
        n_ratio = other.get_perpendicular_degree(other_surface_edge) / self.get_perpendicular_degree(surface_edge)
        for row_index in range(self.get_parallel_degree(surface_edge) + 1):

            f_row = f if isinstance(f, float) else f[row_index]

            w_i0_b = self.get_weight(row_index, 0, surface_edge)
            w_im_a = other.get_weight(row_index, 0, other_surface_edge)
            w_im1_a = other.get_weight(row_index, 1, other_surface_edge)

            w_i1_b = w_i0_b + f_row * n_ratio * (w_im_a - w_im1_a)

            if w_i1_b < 0:
                raise NegativeWeightError("G1 enforcement generated a negative weight")

            self.set_weight(w_i1_b, row_index, 1, surface_edge)

            P_i0_b = self.get_point(row_index, 0, surface_edge)
            P_im_a = other.get_point(row_index, 0, other_surface_edge)
            P_im1_a = other.get_point(row_index, 1, other_surface_edge)

            P_i1_b = w_i0_b / w_i1_b * P_i0_b + f_row * n_ratio / w_i1_b * (w_im_a * P_im_a - w_im1_a * P_im1_a)
            self.set_point(P_i1_b, row_index, 1, surface_edge)

    def enforce_c0c1(self, other: "RationalBezierSurface",
                     surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge):
        self.enforce_g0g1(other, 1.0, surface_edge, other_surface_edge)

    def enforce_g0g1_multiface(self, f: float,
                               adjacent_surf_north: "RationalBezierSurface" = None,
                               adjacent_surf_south: "RationalBezierSurface" = None,
                               adjacent_surf_east: "RationalBezierSurface" = None,
                               adjacent_surf_west: "RationalBezierSurface" = None):
        pass

    def enforce_g0g1g2(self, other: "RationalBezierSurface", f: float or np.ndarray,
                       surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge):
        self.enforce_g0g1(other, f, surface_edge, other_surface_edge)
        n_ratio = (other.get_perpendicular_degree(other_surface_edge)**2 -
                   other.get_perpendicular_degree(other_surface_edge)) / (
            self.get_perpendicular_degree(surface_edge)**2 - self.get_perpendicular_degree(surface_edge))
        for row_index in range(self.get_parallel_degree(surface_edge) + 1):

            w_i0_b = self.get_weight(row_index, 0, surface_edge)
            w_i1_b = self.get_weight(row_index, 1, surface_edge)
            w_im_a = other.get_weight(row_index, 0, other_surface_edge)
            w_im1_a = other.get_weight(row_index, 1, other_surface_edge)
            w_im2_a = other.get_weight(row_index, 2, other_surface_edge)

            f_row = f if isinstance(f, float) else f[row_index]

            w_i2_b = 2.0 * w_i1_b - w_i0_b + f_row**2 * n_ratio * (w_im_a - 2.0 * w_im1_a + w_im2_a)

            if w_i2_b < 0:
                raise NegativeWeightError("G2 enforcement generated a negative weight")

            self.set_weight(w_i2_b, row_index, 2, surface_edge)

            P_i0_b = self.get_point(row_index, 0, surface_edge)
            P_i1_b = self.get_point(row_index, 1, surface_edge)
            P_im_a = other.get_point(row_index, 0, other_surface_edge)
            P_im1_a = other.get_point(row_index, 1, other_surface_edge)
            P_im2_a = other.get_point(row_index, 2, other_surface_edge)

            P_i2_b = (2.0 * w_i1_b / w_i2_b * P_i1_b - w_i0_b / w_i2_b * P_i0_b) + f_row**2 * n_ratio * (1 / w_i2_b) * (
                    w_im_a * P_im_a - 2.0 * w_im1_a * P_im1_a + w_im2_a * P_im2_a)
            self.set_point(P_i2_b, row_index, 2, surface_edge)

    def enforce_c0c1c2(self, other: "RationalBezierSurface",
                       surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge):
        self.enforce_g0g1g2(other, 1.0, surface_edge, other_surface_edge)
    
    @staticmethod
    def _cast_uv(u: float or np.ndarray, v: float or np.ndarray) -> (float, float) or (np.ndarray, np.ndarray):
        if not isinstance(u, np.ndarray):
            u = np.array([u])
        if not isinstance(v, np.ndarray):
            v = np.array([v])

        #print(f"{u=},{v=}")
        return u, v
    
    def dSdu_v2(self,u: float, v: float):
        n, m = self.degree_u, self.degree_v
        P = self.get_control_point_array()
        w=self.weights
        u,v=self._cast_uv(u,v)
        if isinstance(u, np.ndarray):
            assert u.shape == v.shape

        A_1 = np.zeros(P.shape[2])
        A_2 = np.zeros(P.shape[2])
        B_1 = np.zeros(P.shape[2])
        B_2 = np.zeros(P.shape[2])
        

        for i in range(self.degree_u + 1):
            for j in range(self.degree_v + 1):
                dbudu=self.degree_u*(bernstein_poly(self.degree_u-1, i-1, u)-bernstein_poly(self.degree_u-1, i, u))
                A_1 += bernstein_poly(n,i,u)*bernstein_poly(m,j,v)*w[i,j]
                A_2 += dbudu*bernstein_poly(m,j,v)*w[i,j]*P[i,j,:]
                B_1 += bernstein_poly(n,i,u)*bernstein_poly(m,j,v)*w[i,j]*P[i,j,:]
                B_2 += dbudu*bernstein_poly(m,j,v)*w[i,j]
        A=A_1*A_2
        B=B_1*B_2
        return (A-B)/(A_1**2)
    
    def dSdv_v2(self, u:float, v:float):
        n, m = self.degree_u, self.degree_v
        P = self.get_control_point_array()
        w=self.weights
        u,v=self._cast_uv(u,v)
        if isinstance(u, np.ndarray):
            assert u.shape == v.shape

        A_1 = np.zeros(P.shape[2])
        A_2 = np.zeros(P.shape[2])
        B_1 = np.zeros(P.shape[2])
        B_2 = np.zeros(P.shape[2])

        for i in range(self.degree_u + 1):
            for j in range(self.degree_v + 1):
                dbvdv=m*(bernstein_poly(m-1, j-1, v)-bernstein_poly(m-1, j, v))
                A_1 += bernstein_poly(n,i,u)*bernstein_poly(m,j,v)*w[i,j]
                A_2 += bernstein_poly(n,i,u)*dbvdv*w[i,j]*P[i,j,:]
                B_1 += bernstein_poly(n,i,u)*bernstein_poly(m,j,v)*w[i,j]*P[i,j,:]
                B_2 += bernstein_poly(n,i,u)*dbvdv*w[i,j]
        A=A_1*A_2
        B=B_1*B_2
        return (A-B)/(A_1**2)


    def dSdu(self, u: float or np.ndarray, v: float or np.ndarray):
        n, m = self.degree_u, self.degree_v
        P = self.get_control_point_array()
        u,v=self._cast_uv(u,v)
        if isinstance(u, np.ndarray):
            assert u.shape == v.shape

        weight_arr = np.array([[bernstein_poly(n, i, u) * bernstein_poly(m, j, v) * self.weights[i, j]
                                for j in range(m + 1)] for i in range(n + 1)])
        weight_sum = weight_arr.reshape(-1, weight_arr.shape[-1]).sum(axis=0)

        point_arr = np.array([[np.array([(bernstein_poly(n, i, u) * bernstein_poly(m, j, v) *
                                          self.weights[i, j])]).T @ np.array([P[i, j, :]])
                               for j in range(m + 1)] for i in range(n + 1)])
        point_sum = point_arr.reshape(-1, len(u), 3).sum(axis=0)

        point_arr_deriv = np.array([[np.array([((bernstein_poly(n-1, i-1, u) - bernstein_poly(n-1, i, u)) *
                                                bernstein_poly(m, j, v) * self.weights[i, j])]).T @
                                     np.array([P[i, j, :]]) for j in range(m + 1)] for i in range(n + 1)])
        point_deriv_sum = point_arr_deriv.reshape(-1, len(u), 3).sum(axis=0)

        weight_arr_deriv = np.array([[(bernstein_poly(n-1, i-1, u) - bernstein_poly(n-1, i, u)) *
                                      bernstein_poly(m, j, v) * self.weights[i, j]
                                      for j in range(m + 1)] for i in range(n + 1)])
        weight_deriv_sum = weight_arr_deriv.reshape(-1, weight_arr_deriv.shape[-1]).sum(axis=0)

        A = n * np.tile(weight_sum, (3, 1)).T * point_deriv_sum
        B = n * point_sum * np.tile(weight_deriv_sum, (3, 1)).T
        W = np.tile(weight_sum ** 2, (3, 1)).T

        return (A - B) / W

    def dSdv(self, u: float or np.ndarray, v: float or np.ndarray):
        
        n, m = self.degree_u, self.degree_v
        P = self.get_control_point_array()
        #assert type(u) == type(v)
        u,v=self._cast_uv(u,v)
        if isinstance(u, np.ndarray):
            assert u.shape == v.shape

        weight_arr = np.array([[bernstein_poly(n, i, u) * bernstein_poly(m, j, v) * self.weights[i, j]
                                for j in range(m + 1)] for i in range(n + 1)])
        weight_sum = weight_arr.reshape(-1, weight_arr.shape[-1]).sum(axis=0)

        point_arr = np.array([[np.array([(bernstein_poly(n, i, u) * bernstein_poly(m, j, v) *
                                          self.weights[i, j])]).T @ np.array([P[i, j, :]])
                               for j in range(m + 1)] for i in range(n + 1)])
        point_sum = point_arr.reshape(-1, len(u), 3).sum(axis=0)

        point_arr_deriv = np.array([[np.array([((bernstein_poly(m-1, j-1, v) - bernstein_poly(m-1, j, v)) *
                                                bernstein_poly(n, i, u) * self.weights[i, j])]).T @
                                     np.array([P[i, j, :]]) for j in range(m + 1)] for i in range(n + 1)])
        point_deriv_sum = point_arr_deriv.reshape(-1, len(u), 3).sum(axis=0)

        weight_arr_deriv = np.array([[(bernstein_poly(m-1, j-1, v) - bernstein_poly(m-1, j, v)) *
                                      bernstein_poly(n, i, u) * self.weights[i, j]
                                      for j in range(m + 1)] for i in range(n + 1)])
        weight_deriv_sum = weight_arr_deriv.reshape(-1, weight_arr_deriv.shape[-1]).sum(axis=0)

        A = m * np.tile(weight_sum, (3, 1)).T * point_deriv_sum
        B = m * point_sum * np.tile(weight_deriv_sum, (3, 1)).T
        W = np.tile(weight_sum**2, (3, 1)).T

        return (A - B) / W

    def d2Sdu2(self, u: float or np.ndarray, v: float or np.ndarray):
        n, m = self.degree_u, self.degree_v
        P = self.get_control_point_array()
        #assert type(u) == type(v)
        u,v=self._cast_uv(u,v)
        if isinstance(u, np.ndarray):
            assert u.shape == v.shape

        weight_arr = np.array([[bernstein_poly(n, i, u) * bernstein_poly(m, j, v) * self.weights[i, j]
                                for j in range(m + 1)] for i in range(n + 1)])
        weight_sum = weight_arr.reshape(-1, weight_arr.shape[-1]).sum(axis=0)

        point_arr = np.array([[np.array([(bernstein_poly(n, i, u) * bernstein_poly(m, j, v) *
                                          self.weights[i, j])]).T @ np.array([P[i, j, :]])
                               for j in range(m + 1)] for i in range(n + 1)])
        point_sum = point_arr.reshape(-1, len(u), 3).sum(axis=0)

        point_arr_deriv = np.array([[np.array([((bernstein_poly(n-1, i-1, u) - bernstein_poly(n-1, i, u)) *
                                                bernstein_poly(m, j, v) * self.weights[i, j])]).T @
                                     np.array([P[i, j, :]]) for j in range(m + 1)] for i in range(n + 1)])
        point_deriv_sum = point_arr_deriv.reshape(-1, len(u), 3).sum(axis=0)

        weight_arr_deriv = np.array([[(bernstein_poly(n-1, i-1, u) - bernstein_poly(n-1, i, u)) *
                                      bernstein_poly(m, j, v) * self.weights[i, j]
                                      for j in range(m + 1)] for i in range(n + 1)])
        weight_deriv_sum = weight_arr_deriv.reshape(-1, weight_arr_deriv.shape[-1]).sum(axis=0)

        point_arr_deriv2 = np.array([[np.array([((bernstein_poly(n - 2, i - 2, u) -
                                                  2 * bernstein_poly(n - 2, i - 1, u) +
                                                  bernstein_poly(n - 2, i, u)) *
                                                 bernstein_poly(m, j, v) * self.weights[i, j])]).T @
                                      np.array([P[i, j, :]]) for j in range(m + 1)] for i in range(n + 1)])
        point_deriv2_sum = point_arr_deriv2.reshape(-1, len(u), 3).sum(axis=0)

        weight_arr_deriv2 = np.array([[(bernstein_poly(n - 2, i - 2, u) -
                                        2 * bernstein_poly(n - 2, i - 1, u) +
                                        bernstein_poly(n - 2, i, u)) *
                                       bernstein_poly(m, j, v) * self.weights[i, j]
                                       for j in range(m + 1)] for i in range(n + 1)])
        weight_deriv2_sum = weight_arr_deriv2.reshape(-1, weight_arr_deriv2.shape[-1]).sum(axis=0)

        A = n * np.tile(weight_sum, (3, 1)).T * point_deriv_sum
        B = n * point_sum * np.tile(weight_deriv_sum, (3, 1)).T
        W = np.tile(weight_sum ** 2, (3, 1)).T

        dA = n ** 2 * np.tile(weight_deriv_sum, (3, 1)).T * point_deriv_sum + np.tile(
            weight_sum, (3, 1)).T * point_deriv2_sum
        dB = n ** 2 * point_deriv_sum * np.tile(weight_deriv_sum, (3, 1)).T + point_sum * np.tile(
            weight_deriv2_sum, (3, 1)).T
        dW = 2 * n * np.tile(weight_sum, (3, 1)).T * np.tile(weight_deriv_sum, (3, 1)).T

        return (W * (dA - dB) - dW * (A - B)) / W**2

    def d2Sdv2(self, u: float or np.ndarray, v: float or np.ndarray):
        n, m = self.degree_u, self.degree_v
        P = self.get_control_point_array()
        #assert type(u) == type(v)
        u,v=self._cast_uv(u,v)
        if isinstance(u, np.ndarray):
            assert u.shape == v.shape

        weight_arr = np.array([[bernstein_poly(n, i, u) * bernstein_poly(m, j, v) * self.weights[i, j]
                                for j in range(m + 1)] for i in range(n + 1)])
        weight_sum = weight_arr.reshape(-1, weight_arr.shape[-1]).sum(axis=0)

        point_arr = np.array([[np.array([(bernstein_poly(n, i, u) * bernstein_poly(m, j, v) *
                                          self.weights[i, j])]).T @ np.array([P[i, j, :]])
                               for j in range(m + 1)] for i in range(n + 1)])
        point_sum = point_arr.reshape(-1, len(u), 3).sum(axis=0)

        point_arr_deriv = np.array([[np.array([((bernstein_poly(m-1, j-1, v) - bernstein_poly(m-1, j, v)) *
                                                bernstein_poly(n, i, u) * self.weights[i, j])]).T @
                                     np.array([P[i, j, :]]) for j in range(m + 1)] for i in range(n + 1)])
        point_deriv_sum = point_arr_deriv.reshape(-1, len(u), 3).sum(axis=0)

        weight_arr_deriv = np.array([[(bernstein_poly(m-1, j-1, v) - bernstein_poly(m-1, j, v)) *
                                      bernstein_poly(n, i, u) * self.weights[i, j]
                                      for j in range(m + 1)] for i in range(n + 1)])
        weight_deriv_sum = weight_arr_deriv.reshape(-1, weight_arr_deriv.shape[-1]).sum(axis=0)

        point_arr_deriv2 = np.array([[np.array([((bernstein_poly(m-2, j-2, v) -
                                                  2 * bernstein_poly(m-2, j-1, v) +
                                                  bernstein_poly(m-2, j, v)) *
                                                bernstein_poly(n, i, u) * self.weights[i, j])]).T @
                                     np.array([P[i, j, :]]) for j in range(m + 1)] for i in range(n + 1)])
        point_deriv2_sum = point_arr_deriv2.reshape(-1, len(u), 3).sum(axis=0)

        weight_arr_deriv2 = np.array([[(bernstein_poly(m-2, j-2, v) -
                                        2 * bernstein_poly(m-2, j-1, v) +
                                        bernstein_poly(m-2, j, v)) *
                                      bernstein_poly(n, i, u) * self.weights[i, j]
                                      for j in range(m + 1)] for i in range(n + 1)])
        weight_deriv2_sum = weight_arr_deriv2.reshape(-1, weight_arr_deriv2.shape[-1]).sum(axis=0)

        A = m * np.tile(weight_sum, (3, 1)).T * point_deriv_sum
        B = m * point_sum * np.tile(weight_deriv_sum, (3, 1)).T
        W = np.tile(weight_sum**2, (3, 1)).T

        dA = m**2 * np.tile(weight_deriv_sum, (3, 1)).T * point_deriv_sum + np.tile(
            weight_sum, (3, 1)).T * point_deriv2_sum
        dB = m**2 * point_deriv_sum * np.tile(weight_deriv_sum, (3, 1)).T + point_sum * np.tile(
            weight_deriv2_sum, (3, 1)).T
        dW = 2 * m * np.tile(weight_sum, (3, 1)).T * np.tile(weight_deriv_sum, (3, 1)).T

        return (W * (dA - dB) - dW * (A - B)) / W**2
    
    def get_edge(self, edge: SurfaceEdge, n_points: int = 10) -> np.ndarray:
        if edge == SurfaceEdge.North:
            return np.array([self.evaluate_ndarray(u, 1.0) for u in np.linspace(0.0, 1.0, n_points)])
        elif edge == SurfaceEdge.South:
            return np.array([self.evaluate_ndarray(u, 0.0) for u in np.linspace(0.0, 1.0, n_points)])
        elif edge == SurfaceEdge.East:
            return np.array([self.evaluate_ndarray(1.0, v) for v in np.linspace(0.0, 1.0, n_points)])
        elif edge == SurfaceEdge.West:
            return np.array([self.evaluate_ndarray(0.0, v) for v in np.linspace(0.0, 1.0, n_points)])
        else:
            raise ValueError(f"No edge called {edge}")

    def get_first_derivs_along_edge(self, edge: SurfaceEdge, n_points: int = 10, perp=True) -> np.ndarray:
        if edge == SurfaceEdge.North:
            return np.array([(self.dSdv(u, 1.0) if perp else
                              self.dSdu(u, 1.0)) for u in np.linspace(0.0, 1.0, n_points)])
        elif edge == SurfaceEdge.South:
            return np.array([(self.dSdv(u, 0.0) if perp else
                              self.dSdu(u, 0.0)) for u in np.linspace(0.0, 1.0, n_points)])
        elif edge == SurfaceEdge.East:
            return np.array([(self.dSdu(1.0, v) if perp else
                              self.dSdv(1.0, v)) for v in np.linspace(0.0, 1.0, n_points)])
        elif edge == SurfaceEdge.West:
            return np.array([(self.dSdu(0.0, v) if perp else
                              self.dSdv(0.0, v)) for v in np.linspace(0.0, 1.0, n_points)])
        else:
            raise ValueError(f"No edge called {edge}")
        
    def get_first_derivs_along_edge_v2(self, edge: SurfaceEdge, n_points: int = 10, perp=True) -> np.ndarray:
        if edge == SurfaceEdge.North:
            return np.array([(self.dSdv_v2(u, 1.0) if perp else
                              self.dSdu_v2(u, 1.0)) for u in np.linspace(0.0, 1.0, n_points)])
        elif edge == SurfaceEdge.South:
            return np.array([(self.dSdv_v2(u, 0.0) if perp else
                              self.dSdu_v2(u, 0.0)) for u in np.linspace(0.0, 1.0, n_points)])
        elif edge == SurfaceEdge.East:
            return np.array([(self.dSdu_v2(1.0, v) if perp else
                              self.dSdv_v2(1.0, v)) for v in np.linspace(0.0, 1.0, n_points)])
        elif edge == SurfaceEdge.West:
            return np.array([(self.dSdu_v2(0.0, v) if perp else
                              self.dSdv_v2(0.0, v)) for v in np.linspace(0.0, 1.0, n_points)])
        else:
            raise ValueError(f"No edge called {edge}")

    def get_second_derivs_along_edge(self, edge: SurfaceEdge, n_points: int = 10, perp=True) -> np.ndarray:
        if edge == SurfaceEdge.North:
            return np.array([(self.d2Sdv2(u, 1.0) if perp else
                              self.d2Sdu2(u, 1.0)) for u in np.linspace(0.0, 1.0, n_points)])
        elif edge == SurfaceEdge.South:
            return np.array([(self.d2Sdv2(u, 0.0) if perp else
                              self.d2Sdu2(u, 0.0)) for u in np.linspace(0.0, 1.0, n_points)])
        elif edge == SurfaceEdge.East:
            return np.array([(self.d2Sdu2(1.0, v) if perp else
                              self.d2Sdv2(1.0, v)) for v in np.linspace(0.0, 1.0, n_points)])
        elif edge == SurfaceEdge.West:
            return np.array([(self.d2Sdu2(0.0, v) if perp else
                              self.d2Sdv2(0.0, v)) for v in np.linspace(0.0, 1.0, n_points)])
        else:
            raise ValueError(f"No edge called {edge}")
    
    def verify_g0(self, other: 'RationalBezierSurface', surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge,
                n_points: int = 10):
        """ Verifies that two RationalBezierSurfaces are G0 continuous along their shared edge"""
        self_edge = self.get_edge(surface_edge, n_points=n_points)
        other_edge = other.get_edge(other_surface_edge, n_points=n_points)
        assert np.array_equal(self_edge, other_edge)

    def verify_g1(self, other: "RationalBezierSurface", surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge,
                  n_points: int = 10):
        """
        Verifies that two RationalBezierSurfaces are G1 continuous along their shared edge
        """
        # Get the first derivatives at the boundary and perpendicular to the boundary for each surface,
        # evaluated at "n_points" locations along the boundary
        self_perp_edge_derivs = self.get_first_derivs_along_edge(surface_edge, n_points=n_points, perp=True)
        other_perp_edge_derivs = other.get_first_derivs_along_edge(other_surface_edge, n_points=n_points, perp=True)
        self_perp_edge_derivs[np.absolute(self_perp_edge_derivs)<1e-6]=0.0
        other_perp_edge_derivs[np.absolute(other_perp_edge_derivs)<1e-6]=0.0


        # Initialize an array of ratios of magnitude of the derivative values at each point for both sides
        # of the boundary
        magnitude_ratios = []

        # Loop over each pair of cross-derivatives evaluated along the boundary
        for point_idx, (self_perp_edge_deriv, other_perp_edge_deriv) in enumerate(zip(
                self_perp_edge_derivs, other_perp_edge_derivs)):

            # Ensure that each derivative vector has the same direction along the boundary for each surface
            try:
                assert np.allclose(
                    np.nan_to_num(self_perp_edge_deriv / np.linalg.norm(self_perp_edge_deriv)),
                    np.nan_to_num(other_perp_edge_deriv / np.linalg.norm(other_perp_edge_deriv))
                )
            except AssertionError:
                assert np.allclose(
                    np.nan_to_num(self_perp_edge_deriv / np.linalg.norm(self_perp_edge_deriv)),
                    np.nan_to_num(-other_perp_edge_deriv / np.linalg.norm(other_perp_edge_deriv))
                )


            # Compute the ratio of the magnitudes for each derivative vector along the boundary for each surface.
            # These will be compared at the end.
            #print(f"{self_perp_edge_deriv=},{other_perp_edge_deriv=}")
            np.seterr(divide='ignore', invalid='ignore')
            with np.errstate(divide="ignore"):
                magnitude_ratios.append(np.nan_to_num(self_perp_edge_deriv / other_perp_edge_deriv,nan=0))
                    
        #print("Rational",f"{magnitude_ratios=}")
        # Assert that the first derivatives along each boundary are proportional
        current_f = None
        for magnitude_ratio in magnitude_ratios:
            for dxdydz_ratio in magnitude_ratio:
                if np.any(np.isinf(dxdydz_ratio)) or np.any(np.isnan(dxdydz_ratio)) or np.any(dxdydz_ratio == 0.0):
                    continue
                if current_f is None:
                    current_f = dxdydz_ratio
                    continue
                assert np.all(np.isclose(dxdydz_ratio, current_f))

    def verify_g2(self, other: "RationalBezierSurface", surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge,
                  n_points: int = 10):
        """
        Verifies that two RationalBezierSurfaces are G2 continuous along their shared edge
        """
        # Get the first derivatives at the boundary and perpendicular to the boundary for each surface,
        # evaluated at "n_points" locations along the boundary
        self_perp_edge_derivs = self.get_second_derivs_along_edge(surface_edge, n_points=n_points, perp=True)
        other_perp_edge_derivs = other.get_second_derivs_along_edge(other_surface_edge, n_points=n_points, perp=True)
        self_perp_edge_derivs[np.absolute(self_perp_edge_derivs)<1e-6]=0.0
        other_perp_edge_derivs[np.absolute(other_perp_edge_derivs)<1e-6]=0.0
        
        ratios_other_self=other_perp_edge_derivs/self_perp_edge_derivs
        print(f"{ratios_other_self=}")
        #print(f"{self_perp_edge_derivs=},{other_perp_edge_derivs=}")
        # Initialize an array of ratios of magnitude of the derivative values at each point for both sides
        # of the boundary
        magnitude_ratios = []

        # Loop over each pair of cross-derivatives evaluated along the boundary
        for point_idx, (self_perp_edge_deriv, other_perp_edge_deriv) in enumerate(zip(
                self_perp_edge_derivs, other_perp_edge_derivs)):

            # Ensure that each derivative vector has the same direction along the boundary for each surface
            assert np.allclose(
                np.nan_to_num(self_perp_edge_deriv / np.linalg.norm(self_perp_edge_deriv)),
                np.nan_to_num(other_perp_edge_deriv / np.linalg.norm(other_perp_edge_deriv))
            )

            # Compute the ratio of the magnitudes for each derivative vector along the boundary for each surface.
            # These will be compared at the end.
            with np.errstate(divide="ignore"):
                magnitude_ratios.append(self_perp_edge_deriv / other_perp_edge_deriv)

        # Assert that the second derivatives along each boundary are proportional
        current_f = None
        for magnitude_ratio in magnitude_ratios:
            for dxdydz_ratio in magnitude_ratio:
                if np.any(np.isinf(dxdydz_ratio)) or np.any(np.isnan(dxdydz_ratio)) or np.any(dxdydz_ratio == 0.0):
                    continue
                if current_f is None:
                    current_f = dxdydz_ratio
                    continue
                assert np.all(np.isclose(dxdydz_ratio, current_f))

    def generate_control_point_net(self) -> (typing.List[Point3D], typing.List[Line3D]):

        control_points = self.get_control_point_array()
        points = []
        lines = []

        for i in range(self.Nu):
            for j in range(self.Nv):
                points.append(Point3D.from_array(control_points[i, j, :]))

        for i in range(self.Nu - 1):
            for j in range(self.Nv - 1):
                point_obj_1 = Point3D.from_array(control_points[i, j, :])
                point_obj_2 = Point3D.from_array(control_points[i + 1, j, :])
                point_obj_3 = Point3D.from_array(control_points[i, j + 1, :])

                line_1 = Line3D(p0=point_obj_1, p1=point_obj_2)
                line_2 = Line3D(p0=point_obj_1, p1=point_obj_3)
                lines.extend([line_1, line_2])

                if i < self.Nu - 2 and j < self.Nv - 2:
                    continue

                point_obj_4 = Point3D.from_array(control_points[i + 1, j + 1, :])
                line_3 = Line3D(p0=point_obj_3, p1=point_obj_4)
                line_4 = Line3D(p0=point_obj_2, p1=point_obj_4)
                lines.extend([line_3, line_4])

        return points, lines

    def plot_surface(self, plot: pv.Plotter, **mesh_kwargs):
        XYZ = self.evaluate(50, 50)
        grid = pv.StructuredGrid(XYZ[:, :, 0], XYZ[:, :, 1], XYZ[:, :, 2])
        plot.add_mesh(grid, **mesh_kwargs)
        return grid

    def plot_control_point_mesh_lines(self, plot: pv.Plotter, **line_kwargs):
        _, line_objs = self.generate_control_point_net()
        line_arr = np.array([[line_obj.p0.as_array(), line_obj.p1.as_array()] for line_obj in line_objs])
        line_arr = line_arr.reshape((len(line_objs) * 2, 3))
        plot.add_lines(line_arr, **line_kwargs)

    def plot_control_points(self,  plot: pv.Plotter, **point_kwargs):
        point_objs, _ = self.generate_control_point_net()
        point_arr = np.array([point_obj.as_array() for point_obj in point_objs])
        plot.add_points(point_arr, **point_kwargs)


class NURBSSurface(Surface):
    def __init__(self,
                 control_points: np.ndarray,
                 knots_u: np.ndarray,
                 knots_v: np.ndarray,
                 weights: np.ndarray,
                 degree_u: int, degree_v: int,
                 ):
        assert control_points.ndim == 3
        assert knots_u.ndim == 1
        assert knots_v.ndim == 1
        assert weights.ndim == 2
        assert len(knots_u) == control_points.shape[0] + degree_u + 1
        assert len(knots_v) == control_points.shape[1] + degree_v + 1
        assert control_points[:, :, 0].shape == weights.shape

        # Negative weight check
        for weight_row in weights:
            for weight in weight_row:
                if weight < 0:
                    raise NegativeWeightError("All weights must be non-negative")

        self.control_points = control_points
        self.knots_u = knots_u
        self.knots_v = knots_v
        self.weights = weights
        self.degree_u = degree_u
        self.degree_v = degree_v
        self.Nu, self.Nv = control_points.shape[0], control_points.shape[1]
        self.possible_spans_u, self.possible_span_indices_u = self._get_possible_spans(self.knots_u)
        self.possible_spans_v, self.possible_span_indices_v = self._get_possible_spans(self.knots_v)

    def to_iges(self, *args, **kwargs) -> astk.iges.entity.IGESEntity:
        return astk.iges.surfaces.RationalBSplineSurfaceIGES(
            control_points=self.control_points,
            knots_u=self.knots_u,
            knots_v=self.knots_v,
            weights=self.weights,
            degree_u=self.degree_u,
            degree_v=self.degree_v
        )

    @classmethod
    def from_bezier_revolve(cls, bezier: Bezier3D, axis: Line3D, start_angle: Angle, end_angle: Angle):

        def _determine_angle_distribution() -> typing.List[Angle]:
            angle_diff = end_angle.rad - start_angle.rad

            if angle_diff == 0.0:
                raise InvalidGeometryError("Starting and ending angles cannot be the same for a "
                                           "NURBSSurface from revolved Bezier curve")

            if angle_diff % (0.5 * np.pi) == 0.0:  # If angle difference is a multiple of 90 degrees
                N_angles = 2 * int(angle_diff // (0.5 * np.pi)) + 1
            else:
                N_angles = 2 * int(angle_diff // (0.5 * np.pi)) + 3

            rad_dist = np.linspace(start_angle.rad, end_angle.rad, N_angles)
            return [Angle(rad=r) for r in rad_dist]

        control_points = []
        weights = []
        angles = _determine_angle_distribution()

        for point in bezier.points:

            axis_projection = project_point_onto_line(point, axis)
            radius = measure_distance_point_line(point, axis)
            if radius == 0.0:
                new_points = [point for _ in angles]
            else:
                new_points = [rotate_point_about_axis(point, axis, angle) for angle in angles]

            for idx, rotated_point in enumerate(new_points):
                if idx == 0:
                    weights.append([])
                if not idx % 2:  # Skip even indices (these represent the "through" control points)
                    weights[-1].append(1.0)
                    continue
                sine_half_angle = np.sin(0.5 * np.pi - 0.5 * (angles[idx + 1].rad - angles[idx - 1].rad))

                if radius != 0.0:
                    distance = radius / sine_half_angle  # radius / sin(half angle)
                    vector = Vector3D(p0=axis_projection, p1=rotated_point)
                    new_points[idx] = axis_projection + Point3D.from_array(
                        distance * np.array(vector.normalized_value()))

                weights[-1].append(sine_half_angle)

            control_points.append(np.array([new_point.as_array() for new_point in new_points]))

        control_points = np.array(control_points)
        weights = np.array(weights)

        knots_v = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        n_knots_to_insert = len(angles) - 3
        if n_knots_to_insert > 0:
            delta = 1.0 / (n_knots_to_insert / 2 + 1)
            for idx in range(n_knots_to_insert):
                new_knot = knots_v[2 + idx] if idx % 2 else knots_v[2 + idx] + delta
                knots_v = np.insert(knots_v, 3 + idx, new_knot)

        knots_u = np.array([0.0 for _ in bezier.points] + [1.0 for _ in bezier.points])
        degree_v = 2
        degree_u = len(bezier.points) - 1

        return cls(control_points, knots_u, knots_v, weights, degree_u, degree_v)

    @staticmethod
    def _get_possible_spans(knot_vector) -> (np.ndarray, np.ndarray):
        possible_span_indices = np.array([], dtype=int)
        possible_spans = []
        for knot_idx, (knot_1, knot_2) in enumerate(zip(knot_vector[:-1], knot_vector[1:])):
            if knot_1 == knot_2:
                continue
            possible_span_indices = np.append(possible_span_indices, knot_idx)
            possible_spans.append([knot_1, knot_2])
        return np.array(possible_spans), possible_span_indices

    def _cox_de_boor(self, t: float, i: int, p: int, knot_vector: np.ndarray,
                     possible_spans_u_or_v: np.ndarray, possible_span_indices_u_or_v: np.ndarray) -> float:
        if p == 0:
            return 1.0 if i in possible_span_indices_u_or_v and self._find_span(t, possible_spans_u_or_v, possible_span_indices_u_or_v) == i else 0.0
        else:
            with (np.errstate(divide="ignore", invalid="ignore")):
                f = (t - knot_vector[i]) / (knot_vector[i + p] - knot_vector[i])
                g = (knot_vector[i + p + 1] - t) / (knot_vector[i + p + 1] - knot_vector[i + 1])
                if np.isinf(f) or np.isnan(f):
                    f = 0.0
                if np.isinf(g) or np.isnan(g):
                    g = 0.0
                if f == 0.0 and g == 0.0:
                    return 0.0
                elif f != 0.0 and g == 0.0:
                    return f * self._cox_de_boor(t, i, p - 1, knot_vector,
                                                 possible_spans_u_or_v, possible_span_indices_u_or_v)
                elif f == 0.0 and g != 0.0:
                    return g * self._cox_de_boor(t, i + 1, p - 1, knot_vector,
                                                 possible_spans_u_or_v, possible_span_indices_u_or_v)
                else:
                    return f * self._cox_de_boor(t, i, p - 1, knot_vector,
                                                 possible_spans_u_or_v, possible_span_indices_u_or_v) + \
                    g * self._cox_de_boor(t, i + 1, p - 1, knot_vector,
                                          possible_spans_u_or_v, possible_span_indices_u_or_v)

    def _basis_functions(self, t: float, p: int, knot_vector: np.ndarray, n_control_points_u_or_v: int,
                         possible_spans_u_or_v: np.ndarray, possible_span_indices_u_or_v: np.ndarray):
        """
        Compute the non-zero basis functions at parameter t
        """
        return np.array([self._cox_de_boor(t, i, p, knot_vector, possible_spans_u_or_v, possible_span_indices_u_or_v) for i in range(n_control_points_u_or_v)])

    @staticmethod
    def _find_span(t: float, possible_spans_u_or_v: np.ndarray, possible_span_indices_u_or_v: np.ndarray):
        """
        Find the knot span index
        """
        # insertion_point = bisect.bisect_left(self.non_repeated_knots, t)
        # return self.possible_spans[insertion_point - 1]
        for knot_span, knot_span_idx in zip(possible_spans_u_or_v, possible_span_indices_u_or_v):
            if knot_span[0] <= t < knot_span[1]:
                return knot_span_idx
        if t == possible_spans_u_or_v[-1][1]:
            return possible_span_indices_u_or_v[-1]

    def evaluate_ndarray(self, u: float, v: float) -> np.ndarray:
        Bu = self._basis_functions(u, self.degree_u, self.knots_u, self.Nu,
                                   self.possible_spans_u, self.possible_span_indices_u)
        Bv = self._basis_functions(v, self.degree_v, self.knots_v, self.Nv,
                                   self.possible_spans_v, self.possible_span_indices_v)

        weighted_cps = np.zeros(self.control_points.shape[2])
        denominator = 0.0

        for i in range(self.Nu):
            for j in range(self.Nv):
                weighted_cps += self.control_points[i][j] * Bu[i] * Bv[j] * self.weights[i][j]
                denominator += Bu[i] * Bv[j] * self.weights[i][j]

        return weighted_cps / denominator

    def evaluate_simple(self, u: float, v: float) -> Point3D:
        return Point3D.from_array(self.evaluate_ndarray(u, v))

    def evaluate(self, Nu: int, Nv: int) -> np.ndarray:
        U, V = np.meshgrid(np.linspace(0.0, 1.0, Nu), np.linspace(0.0, 1.0, Nv))
        return np.array([[self.evaluate_ndarray(U[i, j], V[i, j]) for j in range(U.shape[1])] for i in range(U.shape[0])])

    def generate_control_point_net(self) -> (typing.List[Point3D], typing.List[Line3D]):

        points = []
        lines = []

        for i in range(self.Nu):
            for j in range(self.Nv):
                points.append(Point3D.from_array(self.control_points[i, j, :]))

        for i in range(self.Nu - 1):
            for j in range(self.Nv - 1):
                point_obj_1 = Point3D.from_array(self.control_points[i, j, :])
                point_obj_2 = Point3D.from_array(self.control_points[i + 1, j, :])
                point_obj_3 = Point3D.from_array(self.control_points[i, j + 1, :])

                line_1 = Line3D(p0=point_obj_1, p1=point_obj_2)
                line_2 = Line3D(p0=point_obj_1, p1=point_obj_3)
                lines.extend([line_1, line_2])

                if i < self.Nu - 2 and j < self.Nv - 2:
                    continue

                point_obj_4 = Point3D.from_array(self.control_points[i + 1, j + 1, :])
                line_3 = Line3D(p0=point_obj_3, p1=point_obj_4)
                line_4 = Line3D(p0=point_obj_2, p1=point_obj_4)
                lines.extend([line_3, line_4])

        return points, lines

    def plot_surface(self, plot: pv.Plotter, **mesh_kwargs):
        XYZ = self.evaluate(50, 50)
        grid = pv.StructuredGrid(XYZ[:, :, 0], XYZ[:, :, 1], XYZ[:, :, 2])
        plot.add_mesh(grid, **mesh_kwargs)
        return grid

    def plot_control_point_mesh_lines(self, plot: pv.Plotter, **line_kwargs):
        _, line_objs = self.generate_control_point_net()
        line_arr = np.array([[line_obj.p0.as_array(), line_obj.p1.as_array()] for line_obj in line_objs])
        line_arr = line_arr.reshape((len(line_objs) * 2, 3))
        plot.add_lines(line_arr, **line_kwargs)

    def plot_control_points(self,  plot: pv.Plotter, **point_kwargs):
        point_objs, _ = self.generate_control_point_net()
        point_arr = np.array([point_obj.as_array() for point_obj in point_objs])
        plot.add_points(point_arr, **point_kwargs)
