import typing
from copy import deepcopy
from enum import Enum

import numpy as np
import pyvista as pv
from scipy.optimize import fsolve
import shapely

import aerocaps.iges.entity
import aerocaps.iges.curves
import aerocaps.iges.surfaces
from aerocaps.geom import Surface, InvalidGeometryError, NegativeWeightError, Geometry3D
from aerocaps.geom.point import Point3D
from aerocaps.geom.curves import Bezier3D, Line3D, RationalBezierCurve3D, NURBSCurve3D, BSpline3D
import aerocaps
from aerocaps.geom.tools import project_point_onto_line, measure_distance_point_line, rotate_point_about_axis
from aerocaps.geom.vector import Vector3D
from aerocaps.units.angle import Angle
from aerocaps.units.length import Length
from aerocaps.utils.math import bernstein_poly

__all__ = [
    "SurfaceEdge",
    "SurfaceCorner",
    "BezierSurface",
    "RationalBezierSurface",
    "NURBSSurface",
    "PlanarFillSurfaceCreator",
    "TrimmedSurface"
]


class SurfaceEdge(Enum):
    """
    Enum describing the name of each edge of a four-sided surface. The names are defined by the name and value of the
    parameter that is constant along the edge.

    .. figure:: ../images/cardinal_transparent.*
        :width: 300
        :align: center

        Surface edge nomenclature
    """
    v1 = 0
    v0 = 1
    u1 = 2
    u0 = 3


class SurfaceCorner(Enum):
    u1v1 = 0
    u0v1 = 1
    u0v0 = 2
    u1v0 = 3


class BezierSurface(Surface):
    def __init__(self, points: typing.List[typing.List[Point3D]] or np.ndarray):
        r"""
        A Bézier surface is a parametric surface described by a matrix of control points and defined on a rectangular
        domain :math:`\{u \in [0,1], v \in [0,1]\}`. The mathematical expression for the Bézier surface is identical
        to that of the Bézier curve except with an extra dimension:

        .. math::

            \mathbf{S}(u,v) = \sum\limits_{i=0}^n \sum\limits_{j=0}^m B_{i,n}(u) B_{j,m}(v) \mathbf{P}_{i,j}

        Where :math:`B_{i,n}(t)` is the Bernstein polynomial given by

        .. math::

            B_{i,n}(t) = {n \choose i} t^i (1-t)^{n-i}

        An example of a Bézier surface with :math:`n=2` and :math:`m=3` is shown below. Note that the only control
        points that lie directly on the surface are the corner points of the control point mesh. This is analogous
        to the fact that only the starting and ending control points of Bézier curves lie directly on the curve.
        In fact, Bézier curves derived from the bounding rows and columns of control points exactly represent the
        boundary curves of the surface. In this example, the control points given by :math:`\mathbf{P}_{i,j=0}` and
        :math:`\mathbf{P}_{i,j=m}` represent quadratic Bézier curves (:math:`n=2`), and the control points given by
        :math:`\mathbf{P}_{i=0,j}` and :math:`\mathbf{P}_{i=n,j}` represent cubic Bézier curves (:math:`m=3`).

        .. figure:: ../images/bezier_surf_2x3.*
            :width: 600
            :align: center

            A :math:`2 \times 3` Bézier surface with control points and control point net lines shown

        .. figure:: ../images/bezier_surf_2x3_mesh_only.*
            :width: 600
            :align: center

            A :math:`2 \times 3` Bézier surface with isoparametric curves in both :math:`u` and :math:`v` shown

        Bézier surfaces can be constructed either via the default constructor with a nested list of
        ``aerocaps.geom.point.Point3D`` objects of by means of the ``generate_from_array`` class method where only a
        3-D ``numpy`` array is required. For example, say we have six ``Point3D`` objects, A-F and would like to use
        them to create a :math:`2 \times 1` Bézier surface.
        Using the default constructor,

        .. code-block:: python

            surf = BezierSurface([[pA, pB], [pC, pD], [pE, pF]])

        Using the array class method and point :math:`xyz` float values given by ``pA_x``, ``pA_y``, ``pA_z``, etc.,

        .. code-block:: python

            control_points = np.array([
                [[pA_x, pA_y, pA_z], [pB_x, pB_y, pB_z]],
                [[pC_x, pC_y, pC_z], [pD_x, pD_y, pD_z]],
                [[pE_x, pE_y, pE_z], [pF_x, pF_y, pF_z]],
            ])

            surf = BezierSurface.generate_from_array(control_points)
        """
        if isinstance(points, np.ndarray):
            points = [[Point3D.from_array(pt_row) for pt_row in pt_mat] for pt_mat in points]
        self.points = points
        self.degree_u = len(points) - 1
        self.degree_v = len(points[0]) - 1
        self.Nu = self.degree_u + 1
        self.Nv = self.degree_v + 1

    def to_iges(self, *args, **kwargs) -> aerocaps.iges.entity.IGESEntity:
        """
        Converts the Bézier surface to an IGES entity. To add this IGES entity to an ``.igs`` file,
        use an :obj:`~aerocaps.iges.iges_generator.IGESGenerator`.
        """
        return aerocaps.iges.surfaces.BezierSurfaceIGES(self.get_control_point_array())

    def get_control_point_array(self) -> np.ndarray:
        """
        Converts the nested list of control points to a 3-D :obj:`~numpy.ndarray`.

        Returns
        -------
        numpy.ndarray
            3-D array
        """
        return np.array([np.array([p.as_array() for p in p_arr]) for p_arr in self.points])

    @classmethod
    def generate_from_array(cls, P: np.ndarray):
        r"""
        Creates a new Bézier surface from a 3-D :obj:`~numpy.ndarray`.

        Parameters
        ----------
        P: numpy.ndarray
            Array of control points of size :math:`(n+1) \times (m+1) \times 3`, where :math:`n` is the surface
            degree in the :math:`u`-parametric direction and :math:`m` is the surface degree in the
            :math:`v`-parametric direction

        Returns
        -------
        BezierSurface
            New Bézier surface created from the input control points
        """
        return cls([
            [Point3D(x=Length(m=xyz[0]), y=Length(m=xyz[1]), z=Length(m=xyz[2])) for xyz in point_arr]
            for point_arr in P])

    def evaluate_ndarray(self, u: float, v: float):
        r"""
        Evaluates the surface at a given :math:`(u,v)` parameter pair.

        Parameters
        ----------
        u: float
            Position along :math:`u` in parametric space. Normally in the range :math:`[0,1]`
        v: float
            Position along :math:`v` in parametric space. Normally in the range :math:`[0,1]`

        Returns
        -------
        numpy.ndarray
            1-D array of the form ``array([x, y, z])`` representing the evaluated point on the surface
        """
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
        r"""
        Evaluates the first derivative of the surface in the :math:`u`-direction,
        :math:`\frac{\partial \mathbf{S}(u,v)}{\partial u}`, at a given :math:`(u,v)` parameter pair.

        Parameters
        ----------
        u: float
            Position along :math:`u` in parametric space. Normally in the range :math:`[0,1]`
        v: float
            Position along :math:`v` in parametric space. Normally in the range :math:`[0,1]`

        Returns
        -------
        numpy.ndarray
            1-D array of the form ``array([dSdu_x, dSdu_y, dSdu_z])`` representing the :math:`x`-, :math:`y`-,
            and :math:`z`-components of :math:`\frac{\partial \mathbf{S}(u,v)}{\partial u}`
        """
        P = self.get_control_point_array()
        deriv_u = np.zeros(P.shape[2])
        for i in range(self.degree_u + 1):
            for j in range(self.degree_v + 1):
                dbudu = self.degree_u * (bernstein_poly(self.degree_u - 1, i - 1, u) - bernstein_poly(
                    self.degree_u - 1, i, u))
                bv = bernstein_poly(self.degree_v, j, v)
                dbudu_bv = dbudu * bv
                deriv_u += P[i, j, :] * dbudu_bv
        return deriv_u

    def dSdv(self, u: float, v: float):
        r"""
        Evaluates the first derivative of the surface in the :math:`v`-direction,
        :math:`\frac{\partial \mathbf{S}(u,v)}{\partial u}`, at a given :math:`(u,v)` parameter pair.

        Parameters
        ----------
        u: float
            Position along :math:`u` in parametric space. Normally in the range :math:`[0,1]`
        v: float
            Position along :math:`v` in parametric space. Normally in the range :math:`[0,1]`

        Returns
        -------
        numpy.ndarray
            1-D array of the form ``array([dSdv_x, dSdv_y, dSdv_z])`` representing the :math:`x`-, :math:`y`-,
            and :math:`z`-components of :math:`\frac{\partial \mathbf{S}(u,v)}{\partial v}`
        """
        P = self.get_control_point_array()
        deriv_v = np.zeros(P.shape[2])
        for i in range(self.degree_u + 1):
            for j in range(self.degree_v + 1):
                dbvdv = self.degree_v * (bernstein_poly(self.degree_v - 1, j - 1, v) - bernstein_poly(
                    self.degree_v - 1, j, v))
                bu = bernstein_poly(self.degree_u, i, u)
                bu_dbvdv = bu * dbvdv
                deriv_v += P[i, j, :] * bu_dbvdv
        return deriv_v

    def d2Sdu2(self, u: float, v: float):
        r"""
        Evaluates the second pure derivative of the surface in the :math:`u`-direction,
        :math:`\frac{\partial^2 \mathbf{S}(u,v)}{\partial u^2}`, at a given :math:`(u,v)` parameter pair.

        Parameters
        ----------
        u: float
            Position along :math:`u` in parametric space. Normally in the range :math:`[0,1]`
        v: float
            Position along :math:`v` in parametric space. Normally in the range :math:`[0,1]`

        Returns
        -------
        numpy.ndarray
            1-D array of the form ``array([dS2du2_x, dS2du2_y, dS2du2_z])`` representing the :math:`x`-, :math:`y`-,
            and :math:`z`-components of :math:`\frac{\partial^2 \mathbf{S}(u,v)}{\partial u^2}`
        """
        P = self.get_control_point_array()
        deriv_u_2 = np.zeros(P.shape[2])
        for i in range(self.degree_u + 1):
            for j in range(self.degree_v + 1):
                term_1 = self.degree_u * (self.degree_u - 1) * (
                        bernstein_poly(self.degree_u - 2, i - 2, u) - bernstein_poly(self.degree_u - 2, i - 1, u))
                term_2 = self.degree_u * (self.degree_u - 1) * (
                        bernstein_poly(self.degree_u - 2, i - 1, u) - bernstein_poly(self.degree_u - 2, i, u))
                d2budu2 = term_1 - term_2
                Bv = bernstein_poly(self.degree_v, j, v)
                d2budu2_Bv = d2budu2 * Bv
                deriv_u_2 += P[i, j, :] * d2budu2_Bv
        return deriv_u_2

    def d2Sdv2(self, u: float, v: float):
        r"""
        Evaluates the second pure derivative of the surface in the :math:`v`-direction,
        :math:`\frac{\partial^2 \mathbf{S}(u,v)}{\partial v^2}`, at a given :math:`(u,v)` parameter pair.

        Parameters
        ----------
        u: float
            Position along :math:`u` in parametric space. Normally in the range :math:`[0,1]`
        v: float
            Position along :math:`v` in parametric space. Normally in the range :math:`[0,1]`

        Returns
        -------
        numpy.ndarray
            1-D array of the form ``array([dS2dv2_x, dS2dv2_y, dS2dv2_z])`` representing the :math:`x`-, :math:`y`-,
            and :math:`z`-components of :math:`\frac{\partial^2 \mathbf{S}(u,v)}{\partial v^2}`
        """
        P = self.get_control_point_array()
        deriv_v_2 = np.zeros(P.shape[2])
        for i in range(self.degree_u + 1):
            for j in range(self.degree_v + 1):
                term1 = self.degree_v * (self.degree_v - 1) * (
                            bernstein_poly(self.degree_v - 2, j - 2, v) - bernstein_poly(self.degree_v - 2, j - 1, v))
                term2 = self.degree_v * (self.degree_v - 1) * (
                            bernstein_poly(self.degree_v - 2, j - 1, v) - bernstein_poly(self.degree_v - 2, j, v))
                d2bvdv2 = term1 - term2
                Bu = bernstein_poly(self.degree_u, i, u)
                Bu_d2bvdv2 = Bu * d2bvdv2
                deriv_v_2 += P[i, j, :] * Bu_d2bvdv2
        return deriv_v_2

    def get_edge(self, edge: SurfaceEdge, n_points: int = 10) -> np.ndarray:
        r"""
        Evaluates the surface at ``n_points`` parameter locations along a given edge.

        Parameters
        ----------
        edge: SurfaceEdge
            Edge along which to evaluate
        n_points: int
            Number of evenly-spaced parameter locations at which to evaluate the edge curve

        Returns
        -------
        numpy.ndarray
            2-D array of size :math:`n_\text{points} \times 3`
        """
        if edge == SurfaceEdge.v1:
            return np.array([self.evaluate_ndarray(u, 1) for u in np.linspace(0.0, 1.0, n_points)])
        elif edge == SurfaceEdge.v0:
            return np.array([self.evaluate_ndarray(u, 0) for u in np.linspace(0.0, 1.0, n_points)])
        elif edge == SurfaceEdge.u1:
            return np.array([self.evaluate_ndarray(1, v) for v in np.linspace(0.0, 1.0, n_points)])
        elif edge == SurfaceEdge.u0:
            return np.array([self.evaluate_ndarray(0, v) for v in np.linspace(0.0, 1.0, n_points)])
        else:
            raise ValueError(f"No edge called {edge}")

    def get_first_derivs_along_edge(self, edge: SurfaceEdge, n_points: int = 10, perp: bool = True) -> np.ndarray:
        if edge == SurfaceEdge.v1:
            return np.array([(self.dSdv(u, 1.0) if perp else
                              self.dSdu(u, 1.0)) for u in np.linspace(0.0, 1.0, n_points)])
        elif edge == SurfaceEdge.v0:
            return np.array([(self.dSdv(u, 0.0) if perp else
                              self.dSdu(u, 0.0)) for u in np.linspace(0.0, 1.0, n_points)])
        elif edge == SurfaceEdge.u1:
            return np.array([(self.dSdu(1.0, v) if perp else
                              self.dSdv(1.0, v)) for v in np.linspace(0.0, 1.0, n_points)])
        elif edge == SurfaceEdge.u0:
            return np.array([(self.dSdu(0.0, v) if perp else
                              self.dSdv(0.0, v)) for v in np.linspace(0.0, 1.0, n_points)])
        else:
            raise ValueError(f"No edge called {edge}")

    def get_second_derivs_along_edge(self, edge: SurfaceEdge, n_points: int = 10, perp: bool = True) -> np.ndarray:
        if edge == SurfaceEdge.v1:
            return np.array([(self.d2Sdv2(u, 1.0) if perp else
                              self.d2Sdu2(u, 1.0)) for u in np.linspace(0.0, 1.0, n_points)])
        elif edge == SurfaceEdge.v0:
            return np.array([(self.d2Sdv2(u, 0.0) if perp else
                              self.d2Sdu2(u, 0.0)) for u in np.linspace(0.0, 1.0, n_points)])
        elif edge == SurfaceEdge.u1:
            return np.array([(self.d2Sdu2(1.0, v) if perp else
                              self.d2Sdv2(1.0, v)) for v in np.linspace(0.0, 1.0, n_points)])
        elif edge == SurfaceEdge.u0:
            return np.array([(self.d2Sdu2(0.0, v) if perp else
                              self.d2Sdv2(0.0, v)) for v in np.linspace(0.0, 1.0, n_points)])
        else:
            raise ValueError(f"No edge called {edge}")

    def verify_g0(self, other: "BezierSurface", surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge,
                  n_points: int = 10):
        r"""
        Verifies that two Bézier surfaces are :math:`G^0`-continuous along their shared edge
        """
        self_edge = self.get_edge(surface_edge, n_points=n_points)
        other_edge = other.get_edge(other_surface_edge, n_points=n_points)
        assert np.array_equal(self_edge, other_edge)

    def verify_g1(self, other: "BezierSurface", surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge,
                  n_points: int = 10):
        r"""
        Verifies that two Bézier surfaces are :math:`G^1`-continuous along their shared edge
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
        Verifies that two Bézier surfaces are :math:`G^2`-continuous along their shared edge
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

    def extract_edge_curve(self, surface_edge: SurfaceEdge) -> Bezier3D:
        P = self.get_control_point_array()

        if surface_edge == SurfaceEdge.u0:
            return Bezier3D.generate_from_array(P[0, :, :])
        if surface_edge == SurfaceEdge.u1:
            return Bezier3D.generate_from_array(P[-1, :, :])
        if surface_edge == SurfaceEdge.v0:
            return Bezier3D.generate_from_array(P[:, 0, :])
        if surface_edge == SurfaceEdge.v1:
            return Bezier3D.generate_from_array(P[:, -1, :])

        raise ValueError(f"Invalid surface edge {surface_edge}")

    def extract_isoparametric_curve_u(self, Nu: int, v: float):
        u_vec = np.linspace(0.0, 1.0, Nu)
        return np.array([self.evaluate_ndarray(u, v) for u in u_vec])

    def extract_isoparametric_curve_v(self, Nv: int, u: float):
        v_vec = np.linspace(0.0, 1.0, Nv)
        return np.array([self.evaluate_ndarray(u, v) for v in v_vec])

    def get_parallel_degree(self, surface_edge: SurfaceEdge):
        if surface_edge in [SurfaceEdge.v1, SurfaceEdge.v0]:
            return self.degree_u
        return self.degree_v

    def get_perpendicular_degree(self, surface_edge: SurfaceEdge):
        if surface_edge in [SurfaceEdge.v1, SurfaceEdge.v0]:
            return self.degree_v
        return self.degree_u

    def get_point(self, row_index: int, continuity_index: int, surface_edge: SurfaceEdge):
        if surface_edge == SurfaceEdge.v1:
            return self.points[row_index][-(continuity_index + 1)]
        elif surface_edge == SurfaceEdge.v0:
            return self.points[row_index][continuity_index]
        elif surface_edge == SurfaceEdge.u1:
            return self.points[-(continuity_index + 1)][row_index]
        elif surface_edge == SurfaceEdge.u0:
            return self.points[continuity_index][row_index]
        else:
            raise ValueError("Invalid surface_edge value")

    def set_point(self, point: Point3D, row_index: int, continuity_index: int, surface_edge: SurfaceEdge):
        if surface_edge == SurfaceEdge.v1:
            self.points[row_index][-(continuity_index + 1)] = point
        elif surface_edge == SurfaceEdge.v0:
            self.points[row_index][continuity_index] = point
        elif surface_edge == SurfaceEdge.u1:
            self.points[-(continuity_index + 1)][row_index] = point
        elif surface_edge == SurfaceEdge.u0:
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
        n_ratio = (p_perp_a ** 2 - p_perp_a) / (p_perp_b ** 2 - p_perp_b)
        for row_index in range(self.get_parallel_degree(surface_edge) + 1):
            P_i0_b = self.get_point(row_index, 0, surface_edge)
            P_i1_b = self.get_point(row_index, 1, surface_edge)
            P_im_a = other.get_point(row_index, 0, other_surface_edge)
            P_im1_a = other.get_point(row_index, 1, other_surface_edge)
            P_im2_a = other.get_point(row_index, 2, other_surface_edge)

            P_i2_b = (2.0 * P_i1_b - P_i0_b) + f ** 2 * n_ratio * (P_im_a - 2.0 * P_im1_a + P_im2_a)
            self.set_point(P_i2_b, row_index, 2, surface_edge)

    def enforce_c0c1c2(self, other: "BezierSurface",

                       surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge):
        self.enforce_g0g1g2(other, 1.0, surface_edge, other_surface_edge)

    def split_at_u(self, u0: float) -> ("BezierSurface", "BezierSurface"):
        """
        Splits the Bezier surface at :math:`u=u_0` along the :math:`v`-parametric direction.
        """
        P = self.get_control_point_array()

        def de_casteljau(i: int, j: int, k: int) -> np.ndarray:
            """
            Based on https://en.wikipedia.org/wiki/De_Casteljau%27s_algorithm. Recursive algorithm where the
            base case is just the value of the ith original control point.

            Parameters
            ----------
            i: int
                Lower index
            j: int
                Upper index
            k: int
                Control point row index

            Returns
            -------
            np.ndarray
                A one-dimensional array containing the :math:`x` and :math:`y` values of a control point evaluated
                at :math:`(i,j)` for a Bézier curve split at the parameter value ``t_split``
            """
            if j == 0:
                return P[i, k, :]
            return de_casteljau(i, j - 1, k) * (1 - u0) + de_casteljau(i + 1, j - 1, k) * u0

        bez_surf_split_1_P = np.array([
            [de_casteljau(i=0, j=i, k=k) for i in range(self.Nu)] for k in range(self.Nv)
        ])
        bez_surf_split_2_P = np.array([
            [de_casteljau(i=i, j=self.degree_u - i, k=k) for i in range(self.Nu)] for k in range(self.Nv)
        ])

        return (
            BezierSurface.generate_from_array(
                np.transpose(
                    bez_surf_split_1_P, (1, 0, 2)
                )
            ),
            BezierSurface.generate_from_array(
                np.transpose(
                    bez_surf_split_2_P, (1, 0, 2)
                )
            )
        )

    def split_at_v(self, v0: float) -> ("BezierSurface", "BezierSurface"):
        """
        Splits the Bezier surface at :math:`v=v_0` along the :math:`u`-parametric direction.
        """
        P = self.get_control_point_array()

        def de_casteljau(i: int, j: int, k: int) -> np.ndarray:
            """
            Based on https://en.wikipedia.org/wiki/De_Casteljau%27s_algorithm. Recursive algorithm where the
            base case is just the value of the ith original control point.

            Parameters
            ----------
            i: int
                Lower index
            j: int
                Upper index
            k: int
                Control point row index

            Returns
            -------
            np.ndarray
                A one-dimensional array containing the :math:`x` and :math:`y` values of a control point evaluated
                at :math:`(i,j)` for a Bézier curve split at the parameter value ``t_split``
            """
            if j == 0:
                return P[k, i, :]
            return de_casteljau(i, j - 1, k) * (1 - v0) + de_casteljau(i + 1, j - 1, k) * v0

        bez_surf_split_1_P = np.array([
            [de_casteljau(i=0, j=i, k=k) for i in range(self.Nv)] for k in range(self.Nu)
        ])
        bez_surf_split_2_P = np.array([
            [de_casteljau(i=i, j=self.degree_v - i, k=k) for i in range(self.Nv)] for k in range(self.Nu)
        ])

        return (
            BezierSurface.generate_from_array(bez_surf_split_1_P),
            BezierSurface.generate_from_array(bez_surf_split_2_P)
        )

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

    def plot_control_points(self, plot: pv.Plotter, **point_kwargs):
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

    def to_iges(self, *args, **kwargs) -> aerocaps.iges.entity.IGESEntity:
        return aerocaps.iges.surfaces.RationalBSplineSurfaceIGES(
            control_points=self.get_control_point_array(),
            knots_u=self.knots_u,
            knots_v=self.knots_v,
            weights=self.weights,
            degree_u=self.degree_u,
            degree_v=self.degree_v
        )

    def get_control_point_array(self) -> np.ndarray:
        return np.array([np.array([p.as_array() for p in p_arr]) for p_arr in self.points])

    def get_xyzw_array(self) -> np.ndarray:
        return np.array([np.array([(p.as_array() * w).tolist() + [w] for p, w in zip(p_arr, w_arr)])
                         for p_arr, w_arr in zip(self.points, self.weights)])

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

        for point in bezier.control_points:

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

    def extract_edge_curve(self, surface_edge: SurfaceEdge) -> RationalBezierCurve3D:
        P = self.get_control_point_array()
        w = self.weights

        if surface_edge == SurfaceEdge.u0:
            return RationalBezierCurve3D.generate_from_array(P[0, :, :], w[0, :])
        if surface_edge == SurfaceEdge.u1:
            return RationalBezierCurve3D.generate_from_array(P[-1, :, :], w[-1, :])
        if surface_edge == SurfaceEdge.v0:
            return RationalBezierCurve3D.generate_from_array(P[:, 0, :], w[:, 0])
        if surface_edge == SurfaceEdge.v1:
            return RationalBezierCurve3D.generate_from_array(P[:, -1, :], w[:, -1])

        raise ValueError(f"Invalid surface edge {surface_edge}")

    def get_parallel_degree(self, surface_edge: SurfaceEdge):
        if surface_edge in [SurfaceEdge.v1, SurfaceEdge.v0]:
            return self.degree_u
        return self.degree_v

    def get_perpendicular_degree(self, surface_edge: SurfaceEdge):
        if surface_edge in [SurfaceEdge.v1, SurfaceEdge.v0]:
            return self.degree_v
        return self.degree_u

    def get_corner_index(self, surface_corner: SurfaceCorner) -> (int, int):
        if surface_corner == SurfaceCorner.u1v1:
            return self.degree_u, self.degree_v
        elif surface_corner == SurfaceCorner.u0v1:
            return 0, self.degree_v
        elif surface_corner == SurfaceCorner.u0v0:
            return 0, 0
        elif surface_corner == SurfaceCorner.u1v0:
            return self.degree_u, 1
        else:
            raise ValueError("Invalid surface_corner value")

    def get_point(self, row_index: int, continuity_index: int, surface_edge: SurfaceEdge):
        if surface_edge == SurfaceEdge.v1:
            return self.points[row_index][-(continuity_index + 1)]
        elif surface_edge == SurfaceEdge.v0:
            return self.points[row_index][continuity_index]
        elif surface_edge == SurfaceEdge.u1:
            return self.points[-(continuity_index + 1)][row_index]
        elif surface_edge == SurfaceEdge.u0:
            return self.points[continuity_index][row_index]
        else:
            raise ValueError("Invalid surface_edge value")

    def set_point(self, point: Point3D, row_index: int, continuity_index: int, surface_edge: SurfaceEdge):
        if surface_edge == SurfaceEdge.v1:
            self.points[row_index][-(continuity_index + 1)] = point
        elif surface_edge == SurfaceEdge.v0:
            self.points[row_index][continuity_index] = point
        elif surface_edge == SurfaceEdge.u1:
            self.points[-(continuity_index + 1)][row_index] = point
        elif surface_edge == SurfaceEdge.u0:
            self.points[continuity_index][row_index] = point
        else:
            raise ValueError("Invalid surface_edge value")

    def get_weight(self, row_index: int, continuity_index: int, surface_edge: SurfaceEdge):
        if surface_edge == SurfaceEdge.v1:
            return self.weights[row_index][-(continuity_index + 1)]
        elif surface_edge == SurfaceEdge.v0:
            return self.weights[row_index][continuity_index]
        elif surface_edge == SurfaceEdge.u1:
            return self.weights[-(continuity_index + 1)][row_index]
        elif surface_edge == SurfaceEdge.u0:
            return self.weights[continuity_index][row_index]
        else:
            raise ValueError("Invalid surface_edge value")

    def set_weight(self, weight: float, row_index: int, continuity_index: int, surface_edge: SurfaceEdge):
        if surface_edge == SurfaceEdge.v1:
            self.weights[row_index][-(continuity_index + 1)] = weight
        elif surface_edge == SurfaceEdge.v0:
            self.weights[row_index][continuity_index] = weight
        elif surface_edge == SurfaceEdge.u1:
            self.weights[-(continuity_index + 1)][row_index] = weight
        elif surface_edge == SurfaceEdge.u0:
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
        n_ratio = (other.get_perpendicular_degree(other_surface_edge) ** 2 -
                   other.get_perpendicular_degree(other_surface_edge)) / (
                          self.get_perpendicular_degree(surface_edge) ** 2 - self.get_perpendicular_degree(
                      surface_edge))
        for row_index in range(self.get_parallel_degree(surface_edge) + 1):

            w_i0_b = self.get_weight(row_index, 0, surface_edge)
            w_i1_b = self.get_weight(row_index, 1, surface_edge)
            w_im_a = other.get_weight(row_index, 0, other_surface_edge)
            w_im1_a = other.get_weight(row_index, 1, other_surface_edge)
            w_im2_a = other.get_weight(row_index, 2, other_surface_edge)

            f_row = f if isinstance(f, float) else f[row_index]

            w_i2_b = 2.0 * w_i1_b - w_i0_b + f_row ** 2 * n_ratio * (w_im_a - 2.0 * w_im1_a + w_im2_a)

            if w_i2_b < 0:
                raise NegativeWeightError("G2 enforcement generated a negative weight")

            self.set_weight(w_i2_b, row_index, 2, surface_edge)

            P_i0_b = self.get_point(row_index, 0, surface_edge)
            P_i1_b = self.get_point(row_index, 1, surface_edge)
            P_im_a = other.get_point(row_index, 0, other_surface_edge)
            P_im1_a = other.get_point(row_index, 1, other_surface_edge)
            P_im2_a = other.get_point(row_index, 2, other_surface_edge)

            P_i2_b = (2.0 * w_i1_b / w_i2_b * P_i1_b - w_i0_b / w_i2_b * P_i0_b) + f_row ** 2 * n_ratio * (
                        1 / w_i2_b) * (
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

    def dSdu_v2(self, u: float, v: float):
        n, m = self.degree_u, self.degree_v
        P = self.get_control_point_array()
        w = self.weights
        u, v = self._cast_uv(u, v)
        if isinstance(u, np.ndarray):
            assert u.shape == v.shape

        A_1 = np.zeros(P.shape[2])
        A_2 = np.zeros(P.shape[2])
        B_1 = np.zeros(P.shape[2])
        B_2 = np.zeros(P.shape[2])

        for i in range(self.degree_u + 1):
            for j in range(self.degree_v + 1):
                dbudu = self.degree_u * (
                            bernstein_poly(self.degree_u - 1, i - 1, u) - bernstein_poly(self.degree_u - 1, i, u))
                A_1 += bernstein_poly(n, i, u) * bernstein_poly(m, j, v) * w[i, j]
                A_2 += dbudu * bernstein_poly(m, j, v) * w[i, j] * P[i, j, :]
                B_1 += bernstein_poly(n, i, u) * bernstein_poly(m, j, v) * w[i, j] * P[i, j, :]
                B_2 += dbudu * bernstein_poly(m, j, v) * w[i, j]
        A = A_1 * A_2
        B = B_1 * B_2
        return (A - B) / (A_1 ** 2)

    def dSdv_v2(self, u: float, v: float):
        n, m = self.degree_u, self.degree_v
        P = self.get_control_point_array()
        w = self.weights
        u, v = self._cast_uv(u, v)
        if isinstance(u, np.ndarray):
            assert u.shape == v.shape

        A_1 = np.zeros(P.shape[2])
        A_2 = np.zeros(P.shape[2])
        B_1 = np.zeros(P.shape[2])
        B_2 = np.zeros(P.shape[2])

        for i in range(self.degree_u + 1):
            for j in range(self.degree_v + 1):
                dbvdv = m * (bernstein_poly(m - 1, j - 1, v) - bernstein_poly(m - 1, j, v))
                A_1 += bernstein_poly(n, i, u) * bernstein_poly(m, j, v) * w[i, j]
                A_2 += bernstein_poly(n, i, u) * dbvdv * w[i, j] * P[i, j, :]
                B_1 += bernstein_poly(n, i, u) * bernstein_poly(m, j, v) * w[i, j] * P[i, j, :]
                B_2 += bernstein_poly(n, i, u) * dbvdv * w[i, j]
        A = A_1 * A_2
        B = B_1 * B_2
        return (A - B) / (A_1 ** 2)

    def dSdu(self, u: float or np.ndarray, v: float or np.ndarray):
        n, m = self.degree_u, self.degree_v
        P = self.get_control_point_array()
        u, v = self._cast_uv(u, v)
        if isinstance(u, np.ndarray):
            assert u.shape == v.shape

        weight_arr = np.array([[bernstein_poly(n, i, u) * bernstein_poly(m, j, v) * self.weights[i, j]
                                for j in range(m + 1)] for i in range(n + 1)])
        weight_sum = weight_arr.reshape(-1, weight_arr.shape[-1]).sum(axis=0)

        point_arr = np.array([[np.array([(bernstein_poly(n, i, u) * bernstein_poly(m, j, v) *
                                          self.weights[i, j])]).T @ np.array([P[i, j, :]])
                               for j in range(m + 1)] for i in range(n + 1)])
        point_sum = point_arr.reshape(-1, len(u), 3).sum(axis=0)

        point_arr_deriv = np.array([[np.array([((bernstein_poly(n - 1, i - 1, u) - bernstein_poly(n - 1, i, u)) *
                                                bernstein_poly(m, j, v) * self.weights[i, j])]).T @
                                     np.array([P[i, j, :]]) for j in range(m + 1)] for i in range(n + 1)])
        point_deriv_sum = point_arr_deriv.reshape(-1, len(u), 3).sum(axis=0)

        weight_arr_deriv = np.array([[(bernstein_poly(n - 1, i - 1, u) - bernstein_poly(n - 1, i, u)) *
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
        u, v = self._cast_uv(u, v)
        if isinstance(u, np.ndarray):
            assert u.shape == v.shape

        weight_arr = np.array([[bernstein_poly(n, i, u) * bernstein_poly(m, j, v) * self.weights[i, j]
                                for j in range(m + 1)] for i in range(n + 1)])
        weight_sum = weight_arr.reshape(-1, weight_arr.shape[-1]).sum(axis=0)

        point_arr = np.array([[np.array([(bernstein_poly(n, i, u) * bernstein_poly(m, j, v) *
                                          self.weights[i, j])]).T @ np.array([P[i, j, :]])
                               for j in range(m + 1)] for i in range(n + 1)])
        point_sum = point_arr.reshape(-1, len(u), 3).sum(axis=0)

        point_arr_deriv = np.array([[np.array([((bernstein_poly(m - 1, j - 1, v) - bernstein_poly(m - 1, j, v)) *
                                                bernstein_poly(n, i, u) * self.weights[i, j])]).T @
                                     np.array([P[i, j, :]]) for j in range(m + 1)] for i in range(n + 1)])
        point_deriv_sum = point_arr_deriv.reshape(-1, len(u), 3).sum(axis=0)

        weight_arr_deriv = np.array([[(bernstein_poly(m - 1, j - 1, v) - bernstein_poly(m - 1, j, v)) *
                                      bernstein_poly(n, i, u) * self.weights[i, j]
                                      for j in range(m + 1)] for i in range(n + 1)])
        weight_deriv_sum = weight_arr_deriv.reshape(-1, weight_arr_deriv.shape[-1]).sum(axis=0)

        A = m * np.tile(weight_sum, (3, 1)).T * point_deriv_sum
        B = m * point_sum * np.tile(weight_deriv_sum, (3, 1)).T
        W = np.tile(weight_sum ** 2, (3, 1)).T

        return (A - B) / W

    def d2Sdu2(self, u: float or np.ndarray, v: float or np.ndarray):
        n, m = self.degree_u, self.degree_v
        P = self.get_control_point_array()
        #assert type(u) == type(v)
        u, v = self._cast_uv(u, v)
        if isinstance(u, np.ndarray):
            assert u.shape == v.shape

        weight_arr = np.array([[bernstein_poly(n, i, u) * bernstein_poly(m, j, v) * self.weights[i, j]
                                for j in range(m + 1)] for i in range(n + 1)])
        weight_sum = weight_arr.reshape(-1, weight_arr.shape[-1]).sum(axis=0)

        point_arr = np.array([[np.array([(bernstein_poly(n, i, u) * bernstein_poly(m, j, v) *
                                          self.weights[i, j])]).T @ np.array([P[i, j, :]])
                               for j in range(m + 1)] for i in range(n + 1)])
        point_sum = point_arr.reshape(-1, len(u), 3).sum(axis=0)

        point_arr_deriv = np.array([[np.array([((bernstein_poly(n - 1, i - 1, u) - bernstein_poly(n - 1, i, u)) *
                                                bernstein_poly(m, j, v) * self.weights[i, j])]).T @
                                     np.array([P[i, j, :]]) for j in range(m + 1)] for i in range(n + 1)])
        point_deriv_sum = point_arr_deriv.reshape(-1, len(u), 3).sum(axis=0)

        weight_arr_deriv = np.array([[(bernstein_poly(n - 1, i - 1, u) - bernstein_poly(n - 1, i, u)) *
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

        return (W * (dA - dB) - dW * (A - B)) / W ** 2

    def d2Sdv2(self, u: float or np.ndarray, v: float or np.ndarray):
        n, m = self.degree_u, self.degree_v
        P = self.get_control_point_array()
        #assert type(u) == type(v)
        u, v = self._cast_uv(u, v)
        if isinstance(u, np.ndarray):
            assert u.shape == v.shape

        weight_arr = np.array([[bernstein_poly(n, i, u) * bernstein_poly(m, j, v) * self.weights[i, j]
                                for j in range(m + 1)] for i in range(n + 1)])
        weight_sum = weight_arr.reshape(-1, weight_arr.shape[-1]).sum(axis=0)

        point_arr = np.array([[np.array([(bernstein_poly(n, i, u) * bernstein_poly(m, j, v) *
                                          self.weights[i, j])]).T @ np.array([P[i, j, :]])
                               for j in range(m + 1)] for i in range(n + 1)])
        point_sum = point_arr.reshape(-1, len(u), 3).sum(axis=0)

        point_arr_deriv = np.array([[np.array([((bernstein_poly(m - 1, j - 1, v) - bernstein_poly(m - 1, j, v)) *
                                                bernstein_poly(n, i, u) * self.weights[i, j])]).T @
                                     np.array([P[i, j, :]]) for j in range(m + 1)] for i in range(n + 1)])
        point_deriv_sum = point_arr_deriv.reshape(-1, len(u), 3).sum(axis=0)

        weight_arr_deriv = np.array([[(bernstein_poly(m - 1, j - 1, v) - bernstein_poly(m - 1, j, v)) *
                                      bernstein_poly(n, i, u) * self.weights[i, j]
                                      for j in range(m + 1)] for i in range(n + 1)])
        weight_deriv_sum = weight_arr_deriv.reshape(-1, weight_arr_deriv.shape[-1]).sum(axis=0)

        point_arr_deriv2 = np.array([[np.array([((bernstein_poly(m - 2, j - 2, v) -
                                                  2 * bernstein_poly(m - 2, j - 1, v) +
                                                  bernstein_poly(m - 2, j, v)) *
                                                 bernstein_poly(n, i, u) * self.weights[i, j])]).T @
                                      np.array([P[i, j, :]]) for j in range(m + 1)] for i in range(n + 1)])
        point_deriv2_sum = point_arr_deriv2.reshape(-1, len(u), 3).sum(axis=0)

        weight_arr_deriv2 = np.array([[(bernstein_poly(m - 2, j - 2, v) -
                                        2 * bernstein_poly(m - 2, j - 1, v) +
                                        bernstein_poly(m - 2, j, v)) *
                                       bernstein_poly(n, i, u) * self.weights[i, j]
                                       for j in range(m + 1)] for i in range(n + 1)])
        weight_deriv2_sum = weight_arr_deriv2.reshape(-1, weight_arr_deriv2.shape[-1]).sum(axis=0)

        A = m * np.tile(weight_sum, (3, 1)).T * point_deriv_sum
        B = m * point_sum * np.tile(weight_deriv_sum, (3, 1)).T
        W = np.tile(weight_sum ** 2, (3, 1)).T

        dA = m ** 2 * np.tile(weight_deriv_sum, (3, 1)).T * point_deriv_sum + np.tile(
            weight_sum, (3, 1)).T * point_deriv2_sum
        dB = m ** 2 * point_deriv_sum * np.tile(weight_deriv_sum, (3, 1)).T + point_sum * np.tile(
            weight_deriv2_sum, (3, 1)).T
        dW = 2 * m * np.tile(weight_sum, (3, 1)).T * np.tile(weight_deriv_sum, (3, 1)).T

        return (W * (dA - dB) - dW * (A - B)) / W ** 2

    def get_edge(self, edge: SurfaceEdge, n_points: int = 10) -> np.ndarray:
        if edge == SurfaceEdge.v1:
            return np.array([self.evaluate_ndarray(u, 1.0) for u in np.linspace(0.0, 1.0, n_points)])
        elif edge == SurfaceEdge.v0:
            return np.array([self.evaluate_ndarray(u, 0.0) for u in np.linspace(0.0, 1.0, n_points)])
        elif edge == SurfaceEdge.u1:
            return np.array([self.evaluate_ndarray(1.0, v) for v in np.linspace(0.0, 1.0, n_points)])
        elif edge == SurfaceEdge.u0:
            return np.array([self.evaluate_ndarray(0.0, v) for v in np.linspace(0.0, 1.0, n_points)])
        else:
            raise ValueError(f"No edge called {edge}")

    def get_first_derivs_along_edge(self, edge: SurfaceEdge, n_points: int = 10, perp=True) -> np.ndarray:
        if edge == SurfaceEdge.v1:
            return np.array([(self.dSdv(u, 1.0) if perp else
                              self.dSdu(u, 1.0)) for u in np.linspace(0.0, 1.0, n_points)])
        elif edge == SurfaceEdge.v0:
            return np.array([(self.dSdv(u, 0.0) if perp else
                              self.dSdu(u, 0.0)) for u in np.linspace(0.0, 1.0, n_points)])
        elif edge == SurfaceEdge.u1:
            return np.array([(self.dSdu(1.0, v) if perp else
                              self.dSdv(1.0, v)) for v in np.linspace(0.0, 1.0, n_points)])
        elif edge == SurfaceEdge.u0:
            return np.array([(self.dSdu(0.0, v) if perp else
                              self.dSdv(0.0, v)) for v in np.linspace(0.0, 1.0, n_points)])
        else:
            raise ValueError(f"No edge called {edge}")

    def get_first_derivs_along_edge_v2(self, edge: SurfaceEdge, n_points: int = 10, perp=True) -> np.ndarray:
        if edge == SurfaceEdge.v1:
            return np.array([(self.dSdv_v2(u, 1.0) if perp else
                              self.dSdu_v2(u, 1.0)) for u in np.linspace(0.0, 1.0, n_points)])
        elif edge == SurfaceEdge.v0:
            return np.array([(self.dSdv_v2(u, 0.0) if perp else
                              self.dSdu_v2(u, 0.0)) for u in np.linspace(0.0, 1.0, n_points)])
        elif edge == SurfaceEdge.u1:
            return np.array([(self.dSdu_v2(1.0, v) if perp else
                              self.dSdv_v2(1.0, v)) for v in np.linspace(0.0, 1.0, n_points)])
        elif edge == SurfaceEdge.u0:
            return np.array([(self.dSdu_v2(0.0, v) if perp else
                              self.dSdv_v2(0.0, v)) for v in np.linspace(0.0, 1.0, n_points)])
        else:
            raise ValueError(f"No edge called {edge}")

    def get_second_derivs_along_edge(self, edge: SurfaceEdge, n_points: int = 10, perp=True) -> np.ndarray:
        if edge == SurfaceEdge.v1:
            return np.array([(self.d2Sdv2(u, 1.0) if perp else
                              self.d2Sdu2(u, 1.0)) for u in np.linspace(0.0, 1.0, n_points)])
        elif edge == SurfaceEdge.v0:
            return np.array([(self.d2Sdv2(u, 0.0) if perp else
                              self.d2Sdu2(u, 0.0)) for u in np.linspace(0.0, 1.0, n_points)])
        elif edge == SurfaceEdge.u1:
            return np.array([(self.d2Sdu2(1.0, v) if perp else
                              self.d2Sdv2(1.0, v)) for v in np.linspace(0.0, 1.0, n_points)])
        elif edge == SurfaceEdge.u0:
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
        self_perp_edge_derivs[np.absolute(self_perp_edge_derivs) < 1e-6] = 0.0
        other_perp_edge_derivs[np.absolute(other_perp_edge_derivs) < 1e-6] = 0.0

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
                magnitude_ratios.append(np.nan_to_num(self_perp_edge_deriv / other_perp_edge_deriv, nan=0))

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
        self_perp_edge_derivs[np.absolute(self_perp_edge_derivs) < 1e-6] = 0.0
        other_perp_edge_derivs[np.absolute(other_perp_edge_derivs) < 1e-6] = 0.0

        ratios_other_self = other_perp_edge_derivs / self_perp_edge_derivs
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

    def get_u_or_v_given_uvxyz(self, u: float = None, v: float = None, uv_guess: float = 0.5,
                               x: float = None, y: float = None, z: float = None):
        """
        Computes one parametric value given the other and a specified :math:`x`-, :math:`y`-, or :math:`z`-location.
        """
        # Validate inputs
        if u is None and v is None or (u is not None and v is not None):
            raise ValueError("Must specify exactly one of either u or v")
        xyz_spec = (x is not None, y is not None, z is not None)
        if len([xyz for xyz in xyz_spec if xyz]) != 1:
            raise ValueError("Must specify exactly one of x, y, or z")

        if x is not None:
            xyz, xyz_val = "x", x
        elif y is not None:
            xyz, xyz_val = "y", y
        elif z is not None:
            xyz, xyz_val = "z", z
        else:
            raise ValueError("Did not detect an x, y, or z input")

        def root_find_func_u(u_current):
            point = self.evaluate_simple(u_current, v)
            return np.array([getattr(point, xyz).m - xyz_val])

        def root_find_func_v(v_current):
            point = self.evaluate_simple(u, v_current)
            return np.array([getattr(point, xyz).m - xyz_val])

        if v is not None:
            return fsolve(root_find_func_u, x0=np.array([uv_guess]))[0]
        if u is not None:
            return fsolve(root_find_func_v, x0=np.array([uv_guess]))[0]
        raise ValueError("Did not detect a u or v input")

    def split_at_u(self, u0: float) -> ("RationalBezierSurface", "RationalBezierSurface"):
        """
        Splits the rational Bezier surface at :math:`u=u_0` along the :math:`v`-parametric direction.
        """
        Pw = self.get_xyzw_array()

        def de_casteljau(i: int, j: int, k: int) -> np.ndarray:
            """
            Based on https://en.wikipedia.org/wiki/De_Casteljau%27s_algorithm. Recursive algorithm where the
            base case is just the value of the ith original control point.

            Parameters
            ----------
            i: int
                Lower index
            j: int
                Upper index
            k: int
                Control point row index

            Returns
            -------
            np.ndarray
                A one-dimensional array containing the :math:`x` and :math:`y` values of a control point evaluated
                at :math:`(i,j)` for a Bézier curve split at the parameter value ``t_split``
            """
            if j == 0:
                return Pw[i, k, :]
            return de_casteljau(i, j - 1, k) * (1 - u0) + de_casteljau(i + 1, j - 1, k) * u0

        bez_surf_split_1_Pw = np.array([
            [de_casteljau(i=0, j=i, k=k) for i in range(self.Nu)] for k in range(self.Nv)
        ])
        bez_surf_split_2_Pw = np.array([
            [de_casteljau(i=i, j=self.degree_u - i, k=k) for i in range(self.Nu)] for k in range(self.Nv)
        ])

        transposed_Pw_1 = np.transpose(bez_surf_split_1_Pw, (1, 0, 2))
        transposed_Pw_2 = np.transpose(bez_surf_split_2_Pw, (1, 0, 2))

        return (
            RationalBezierSurface.generate_from_array(
                transposed_Pw_1[:, :, :3] / np.repeat(transposed_Pw_1[:, :, -1][:, :, np.newaxis], 3, axis=2),
                transposed_Pw_1[:, :, -1]
            ),
            RationalBezierSurface.generate_from_array(
                transposed_Pw_2[:, :, :3] / np.repeat(transposed_Pw_2[:, :, -1][:, :, np.newaxis], 3, axis=2),
                transposed_Pw_2[:, :, -1]
            )
        )

    def split_at_v(self, v0: float) -> ("BezierSurface", "BezierSurface"):
        """
        Splits the rational Bezier surface at :math:`v=v_0` along the :math:`u`-parametric direction.
        """
        Pw = self.get_xyzw_array()

        def de_casteljau(i: int, j: int, k: int) -> np.ndarray:
            """
            Based on https://en.wikipedia.org/wiki/De_Casteljau%27s_algorithm. Recursive algorithm where the
            base case is just the value of the ith original control point.

            Parameters
            ----------
            i: int
                Lower index
            j: int
                Upper index
            k: int
                Control point row index

            Returns
            -------
            np.ndarray
                A one-dimensional array containing the :math:`x` and :math:`y` values of a control point evaluated
                at :math:`(i,j)` for a Bézier curve split at the parameter value ``t_split``
            """
            if j == 0:
                return Pw[k, i, :]
            return de_casteljau(i, j - 1, k) * (1 - v0) + de_casteljau(i + 1, j - 1, k) * v0

        bez_surf_split_1_Pw = np.array([
            [de_casteljau(i=0, j=i, k=k) for i in range(self.Nv)] for k in range(self.Nu)
        ])
        bez_surf_split_2_Pw = np.array([
            [de_casteljau(i=i, j=self.degree_v - i, k=k) for i in range(self.Nv)] for k in range(self.Nu)
        ])

        return (
            RationalBezierSurface.generate_from_array(
                bez_surf_split_1_Pw[:, :, :3] / np.repeat(bez_surf_split_1_Pw[:, :, np.newaxis], 3, axis=2),
                bez_surf_split_1_Pw[:, :, -1]
            ),
            RationalBezierSurface.generate_from_array(
                bez_surf_split_2_Pw[:, :, :3] / np.repeat(bez_surf_split_2_Pw[:, :, np.newaxis], 3, axis=2),
                bez_surf_split_2_Pw[:, :, -1]
            )
        )

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

    def plot_control_points(self, plot: pv.Plotter, **point_kwargs):
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

    def to_iges(self, *args, **kwargs) -> aerocaps.iges.entity.IGESEntity:
        return aerocaps.iges.surfaces.RationalBSplineSurfaceIGES(
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

        for point in bezier.control_points:

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

        knots_u = np.array([0.0 for _ in bezier.control_points] + [1.0 for _ in bezier.control_points])
        degree_v = 2
        degree_u = len(bezier.control_points) - 1

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
            return 1.0 if i in possible_span_indices_u_or_v and self._find_span(t, possible_spans_u_or_v,
                                                                                possible_span_indices_u_or_v) == i else 0.0
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
        return np.array(
            [self._cox_de_boor(t, i, p, knot_vector, possible_spans_u_or_v, possible_span_indices_u_or_v) for i in
             range(n_control_points_u_or_v)])

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
        return np.array(
            [[self.evaluate_ndarray(U[i, j], V[i, j]) for j in range(U.shape[1])] for i in range(U.shape[0])])

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

    def plot_control_points(self, plot: pv.Plotter, **point_kwargs):
        point_objs, _ = self.generate_control_point_net()
        point_arr = np.array([point_obj.as_array() for point_obj in point_objs])
        plot.add_points(point_arr, **point_kwargs)


class TrimmedSurface(Surface):
    def __init__(self,
                 untrimmed_surface: Surface,
                 outer_boundary: Geometry3D,
                 inner_boundaries: typing.List[Geometry3D] = None):
        self.untrimmed_surface = untrimmed_surface
        self.outer_boundary = outer_boundary
        self.inner_boundaries = inner_boundaries

    def evaluate(self, Nu: int, Nv: int) -> np.ndarray:
        raise NotImplementedError("Evaluation not yet implemented for trimmed surfaces")

    def to_iges(self,
                untrimmed_surface_iges: aerocaps.iges.surfaces.IGESEntity,
                outer_boundary_iges: aerocaps.iges.curves.CurveOnParametricSurfaceIGES,
                inner_boundaries_iges: typing.List[aerocaps.iges.curves.CurveOnParametricSurfaceIGES] = None,
                *args, **kwargs) -> aerocaps.iges.entity.IGESEntity:
        return aerocaps.iges.surfaces.TrimmedSurfaceIGES(
            untrimmed_surface_iges,
            outer_boundary_iges,
            inner_boundaries_iges
        )


class PlanarFillSurfaceCreator:
    def __init__(self, closed_curve_loop_list: typing.List[
        Bezier3D or RationalBezierCurve3D or NURBSCurve3D or BSpline3D or Line3D]):
        """
        Generator class for fill surfaces comprised of point-curve polar-like patches.

        .. warning::
            The curves must be co-planar to achieve the expected result
        """
        self.closed_curve_loop_list = closed_curve_loop_list
        self.validate()

    def validate(self):
        endpoints = {}
        for curve in self.closed_curve_loop_list:
            endpoint_1 = curve.control_points[0]
            endpoint_2 = curve.control_points[-1]
            for endpoint_to_test in [endpoint_1, endpoint_2]:
                for endpoint in endpoints:
                    if endpoint.almost_equals(endpoint_to_test):
                        endpoints[endpoint] = True
                        break
                else:
                    endpoints[endpoint_to_test] = False

        for v in endpoints.values():
            if v:
                continue
            raise ValueError("Assigned curve loop is not closed")

    def get_ordered_curve_list(self) -> (
            typing.List[Bezier3D or NURBSCurve3D or BSpline3D or RationalBezierCurve3D or Line3D],
            typing.List[Point3D]):
        # Copy a list of the curves
        curve_stack = deepcopy(self.closed_curve_loop_list)
        ordered_curve_list = [deepcopy(curve_stack[0])]

        # Get points for first curve. This will set the starting point and the direction of the curve loop.
        point_loop = curve_stack.pop(0).control_points

        while curve_stack:  # Loop until the curve stack is empty
            for curve_idx, curve in enumerate(curve_stack):
                if point_loop[-1].almost_equals(curve.control_points[0]):
                    ordered_curve_list.append(curve)
                    point_loop.extend(curve_stack.pop(curve_idx).control_points[1:])
                    break  # Go to the next curve in the stack
                elif point_loop[-1].almost_equals(curve.control_points[-1]):
                    ordered_curve_list.append(curve.reverse())
                    point_loop.extend(curve_stack.pop(curve_idx).control_points[:-1][::-1])
                    break  # Go to the next curve in the stack

        return ordered_curve_list, point_loop

    @staticmethod
    def get_envelope(ordered_curve_list: typing.List[
        Bezier3D or BSpline3D or NURBSCurve3D or RationalBezierCurve3D or Line3D],
                     control_point_loop: typing.List[Point3D]) -> (list, BezierSurface):
        loop_array = np.array([p.as_array() for p in control_point_loop])
        parametric_curves = []

        # Need to convert to 2-D to use shapely. Get the coordinate system of the plane containing the points
        # using cross products of vectors described by the points
        v1 = Vector3D(control_point_loop[0], control_point_loop[1])
        v2 = Vector3D(control_point_loop[0], control_point_loop[2])
        v3 = v1.cross(v2)
        v4 = v1.cross(v3)

        # The coordinate system is now fully described by v1, v3, and v4. v1 and v4 are the in-plane components,
        # while v3 is the out-of-plane component. The origin of this coordinate system is at control_point_loop[0].
        loop_array_transformed = aerocaps.transform_points_into_coordinate_system(
            loop_array, [v1, v4, v3], [aerocaps.IHat3D(), aerocaps.JHat3D(), aerocaps.KHat3D()]
        )
        # Make sure that all the curves are coplanar
        if not all([np.isclose(z, loop_array_transformed[0, 2]) for z in loop_array_transformed[1:, 2]]):
            raise ValueError("Curves are not all coplanar!")
        loop_array_2d = loop_array_transformed[:, :2]
        z_plane = loop_array_transformed[0, 2]

        # Create the polygon and find a point representing the center of the polygon while guaranteed to be inside
        # the polygon
        polygon = shapely.Polygon(loop_array_2d)
        envelope_2d = np.array(shapely.envelope(polygon).exterior.coords)
        envelope_2d[0, 0] -= 3.0
        envelope_2d[0, 1] -= 3.0
        envelope_2d[1, 0] += 3.0
        envelope_2d[1, 1] -= 3.0
        envelope_2d[2, 0] += 3.0
        envelope_2d[2, 1] += 3.0
        envelope_2d[3, 0] -= 3.0
        envelope_2d[3, 1] += 3.0
        x_min, x_max = envelope_2d[:, 0].min(), envelope_2d[:, 0].max()
        y_min, y_max = envelope_2d[:, 1].min(), envelope_2d[:, 1].max()
        dx, dy = (x_max - x_min), (y_max - y_min)

        # Get parametric curves in the plane defined by the envelope for each curve in the ordered curve list
        for curve in ordered_curve_list:
            cps_transformed = aerocaps.transform_points_into_coordinate_system(
                curve.get_control_point_array(), [v1, v4, v3], [aerocaps.IHat3D(), aerocaps.JHat3D(), aerocaps.KHat3D()]
            )
            cps_x = cps_transformed[:, 0]
            cps_y = cps_transformed[:, 1]
            u = [(cp_x - x_min) / dx for cp_x in cps_x]
            v = [(cp_y - y_min) / dy for cp_y in cps_y]
            uv = np.array([u, v]).T
            uv0 = np.column_stack((uv, np.zeros(uv.shape[0])))
            if isinstance(curve, Line3D):
                parametric_curve = curve.__class__(p0=Point3D.from_array(uv0[0, :]), p1=Point3D.from_array(uv0[1, :]))
            elif isinstance(curve, Bezier3D):
                parametric_curve = curve.__class__.generate_from_array(uv0)
            elif isinstance(curve, BSpline3D):
                parametric_curve = curve.__class__(uv0, curve.knot_vector, curve.degree)
            elif isinstance(curve, RationalBezierCurve3D):
                parametric_curve = curve.__class__.generate_from_array(uv0, curve.weights)
            elif isinstance(curve, NURBSCurve3D):
                parametric_curve = curve.__class__(uv0, curve.weights, curve.knot_vector, curve.degree)
            else:
                raise ValueError(f"Invalid curve type {type(curve)}")
            parametric_curves.append(parametric_curve)

        envelope_3d = np.column_stack((envelope_2d, z_plane * np.ones(envelope_2d.shape[0])))

        # Transform the newly created envelope back into the original coordinate system
        reverse_transformed_envelope_3d = aerocaps.transform_points_into_coordinate_system(
            envelope_3d, [aerocaps.IHat3D(), aerocaps.JHat3D(), aerocaps.KHat3D()], [v1, v4, v3]
        )

        # Create a planar rectangular surface from the transformed points
        pa = Point3D.from_array(reverse_transformed_envelope_3d[0, :])
        pb = Point3D.from_array(reverse_transformed_envelope_3d[1, :])
        pc = Point3D.from_array(reverse_transformed_envelope_3d[2, :])
        pd = Point3D.from_array(reverse_transformed_envelope_3d[3, :])
        planar_surf = aerocaps.BezierSurface([[pa, pd], [pb, pc]])

        return parametric_curves, planar_surf

    def generate(self) -> (list, list, BezierSurface):

        ordered_curve_list, control_point_loop = self.get_ordered_curve_list()
        parametric_curves, planar_surf = self.get_envelope(
            ordered_curve_list, control_point_loop
        )

        return ordered_curve_list, parametric_curves, planar_surf

    def to_iges(self) -> typing.List[aerocaps.iges.entity.IGESEntity]:
        entities = []
        ordered_curve_list, parametric_curves, planar_surf = self.generate()

        # Create the composite curves
        composite = aerocaps.CompositeCurve3D(ordered_curve_list)
        composite_para = aerocaps.CompositeCurve3D(parametric_curves)

        # Create the definition for the parametric curve
        curve_on_parametric_surface = aerocaps.CurveOnParametricSurface(
            planar_surf,
            composite_para,
            composite
        )

        # Create the trimmed surface object
        trimmed_surf = aerocaps.TrimmedSurface(planar_surf, curve_on_parametric_surface)

        # Compile the list of entities
        K1 = len(ordered_curve_list)
        K2 = K1 + len(parametric_curves)
        entities = [curve.to_iges() for curve in ordered_curve_list]
        entities.extend([curve.to_iges() for curve in parametric_curves])
        entities.append(composite.to_iges(entities[0:K1]))
        entities.append(composite_para.to_iges(entities[K1:K2]))
        entities.append(planar_surf.to_iges())
        entities.append(curve_on_parametric_surface.to_iges(entities[K2 + 2], entities[K2 + 1], entities[K2]))
        entities.append(trimmed_surf.to_iges(entities[K2 + 2], entities[K2 + 3]))

        return entities
