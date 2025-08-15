"""
Parametric surface classes (two-dimensional geometric objects defined by parameters :math:`u` and :math:`v`
that reside in three-dimensional space)
"""
import typing
from copy import deepcopy
from enum import Enum

import numpy as np
import pyvista as pv
import shapely
from rust_nurbs import *
from scipy.optimize import fsolve, minimize, OptimizeResult

import aerocaps.iges.curves
import aerocaps.iges.entity
import aerocaps.iges.surfaces
from aerocaps.geom.transformation import transform_points_into_coordinate_system, Transformation3D
from aerocaps.geom import Surface, InvalidGeometryError, NegativeWeightError, Geometry3D
from aerocaps.geom.curves import BezierCurve3D, Line3D, RationalBezierCurve3D, NURBSCurve3D, BSplineCurve3D, \
    CurveOnParametricSurface, CompositeCurve3D
from aerocaps.geom.plane import Plane
from aerocaps.geom.point import Point3D
from aerocaps.geom.tools import project_point_onto_line, measure_distance_point_line, rotate_point_about_axis, \
    add_vector_to_point, concave_hull
from aerocaps.geom.vector import Vector3D, IHat3D, JHat3D, KHat3D
from aerocaps.units.angle import Angle
from aerocaps.units.length import Length
from aerocaps.utils.array import unique_with_tolerance

__all__ = [
    "SurfaceEdge",
    "SurfaceCorner",
    "BezierSurface",
    "RationalBezierSurface",
    "BSplineSurface",
    "NURBSSurface",
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
    """
    Bézier surface class. A NURBS surface with no internal knots and all weights equal to unity.
    """
    def __init__(self,
                 points: typing.List[typing.List[Point3D]] or np.ndarray,
                 name: str = "BezierSurface",
                 construction: bool = False):
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
        ``aerocaps.geom.point.Point3D`` objects of by a
        3-D ``numpy`` array. For example, say we have six ``Point3D`` objects, A-F and would like to use
        them to create a :math:`2 \times 1` Bézier surface.
        Using the default constructor with the point objects,

        .. code-block:: python

            surf = BezierSurface([[pA, pB], [pC, pD], [pE, pF]])

        Using the array class method and point :math:`xyz` float values given by ``pA_x``, ``pA_y``, ``pA_z``, etc.,

        .. code-block:: python

            control_points = np.array([
                [[pA_x, pA_y, pA_z], [pB_x, pB_y, pB_z]],
                [[pC_x, pC_y, pC_z], [pD_x, pD_y, pD_z]],
                [[pE_x, pE_y, pE_z], [pF_x, pF_y, pF_z]],
            ])

            surf = BezierSurface(control_points)

        Parameters
        ----------
        points: typing.List[typing.List[Point3D]] or numpy.ndarray
            Control points for the Bézier surface, either as a nested list of :obj:`~aerocaps.geom.point.Point3D`
            objects or an :obj:`~numpy.ndarray` of size :math:`(n+1) \times (m+1) \times 3`,
            where :math:`n` is the surface degree in the :math:`u`-parametric direction and :math:`m` is the
            surface degree in the :math:`v`-parametric direction
        name: str
            Name of the geometric object. May be re-assigned a unique name when added to a
            :obj:`~aerocaps.geom.geometry_container.GeometryContainer`. Default: 'BezierSurface'
        construction: bool
            Whether this is a geometry used only for construction of other geometries. If ``True``, this
            geometry will not be exported or plotted. Default: ``False``
        """
        if isinstance(points, np.ndarray):
            points = [[Point3D.from_array(pt_row) for pt_row in pt_mat] for pt_mat in points]
        self.points = points
        super().__init__(name=name, construction=construction)

    @property
    def n_points_u(self) -> int:
        """Number of control points in the :math:`u`-parametric direction"""
        return len(self.points)

    @property
    def n_points_v(self) -> int:
        """Number of control points in the :math:`v`-parametric direction"""
        return len(self.points[0])

    @property
    def degree_u(self) -> int:
        """Surface degree in the :math:`u`-parametric direction"""
        return self.n_points_u - 1

    @property
    def degree_v(self) -> int:
        """Surface degree in the :math:`v`-parametric direction"""
        return self.n_points_v - 1

    @property
    def n(self) -> int:
        """
        Shorthand for :obj:`~aerocaps.geom.surfaces.BezierSurface.degree_u`

        Returns
        -------
        int
            Surface degree in the :math:`u`-parametric direction
        """
        return self.degree_u

    @property
    def m(self) -> int:
        """
        Shorthand for :obj:`~aerocaps.geom.surfaces.BezierSurface.degree_v`

        Returns
        -------
        int
            Surface degree in the :math:`v`-parametric direction
        """
        return self.degree_v

    def to_iges(self, *args, **kwargs) -> aerocaps.iges.entity.IGESEntity:
        """
        Converts the Bézier surface to an IGES entity. To add this IGES entity to an ``.igs`` file,
        use an :obj:`~aerocaps.iges.iges_generator.IGESGenerator`.
        """
        return aerocaps.iges.surfaces.BezierSurfaceIGES(self.get_control_point_array())

    def to_rational_bezier_surface(self) -> "RationalBezierSurface":
        """
        Converts the current non-rational Bézier surface to a rational Bézier surface by setting all weights to unity.

        Returns
        -------
        RationalBezierSurface
            Converted surface
        """
        return RationalBezierSurface(self.points, np.ones((self.degree_u + 1, self.degree_v + 1)))

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
    def from_curve_extrude(cls, curve: BezierCurve3D, distance: Length, extrude_axis: Vector3D = None,
                           symmetric: bool = False, reverse: bool = False):
        """
        Creates a Bézier surface by extruding a Bézier curve along an axis.

        .. important::

            If the input curve is linear, the ``extrude_axis`` argument must be specified.

        Parameters
        ----------
        curve: BezierCurve3D
            Curve to extrude. The most common use case is a planar curve, but this is not required.
        distance: Length
            Distance along the axis to extrude
        extrude_axis: Vector3D
            Optional direct specification of the extrusion axis. If not specified, a vector normal to the plane
            containing the first, second, and last control points of the curve is used. Default: ``None``
        symmetric: bool
            Whether to extrude in both directions. Default: ``False``
        reverse: bool
            Whether to flip the extrusion vector. Default: ``False``

        Returns
        -------
        BezierSurface
            Extruded surface
        """
        # Input validation
        if curve.degree < 2 and extrude_axis is None:
            raise ValueError("For linear Bézier curves (those with only two control points), "
                             "the 'extrude_axis` argument must be specified")

        # Get the axis along which to extrude
        if extrude_axis is None:
            plane = Plane(p0=curve.control_points[0], p1=curve.control_points[1], p2=curve.control_points[-1])
            extrude_axis = plane.compute_normal()
        else:
            extrude_axis = extrude_axis.get_normalized_vector()

        # Get the scaled extrusion vectors
        extrude_vec_forward = extrude_axis.scale(distance.m)
        extrude_vec_backward = extrude_axis.scale(distance.m)

        # Get the start and end point lists for the surface
        if symmetric:
            start_points = [add_vector_to_point(extrude_vec_forward, p) for p in curve.control_points]
            end_points = [add_vector_to_point(extrude_vec_backward, p) for p in curve.control_points]
        else:
            start_points = curve.control_points
            if reverse:
                end_points = [add_vector_to_point(extrude_vec_backward, p) for p in curve.control_points]
            else:
                end_points = [add_vector_to_point(extrude_vec_forward, p) for p in curve.control_points]

        # Create the extruded surface
        surf = cls([start_points, end_points])

        return surf

    def dSdu(self, u: float, v: float) -> np.ndarray:
        r"""
        Evaluates the first derivative with respect to :math:`u` at a single :math:`(u,v)` pair

        Parameters
        ----------
        u: float
            Position along :math:`u` in parametric space. Normally in the range :math:`[0,1]`
        v: float
            Position along :math:`v` in parametric space. Normally in the range :math:`[0,1]`

        Returns
        -------
        np.ndarray
            1-D array containing the :math:`x`-, :math:`y`-, and :math:`z`-components of the second derivative
        """
        P = self.get_control_point_array()
        return np.array(bezier_surf_dsdu(P, u, v))

    def dSdu_grid(self, Nu: int, Nv: int) -> np.ndarray:
        r"""
        Evaluates the first derivative with respect to :math:`u` on a linearly-spaced grid of :math:`u`- and
        :math:`v`-values.

        Parameters
        ----------
        Nu: int
            Number of evenly spaced :math:`u` values
        Nv: int
            Number of evenly spaced :math:`v` values

        Returns
        -------
        np.ndarray
            Array of size :math:`N_u \times N_v \times 3`
        """
        P = self.get_control_point_array()
        return np.array(bezier_surf_dsdu_grid(P, Nu, Nv))

    def dSdu_uvvecs(self, u: np.ndarray, v: np.ndarray):
        r"""
        Evaluates the first derivative of the surface with respect to :math:`u` at arbitrary vectors of
        :math:`u` and :math:`v`-values.

        Parameters
        ----------
        u: np.ndarray
            1-D array of :math:`u`-parameter values
        v: np.ndarray
            1-D array of :math:`v`-parameter values

        Returns
        -------
        np.ndarray
            Array of size :math:`\text{len}(u) \times \text{len}(v) \times 3`
        """
        P = self.get_control_point_array()
        return np.array(bezier_surf_dsdu_uvvecs(P, u, v))

    def dSdv(self, u: float or np.ndarray, v: float or np.ndarray):
        r"""
        Evaluates the first derivative with respect to :math:`v` at a single :math:`(u,v)` pair

        Parameters
        ----------
        u: float
            Position along :math:`u` in parametric space. Normally in the range :math:`[0,1]`
        v: float
            Position along :math:`v` in parametric space. Normally in the range :math:`[0,1]`

        Returns
        -------
        np.ndarray
            1-D array containing the :math:`x`-, :math:`y`-, and :math:`z`-components of the second derivative
        """
        P = self.get_control_point_array()
        return np.array(bezier_surf_dsdv(P, u, v))

    def dSdv_grid(self, Nu: int, Nv: int) -> np.ndarray:
        r"""
        Evaluates the first derivative with respect to :math:`v` on a linearly-spaced grid of :math:`u`- and
        :math:`v`-values.

        Parameters
        ----------
        Nu: int
            Number of evenly spaced :math:`u` values
        Nv: int
            Number of evenly spaced :math:`v` values

        Returns
        -------
        np.ndarray
            Array of size :math:`N_u \times N_v \times 3`
        """
        P = self.get_control_point_array()
        return np.array(bezier_surf_dsdv_grid(P, Nu, Nv))

    def dSdv_uvvecs(self, u: np.ndarray, v: np.ndarray):
        r"""
        Evaluates the first derivative of the surface with respect to :math:`v` at arbitrary vectors of
        :math:`u` and :math:`v`-values.

        Parameters
        ----------
        u: np.ndarray
            1-D array of :math:`u`-parameter values
        v: np.ndarray
            1-D array of :math:`v`-parameter values

        Returns
        -------
        np.ndarray
            Array of size :math:`\text{len}(u) \times \text{len}(v) \times 3`
        """
        P = self.get_control_point_array()
        return np.array(bezier_surf_dsdv_uvvecs(P, u, v))

    def d2Sdu2(self, u: float, v: float) -> np.ndarray:
        r"""
        Evaluates the second derivative with respect to :math:`u` at a single :math:`(u,v)` pair

        Parameters
        ----------
        u: float
            Position along :math:`u` in parametric space. Normally in the range :math:`[0,1]`
        v: float
            Position along :math:`v` in parametric space. Normally in the range :math:`[0,1]`

        Returns
        -------
        np.ndarray
            1-D array containing the :math:`x`-, :math:`y`-, and :math:`z`-components of the second derivative
        """
        P = self.get_control_point_array()
        return np.array(bezier_surf_d2sdu2(P, u, v))

    def d2Sdu2_grid(self, Nu: int, Nv: int) -> np.ndarray:
        r"""
        Evaluates the second derivative with respect to :math:`u` on a linearly-spaced grid of :math:`u`- and
        :math:`v`-values.

        Parameters
        ----------
        Nu: int
            Number of evenly spaced :math:`u` values
        Nv: int
            Number of evenly spaced :math:`v` values

        Returns
        -------
        np.ndarray
            Array of size :math:`N_u \times N_v \times 3`
        """
        P = self.get_control_point_array()
        return np.array(bezier_surf_d2sdu2_grid(P, Nu, Nv))

    def d2Sdu2_uvvecs(self, u: np.ndarray, v: np.ndarray):
        r"""
        Evaluates the second derivative of the surface with respect to :math:`u` at arbitrary vectors of
        :math:`u` and :math:`v`-values.

        Parameters
        ----------
        u: np.ndarray
            1-D array of :math:`u`-parameter values
        v: np.ndarray
            1-D array of :math:`v`-parameter values

        Returns
        -------
        np.ndarray
            Array of size :math:`\text{len}(u) \times \text{len}(v) \times 3`
        """
        P = self.get_control_point_array()
        return np.array(bezier_surf_d2sdu2_uvvecs(P, u, v))

    def d2Sdv2(self, u: float or np.ndarray, v: float or np.ndarray):
        r"""
        Evaluates the second derivative with respect to :math:`v` at a single :math:`(u,v)` pair

        Parameters
        ----------
        u: float
            Position along :math:`u` in parametric space. Normally in the range :math:`[0,1]`
        v: float
            Position along :math:`v` in parametric space. Normally in the range :math:`[0,1]`

        Returns
        -------
        np.ndarray
            1-D array containing the :math:`x`-, :math:`y`-, and :math:`z`-components of the second derivative
        """
        P = self.get_control_point_array()
        return np.array(bezier_surf_d2sdv2(P, u, v))

    def d2Sdv2_grid(self, Nu: int, Nv: int) -> np.ndarray:
        r"""
        Evaluates the second derivative with respect to :math:`v` on a linearly-spaced grid of :math:`u`- and
        :math:`v`-values.

        Parameters
        ----------
        Nu: int
            Number of evenly spaced :math:`u` values
        Nv: int
            Number of evenly spaced :math:`v` values

        Returns
        -------
        np.ndarray
            Array of size :math:`N_u \times N_v \times 3`
        """
        P = self.get_control_point_array()
        return np.array(bezier_surf_d2sdv2_grid(P, Nu, Nv))

    def d2Sdv2_uvvecs(self, u: np.ndarray, v: np.ndarray):
        r"""
        Evaluates the second derivative of the surface with respect to :math:`v` at arbitrary vectors of
        :math:`u` and :math:`v`-values.

        Parameters
        ----------
        u: np.ndarray
            1-D array of :math:`u`-parameter values
        v: np.ndarray
            1-D array of :math:`v`-parameter values

        Returns
        -------
        np.ndarray
            Array of size :math:`\text{len}(u) \times \text{len}(v) \times 3`
        """
        P = self.get_control_point_array()
        return np.array(bezier_surf_d2sdv2_uvvecs(P, u, v))

    def get_edge(self, edge: SurfaceEdge, n_points: int = 10) -> np.ndarray:
        r"""
        Evaluates the surface at ``n_points`` parameter locations along a given edge.

        Parameters
        ----------
        edge: SurfaceEdge
            Edge along which to evaluate
        n_points: int
            Number of evenly-spaced parameter locations at which to evaluate the edge curve. Default: 10

        Returns
        -------
        numpy.ndarray
            2-D array of size :math:`n_\text{points} \times 3`
        """
        P = self.get_control_point_array()
        if edge == SurfaceEdge.v1:
            return np.array(bezier_surf_eval_iso_v(P, n_points, 1.0))
        elif edge == SurfaceEdge.v0:
            return np.array(bezier_surf_eval_iso_v(P, n_points, 0.0))
        elif edge == SurfaceEdge.u1:
            return np.array(bezier_surf_eval_iso_u(P, 1.0, n_points))
        elif edge == SurfaceEdge.u0:
            return np.array(bezier_surf_eval_iso_u(P, 0.0, n_points))
        else:
            raise ValueError(f"No edge called {edge}")

    def get_first_derivs_along_edge(self, edge: SurfaceEdge, n_points: int = 10, perp: bool = True) -> np.ndarray:
        r"""
        Evaluates the parallel or perpendicular derivative along a surface edge at ``n_points`` parameter locations.
        The derivative represents either :math:`\frac{\partial \mathbf{S}(u,v)}{\partial u}` or
        :math:`\frac{\partial \mathbf{S}(u,v)}{\partial v}` depending on which edge is selected and which value is
        assigned to ``perp``.

        Parameters
        ----------
        edge: SurfaceEdge
            Edge along which to evaluate
        n_points: int
            Number of evenly-spaced parameter locations at which to evaluate the derivative. Default: 10
        perp: bool
            Whether to evaluate the cross-derivative. If ``False``, the derivative along the parameter direction
            parallel to the edge will be evaluated instead. Default: ``True``

        Returns
        -------
        numpy.ndarray
            2-D array of size :math:`n_\text{points} \times 3`
        """
        P = self.get_control_point_array()
        if edge == SurfaceEdge.v1:
            return np.array(bezier_surf_dsdv_iso_v(P, n_points, 1.0)) if perp else np.array(
                bezier_surf_dsdu_iso_v(P, n_points, 1.0))
        elif edge == SurfaceEdge.v0:
            return np.array(bezier_surf_dsdv_iso_v(P, n_points, 0.0)) if perp else np.array(
                bezier_surf_dsdu_iso_v(P, n_points, 0.0))
        elif edge == SurfaceEdge.u1:
            return np.array(bezier_surf_dsdu_iso_u(P, 1.0, n_points)) if perp else np.array(
                bezier_surf_dsdv_iso_u(P, 1.0, n_points))
        elif edge == SurfaceEdge.u0:
            return np.array(bezier_surf_dsdu_iso_u(P, 0.0, n_points)) if perp else np.array(
                bezier_surf_dsdv_iso_u(P, 0.0, n_points))
        else:
            raise ValueError(f"No edge called {edge}")

    def get_first_deriv_cp_sens_along_edge(self, edge: SurfaceEdge, i: int, j: int, n_points: int = 10,
                                           perp: bool = True) -> np.ndarray:
        r"""
        Gets the sensitivity of the first :math:`u`- or :math:`v`-derivative along an edge with respect to
        control point :math:`\mathbf{P}_{i,j}`

        Parameters
        ----------
        edge: SurfaceEdge
            Edge along which to evaluate
        i: int
            :math:`i`-index of the control point
        j: int
            :math:`j`-index of the control point
        n_points: int
            Number of evenly-spaced parameter locations at which to evaluate the derivative. Default: 10
        perp: bool
            Whether to evaluate the cross-derivative. If ``False``, the derivative along the parameter direction
            parallel to the edge will be evaluated instead. Default: ``True``

        Returns
        -------
        numpy.ndarray
            2-D array of size :math:`n_\text{points} \times 3`
        """
        if edge == SurfaceEdge.v1:
            return np.array(bezier_surf_dsdv_dp_iso_v(i, j, self.n, self.m, 3, n_points, 1.0)) if perp else np.array(
                bezier_surf_dsdu_dp_iso_v(i, j, self.n, self.m, 3, n_points, 1.0))
        elif edge == SurfaceEdge.v0:
            return np.array(bezier_surf_dsdv_dp_iso_v(i, j, self.n, self.m, 3, n_points, 0.0)) if perp else np.array(
                bezier_surf_dsdu_dp_iso_v(i, j, self.n, self.m, 3, n_points, 0.0))
        elif edge == SurfaceEdge.u1:
            return np.array(bezier_surf_dsdu_dp_iso_u(i, j, self.n, self.m, 3, 1.0, n_points)) if perp else np.array(
                bezier_surf_dsdv_dp_iso_u(i, j, self.n, self.m, 3, 1.0, n_points))
        elif edge == SurfaceEdge.u0:
            return np.array(bezier_surf_dsdu_dp_iso_u(i, j, self.n, self.m, 3, 0.0, n_points)) if perp else np.array(
                bezier_surf_dsdv_dp_iso_u(i, j, self.n, self.m, 3, 0.0, n_points))
        else:
            raise ValueError(f"No edge called {edge}")

    def get_second_derivs_along_edge(self, edge: SurfaceEdge, n_points: int = 10, perp: bool = True) -> np.ndarray:
        r"""
        Evaluates the parallel or perpendicular second derivative along a surface edge at ``n_points`` parameter
        locations. The derivative represents either :math:`\frac{\partial^2 \mathbf{S}(u,v)}{\partial u^2}` or
        :math:`\frac{\partial^2 \mathbf{S}(u,v)}{\partial v^2}` depending on which edge is selected and which value is
        assigned to ``perp``.

        Parameters
        ----------
        edge: SurfaceEdge
            Edge along which to evaluate
        n_points: int
            Number of evenly-spaced parameter locations at which to evaluate the second derivative. Default: 10
        perp: bool
            Whether to evaluate the cross-derivative. If ``False``, the second derivative along the parameter direction
            parallel to the edge will be evaluated instead. Default: ``True``

        Returns
        -------
        numpy.ndarray
            2-D array of size :math:`n_\text{points} \times 3`
        """
        P = self.get_control_point_array()
        if edge == SurfaceEdge.v1:
            return np.array(bezier_surf_d2sdv2_iso_v(P, n_points, 1.0)) if perp else np.array(
                bezier_surf_d2sdu2_iso_v(P, n_points, 1.0))
        elif edge == SurfaceEdge.v0:
            return np.array(bezier_surf_d2sdv2_iso_v(P, n_points, 0.0)) if perp else np.array(
                bezier_surf_d2sdu2_iso_v(P, n_points, 0.0))
        elif edge == SurfaceEdge.u1:
            return np.array(bezier_surf_d2sdu2_iso_u(P, 1.0, n_points)) if perp else np.array(
                bezier_surf_d2sdv2_iso_u(P, 1.0, n_points))
        elif edge == SurfaceEdge.u0:
            return np.array(bezier_surf_d2sdu2_iso_u(P, 0.0, n_points)) if perp else np.array(
                bezier_surf_d2sdv2_iso_u(P, 0.0, n_points))
        else:
            raise ValueError(f"No edge called {edge}")

    def get_second_deriv_cp_sens_along_edge(self, edge: SurfaceEdge, i: int, j: int, n_points: int = 10,
                                            perp: bool = True) -> np.ndarray:
        r"""
        Gets the sensitivity of the second :math:`u`- or :math:`v`-derivative along an edge with respect to
        control point :math:`\mathbf{P}_{i,j}`

        Parameters
        ----------
        edge: SurfaceEdge
            Edge along which to evaluate
        i: int
            :math:`i`-index of the control point
        j: int
            :math:`j`-index of the control point
        n_points: int
            Number of evenly-spaced parameter locations at which to evaluate the derivative. Default: 10
        perp: bool
            Whether to evaluate the cross-derivative. If ``False``, the derivative along the parameter direction
            parallel to the edge will be evaluated instead. Default: ``True``

        Returns
        -------
        numpy.ndarray
            2-D array of size :math:`n_\text{points} \times 3`
        """
        if edge == SurfaceEdge.v1:
            return np.array(bezier_surf_d2sdv2_dp_iso_v(i, j, self.n, self.m, 3, n_points, 1.0)) if perp else np.array(
                bezier_surf_d2sdu2_dp_iso_v(i, j, self.n, self.m, 3, n_points, 1.0))
        elif edge == SurfaceEdge.v0:
            return np.array(bezier_surf_d2sdv2_dp_iso_v(i, j, self.n, self.m, 3, n_points, 0.0)) if perp else np.array(
                bezier_surf_d2sdu2_dp_iso_v(i, j, self.n, self.m, 3, n_points, 0.0))
        elif edge == SurfaceEdge.u1:
            return np.array(bezier_surf_d2sdu2_dp_iso_u(i, j, self.n, self.m, 3, 1.0, n_points)) if perp else np.array(
                bezier_surf_d2sdv2_dp_iso_u(i, j, self.n, self.m, 3, 1.0, n_points))
        elif edge == SurfaceEdge.u0:
            return np.array(bezier_surf_d2sdu2_dp_iso_u(i, j, self.n, self.m, 3, 0.0, n_points)) if perp else np.array(
                bezier_surf_d2sdv2_dp_iso_u(i, j, self.n, self.m, 3, 0.0, n_points))
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

    def evaluate(self, u: float, v: float):
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
        return np.array(bezier_surf_eval(P, u, v))

    def evaluate_point3d(self, u: float, v: float) -> Point3D:
        r"""
        Evaluates the Bézier surface at a single :math:`(u,v)` parameter pair and returns a point object.

        Parameters
        ----------
        u: float
            Position along :math:`u` in parametric space. Normally in the range :math:`[0,1]`
        v: float
            Position along :math:`v` in parametric space. Normally in the range :math:`[0,1]`

        Returns
        -------
        Point3D
            Point object corresponding to the :math:`(u,v)` pair
        """
        return Point3D.from_array(self.evaluate(u, v))

    def evaluate_grid(self, Nu: int, Nv: int) -> np.ndarray:
        r"""
        Evaluates the Bézier surface on a uniform :math:`N_u \times N_v` grid of parameter values.

        Parameters
        ----------
        Nu: int
            Number of uniformly spaced parameter values in the :math:`u`-direction
        Nv: int
            Number of uniformly spaced parameter values in the :math:`v`-direction

        Returns
        -------
        numpy.ndarray
            Array of size :math:`N_u \times N_v \times 3`
        """
        P = self.get_control_point_array()
        return np.array(bezier_surf_eval_grid(P, Nu, Nv))

    def evaluate_uvvecs(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        r"""
        Evaluates the Bézier surface at arbitrary vectors of :math:`u` and :math:`v`-values.

        Parameters
        ----------
        u: np.ndarray
            1-D array of :math:`u`-parameter values
        v: np.ndarray
            1-D array of :math:`v`-parameter values

        Returns
        -------
        np.ndarray
            Array of size :math:`\text{len}(u) \times \text{len}(v) \times 3`
        """
        P = self.get_control_point_array()
        return np.array(bezier_surf_eval_uvvecs(P, u, v))

    def extract_edge_curve(self, surface_edge: SurfaceEdge) -> BezierCurve3D:
        """
        Extracts the control points from one of the four edges of the Bézier surface and outputs a Bézier curve with
        these control points

        Parameters
        ----------
        surface_edge: SurfaceEdge
            Edge along which to extract the curve

        Returns
        -------
        BezierCurve3D
            Bézier curve with control points corresponding to the control points along the edge of the surface
        """
        P = self.get_control_point_array()

        if surface_edge == SurfaceEdge.u0:
            return BezierCurve3D(P[0, :, :])
        if surface_edge == SurfaceEdge.u1:
            return BezierCurve3D(P[-1, :, :])
        if surface_edge == SurfaceEdge.v0:
            return BezierCurve3D(P[:, 0, :])
        if surface_edge == SurfaceEdge.v1:
            return BezierCurve3D(P[:, -1, :])

        raise ValueError(f"Invalid surface edge {surface_edge}")

    def elevate_degree_u(self) -> "BezierSurface":
        """
        Elevates the degree of the Bézier surface in the :math:`u`-parametric direction.

        .. figure:: ../images/bezier_surface_2x3_u_elevation.*
            :width: 600
            :align: center

            :math:`u` degree (:math:`n`) elevation

        Returns
        -------
        BezierSurface
            A new Bézier surface with identical shape to the current one but with one additional row of control points
            in the :math:`u`-parametric direction
        """
        n = self.degree_u
        m = self.degree_v
        P = self.get_control_point_array()

        # New array has one additional control point (current array only has n+1 control points)
        new_control_points = np.zeros((P.shape[0] + 1, P.shape[1], P.shape[2]))

        # Set starting and ending control points to what they already were
        new_control_points[0, :, :] = P[0, :, :]
        new_control_points[-1, :, :] = P[-1, :, :]

        # Update all the other control points
        for i in range(1, n + 1):  # 1 <= i <= n
            for j in range(0, m + 1):  # for all j
                new_control_points[i, j, :] = i / (n + 1) * P[i - 1, j, :] + (1 - i / (n + 1)) * P[i, j, :]

        return BezierSurface(new_control_points)

    def elevate_degree_v(self) -> "BezierSurface":
        r"""
        Elevates the degree of the Bézier surface in the :math:`v`-parametric direction.

        .. figure:: ../images/bezier_surface_2x3_v_elevation.*
            :width: 600
            :align: center

            :math:`v` degree (:math:`m`) elevation

        Returns
        -------
        BezierSurface
            A new Bézier surface with identical shape to the current one but with one additional row of control points
            in the :math:`v`-parametric direction
        """
        n = self.degree_u
        m = self.degree_v
        P = self.get_control_point_array()

        # New array has one additional control point (current array only has n+1 control points)
        new_control_points = np.zeros((P.shape[0], P.shape[1] + 1, P.shape[2]))

        # Set starting and ending control points to what they already were
        new_control_points[:, 0, :] = P[:, 0, :]
        new_control_points[:, -1, :] = P[:, -1, :]

        # Update all the other control points
        for i in range(0, n + 1):  # for all i
            for j in range(1, m + 1):  # 1 <= j <= m
                new_control_points[i, j, :] = j / (m + 1) * P[i, j - 1, :] + (1 - j / (m + 1)) * P[i, j, :]

        return BezierSurface(new_control_points)

    def extract_isoparametric_curve_u(self, u: float, Nv: int) -> np.ndarray:
        r"""
        Extracts a curve along the :math:`v`-direction at a fixed value of :math:`u`

        Parameters
        ----------
        u: float
            Constant value of :math:`u`
        Nv: int
            Number of points to evaluate, linearly spaced in :math:`v`

        Returns
        -------
        numpy.ndarray
            Array of size :math:`N_v \times 3` representing the :math:`x`-, :math:`y`-, and :math:`z`-coordinates
            of the points evaluated along the isoparametric curve
        """
        v_vec = np.linspace(0.0, 1.0, Nv)
        return np.array([self.evaluate(u, v) for v in v_vec])

    def extract_isoparametric_curve_v(self, Nu: int, v: float) -> np.ndarray:
        r"""
        Extracts a curve along the :math:`u`-direction at a fixed value of :math:`v`

        Parameters
        ----------
        Nu: int
            Number of points to evaluate, linearly spaced in :math:`u`
        v: float
            Constant value of :math:`v`

        Returns
        -------
        numpy.ndarray
            Array of size :math:`N_u \times 3` representing the :math:`x`-, :math:`y`-, and :math:`z`-coordinates
            of the points evaluated along the isoparametric curve
        """
        P = self.get_control_point_array()
        return np.array(bezier_surf_eval_iso_v(P, Nu, v))

    def get_parallel_degree(self, surface_edge: SurfaceEdge) -> int:
        r"""
        Gets the degree of the curve corresponding to the input surface edge.

        Parameters
        ----------
        surface_edge: SurfaceEdge
            Edge along which the parallel degree is evaluated

        Returns
        -------
        int
            Degree parallel to the edge
        """
        if surface_edge in [SurfaceEdge.v1, SurfaceEdge.v0]:
            return self.degree_u
        return self.degree_v

    def get_perpendicular_degree(self, surface_edge: SurfaceEdge) -> int:
        r"""
        Gets the degree of the curve in the parametric direction perpendicular to the input surface edge.

        Parameters
        ----------
        surface_edge: SurfaceEdge
            Edge along which the perpendicular degree is evaluated

        Returns
        -------
        int
            Degree perpendicular to the edge
        """
        if surface_edge in [SurfaceEdge.v1, SurfaceEdge.v0]:
            return self.degree_v
        return self.degree_u

    def get_parallel_n_points(self, surface_edge: SurfaceEdge) -> int:
        r"""
        Gets the number of control points in the parametric direction parallel to the input surface edge.

        Parameters
        ----------
        surface_edge: SurfaceEdge
            Edge along which the parallel number of control points is evaluated

        Returns
        -------
        int
            Number of control points parallel to the edge
        """
        if surface_edge in (SurfaceEdge.v1, SurfaceEdge.v0):
            return self.n_points_u
        return self.n_points_v

    def get_perpendicular_n_points(self, surface_edge: SurfaceEdge) -> int:
        r"""
        Gets the number of control points in the parametric direction perpendicular to the input surface edge.

        Parameters
        ----------
        surface_edge: SurfaceEdge
            Edge along which the perpendicular number of control points is evaluated

        Returns
        -------
        int
            Number of control points perpendicular to the edge
        """
        if surface_edge in (SurfaceEdge.v1, SurfaceEdge.v0):
            return self.n_points_v
        return self.n_points_u

    def get_point(self, row_index: int, continuity_index: int, surface_edge: SurfaceEdge) -> Point3D:
        r"""
        Gets the point corresponding to a particular index along the edge curve with perpendicular index
        corresponding to the level of continuity being applied. For example, for a :math:`6 \times 5` Bézier surface,
        the following code

        .. code-block:: python

            p = surf.get_point(2, 1, ac.SurfaceEdge.v0)

        returns the point :math:`\mathbf{P}_{2,1}` and

        .. code-block:: python

            p = surf.get_point(2, 1, ac.SurfaceEdge.u1)

        returns the point :math:`\mathbf{P}_{6-1,2} = \mathbf{P}_{5,2}`.

        .. seealso::

            :obj:`~aerocaps.geom.surfaces.BezierSurface.set_point`
                Setter equivalent of this method

        Parameters
        ----------
        row_index: int
            Index along the surface edge control points
        continuity_index: int
            Index in the parametric direction perpendicular to the surface edge. Normally either ``0``, ``1``, or ``2``
        surface_edge: SurfaceEdge
            Edge of the surface along which to retrieve the control point

        Returns
        -------
        Point3D
            Point used to enforce :math:`G^x` continuity, where :math:`x` is the value of ``continuity_index``
        """
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

    def get_point_ij(self, row_index: int, continuity_index: int, surface_edge: SurfaceEdge) -> (int, int):
        r"""
        Gets the point indices corresponding to a particular index along the edge curve with perpendicular index
        corresponding to the level of continuity being applied.

        Parameters
        ----------
        row_index: int
            Index along the surface edge control points
        continuity_index: int
            Index in the parametric direction perpendicular to the surface edge. Normally either ``0``, ``1``, or ``2``
        surface_edge: SurfaceEdge
            Edge of the surface along which to retrieve the control point

        Returns
        -------
        int, int
            Point indices used to enforce :math:`G^x` continuity, where :math:`x` is the value of ``continuity_index``
        """
        if surface_edge == SurfaceEdge.v1:
            return row_index, len(self.points[0]) - (continuity_index + 1)
        elif surface_edge == SurfaceEdge.v0:
            return row_index, continuity_index
        elif surface_edge == SurfaceEdge.u1:
            return len(self.points) - (continuity_index + 1), row_index
        elif surface_edge == SurfaceEdge.u0:
            return continuity_index, row_index
        else:
            raise ValueError("Invalid surface_edge value")

    def set_point(self, point: Point3D, row_index: int, continuity_index: int, surface_edge: SurfaceEdge):
        r"""
        Sets the point corresponding to a particular index along the edge curve with perpendicular index
        corresponding to the level of continuity being applied. For example, for a :math:`6 \times 5` Bézier surface,
        the following code

        .. code-block:: python

            p = ac.Point3D.from_array(np.array([3.0, 4.0, 5.0]))
            surf.set_point(p, 2, 1, ac.SurfaceEdge.v0)

        sets the value of point :math:`\mathbf{P}_{2,1}` to :math:`[3,4,5]^T` and

        .. code-block:: python

            p = ac.Point3D.from_array(np.array([3.0, 4.0, 5.0]))
            surf.set_point(p, 2, 1, ac.SurfaceEdge.u1)

        sets the value of point :math:`\mathbf{P}_{6-1,2} = \mathbf{P}_{5,2}` to :math:`[3,4,5]^T`.

        .. seealso::

            :obj:`~aerocaps.geom.surfaces.BezierSurface.get_point`
                Getter equivalent of this method

        Parameters
        ----------
        point: Point3D
            Point object to apply at the specified indices
        row_index: int
            Index along the surface edge control points
        continuity_index: int
            Index in the parametric direction perpendicular to the surface edge. Normally either ``0``, ``1``, or ``2``
        surface_edge: SurfaceEdge
            Edge of the surface along which to retrieve the control point
        """
        if surface_edge == SurfaceEdge.v1:
            self.points[row_index][-(continuity_index + 1)].x.m = point.x.m
            self.points[row_index][-(continuity_index + 1)].y.m = point.y.m
            self.points[row_index][-(continuity_index + 1)].z.m = point.z.m
        elif surface_edge == SurfaceEdge.v0:
            self.points[row_index][continuity_index].x.m = point.x.m
            self.points[row_index][continuity_index].y.m = point.y.m
            self.points[row_index][continuity_index].z.m = point.z.m
        elif surface_edge == SurfaceEdge.u1:
            self.points[-(continuity_index + 1)][row_index].x.m = point.x.m
            self.points[-(continuity_index + 1)][row_index].y.m = point.y.m
            self.points[-(continuity_index + 1)][row_index].z.m = point.z.m
        elif surface_edge == SurfaceEdge.u0:
            self.points[continuity_index][row_index].x.m = point.x.m
            self.points[continuity_index][row_index].y.m = point.y.m
            self.points[continuity_index][row_index].z.m = point.z.m
        else:
            raise ValueError("Invalid surface_edge value")

    @staticmethod
    def _evaluate_f_sign(surf_edge_1: SurfaceEdge, surf_edge_2: SurfaceEdge) -> float:
        """
        Evaluates the sign of the tangent proportionality factor across an edge pair

        Parameters
        ----------
        surf_edge_1: SurfaceEdge
            First surface edge
        surf_edge_2: SurfaceEdge
            Second surface edge

        Returns
        -------
        float
            ``-1.0`` if both surface edges end in 0 or both surface edges end in 1, ``1.0`` otherwise
        """
        surf_edges_0 = (SurfaceEdge.u0, SurfaceEdge.v0)
        surf_edges_1 = (SurfaceEdge.u1, SurfaceEdge.v1)
        if surf_edge_1 in surf_edges_0 and surf_edge_2 in surf_edges_0:
            return -1.0
        if surf_edge_1 in surf_edges_1 and surf_edge_2 in surf_edges_1:
            return -1.0
        return 1.0

    def enforce_g0(self, other: "BezierSurface",
                   surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge):
        r"""
        Enforces :math:`G^0` continuity along the input ``surface_edge`` by equating the control points along this edge
        to the control points along the ``other_surface_edge`` of the Bézier surface given by ``other``.
        The control points of the surface from which this method is called are modified in-place, and the control
        points of ``other`` are left unchanged.

        .. important::

            The parallel degree of the current surface along ``surface_edge`` must be equal to the parallel degree
            of the ``other`` surface along ``other_surface_edge``, otherwise an ``AssertionError`` will be raised.
            If these degrees are not equal, first elevate the degree of the surface with the lower parallel degree
            until the degrees match using either :obj:`~aerocaps.geom.surfaces.BezierSurface.elevate_degree_u`
            or :obj:`~aerocaps.geom.surfaces.BezierSurface.elevate_degree_v`, whichever is appropriate.

        .. seealso::

            :obj:`~aerocaps.geom.surfaces.BezierSurface.enforce_c0`
                Parametric continuity equivalent (:math:`C^0`)

        Parameters
        ----------
        other: BezierSurface
            Another Bézier surface along which an edge will be used for stitching
        surface_edge: SurfaceEdge
            The edge of the current surface to modify
        other_surface_edge: SurfaceEdge
            Tool edge of surface ``other`` which determines the positions of control points along ``surface_edge``
            of the current surface
        """
        self_parallel_degree = self.get_parallel_degree(surface_edge)
        other_parallel_degree = other.get_parallel_degree(other_surface_edge)
        if self_parallel_degree != other_parallel_degree:
            raise ValueError(f"Degree parallel to the edge of the input surface ({self_parallel_degree}) does "
                             f"not match the degree parallel to the edge of the other surface "
                             f"({other_parallel_degree})")
        for row_index in range(self.get_parallel_degree(surface_edge) + 1):
            self.set_point(other.get_point(row_index, 0, other_surface_edge), row_index, 0, surface_edge)

    def enforce_c0(self, other: "BezierSurface", surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge):
        """
        For zeroth-degree continuity, there is no difference between geometric (:math:`G^0`) and parametric
        (:math:`C^0`) continuity. Because this method is simply a convenience method that calls
        :obj:`~aerocaps.geom.surfaces.BezierSurface.enforce_g0`, see the documentation for that method for more
        detailed documentation.

        Parameters
        ----------
        other: BezierSurface
            Another Bézier surface along which an edge will be used for stitching
        surface_edge: SurfaceEdge
            The edge of the current surface to modify
        other_surface_edge: SurfaceEdge
            Tool edge of surface ``other`` which determines the positions of control points along ``surface_edge``
            of the current surface
        """
        self.enforce_g0(other, surface_edge, other_surface_edge)

    def enforce_g0g1(self, other: "BezierSurface", f: float,
                     surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge):
        r"""
        First enforces :math:`G^0` continuity, then tangent (:math:`G^1`) continuity is enforced according to
        the following equation:

        .. math::

            \mathcal{P}^{b,\mathcal{E}_b}_{k,1} = \mathcal{P}^{b,\mathcal{E}_b}_{k,0} + f \frac{p_{\perp}^{a,\mathcal{E}_a}}{p_{\perp}^{b,\mathcal{E}_b}} \left[\mathcal{P}^{a,\mathcal{E}_a}_{k,0} - \mathcal{P}^{a,\mathcal{E}_a}_{k,1} \right] \text{ for }k=0,1,\ldots,p_{\parallel}^{b,\mathcal{E}_b}

        Here, :math:`b` corresponds to the current surface, and :math:`a` corresponds to the ``other`` surface.
        The control points of the surface from which this method is called are modified in-place, and the control
        points of ``other`` are left unchanged.

        .. seealso::

            :obj:`~aerocaps.geom.surfaces.BezierSurface.enforce_g0`
                Geometric point continuity enforcement (:math:`G^0`)
            :obj:`~aerocaps.geom.surfaces.BezierSurface.enforce_c0c1`
                Parametric continuity equivalent (:math:`C^1`)

        Parameters
        ----------
        other: BezierSurface
            Another Bézier surface along which an edge will be used for stitching
        f: float
            Tangent proportionality factor
        surface_edge: SurfaceEdge
            The edge of the current surface to modify
        other_surface_edge: SurfaceEdge
            Tool edge of surface ``other`` which determines the positions of control points along ``surface_edge``
            of the current surface
        """
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
        r"""
        Equivalent to calling :obj:`~aerocaps.geom.surfaces.BezierSurface.enforce_g0g1` with ``f=1.0``. See that
        method for more detailed documentation.

        Parameters
        ----------
        other: BezierSurface
            Another Bézier surface along which an edge will be used for stitching
        surface_edge: SurfaceEdge
            The edge of the current surface to modify
        other_surface_edge: SurfaceEdge
            Tool edge of surface ``other`` which determines the positions of control points along ``surface_edge``
            of the current surface
        """
        self.enforce_g0g1(other, 1.0, surface_edge, other_surface_edge)

    def enforce_g0g1_multiface(self,
                               adjacent_surf_u0: "BezierSurface" = None,
                               adjacent_surf_u1: "BezierSurface" = None,
                               adjacent_surf_v0: "BezierSurface" = None,
                               adjacent_surf_v1: "BezierSurface" = None,
                               other_edge_u0: SurfaceEdge = None,
                               other_edge_u1: SurfaceEdge = None,
                               other_edge_v0: SurfaceEdge = None,
                               other_edge_v1: SurfaceEdge = None,
                               f_u0_initial: float = 1.0,
                               f_u1_initial: float = 1.0,
                               f_v0_initial: float = 1.0,
                               f_v1_initial: float = 1.0,
                               n_deriv_points: int = 10,
                               ) -> OptimizeResult:
        r"""
        .. warning::

            This is an experimental feature and should not be used in production geometries

        Enforces :math:`G^0` and :math:`G^1` continuity across multiple adjacent boundaries of a surface,
        up to all four boundaries. This is done by first enforcing :math:`G^0` continuity at all required boundaries
        and then optimizing the locations of the second rows of control points to minimize :math:`G^1` error at
        ``n_deriv_points`` locations along each of the boundary curves. The following is the cost function that is
        minimized:

        .. math::

            J(x_k) = \sum\limits_{i=0}^{n_p n_{\mathcal{E}}} \left( \left. \frac{\partial \mathbf{S}_i^a(u,v)}{\partial \mu} \right|_{u=u_i,v=v_i} - \frac{f_{\text{sgn},i}}{f_i(x_k)}  \left. \frac{\partial \mathbf{S}^b(u,v,x_k)}{\partial \mu} \right|_{u=u_i,v=v_i} \right)^2

        where

        * :math:`x_k` is the set of design variables to be optimized including the internal control point locations and tangent proportionality factors across each boundary
        * :math:`n_p` is the number of discrete first derivative calculations on each boundary (specified by ``n_deriv_points``)
        * :math:`n_\mathcal{E}` is the number of edges across which continuity is being enforced
        * :math:`\mathbf{S}_i^a(u,v)` is a surface specified by ``adjacent_surf_u0``, etc.
        * :math:`\mu` is equal to either :math:`u` or :math:`v`, determined by the parametric direction perpendicular to the edge
        * :math:`(u_i,v_i)` is the point along an edge where the derivative is being evaluated
        * :math:`f_{\text{sgn},i}` is the sign of the proportionality factor, :math:`-1` if both the target edge and other edge specified by :math:`i` end in :math:`0` or both end in :math:`1`, :math:`1` otherwise
        * :math:`f_i(x_k)` is the tangent proportionality factor
        * :math:`\mathbf{S}^b(u,v,x_k)` is the target surface (``self``)

        For maximum performance of the optimizer, the exact Jacobian is calculated:

        .. math::

            \frac{\partial J(x_k)}{\partial x_k} = 2 \sum\limits_{i=0}^{n_p n_{\mathcal{E}}} \left( \left. \frac{\partial \mathbf{S}_i^a(u,v)}{\partial \mu} \right|_{u=u_i,v=v_i} - \frac{f_{\text{sgn},i}}{f_i(x_k)}  \left. \frac{\partial \mathbf{S}^b(u,v,x_k)}{\partial \mu} \right|_{u=u_i,v=v_i} \right) \left[ -f_{\text{sgn},i} \frac{\partial}{\partial x_k} \left( \frac{1}{f_i} \right) \left( \left. \frac{\partial \mathbf{S}^b(u,v,x_k)}{\partial \mu} \right|_{u=u_i,v=v_i} \right) -\frac{f_{\text{sgn},i}}{f_i} \frac{\partial}{\partial x_k} \left(  \left. \frac{\partial \mathbf{S}^b(u,v,x_k)}{\partial \mu} \right|_{u=u_i,v=v_i} \right) \right]

        .. note::

            This method is reserved for the complex case where continuity is required across boundaries that
            share a surface corner. In the case of continuity with a single surface or a pair of surfaces
            with common boundaries on opposite sides of the surface (such as the :math:`v_0` and :math:`v_1`
            boundaries), the much simpler :obj:`~aerocaps.geom.surfaces.BezierSurface.enforce_g0g1` should
            be used.

        .. figure:: ../images/bezier_enforce_g0g1_multiface.*
            :align: center
            :width: 600

            Multi-face :math:`G^0` and :math:`G^1` continuity enforcement

        Parameters
        ----------
        adjacent_surf_u0: BezierSurface
            Surface sharing the :math:`u_0` boundary of ``target_surf``. Default: ``None``
        adjacent_surf_u1: BezierSurface
            Surface sharing the :math:`u_1` boundary of ``target_surf``. Default: ``None``
        adjacent_surf_v0: BezierSurface
            Surface sharing the :math:`v_0` boundary of ``target_surf``. Default: ``None``
        adjacent_surf_v1: BezierSurface
            Surface sharing the :math:`v_1` boundary of ``target_surf``. Default: ``None``
        other_edge_u0: SurfaceEdge
            Edge of surface ``adjacent_surf_u0`` that will be stitched. Default: ``None``
        other_edge_u1: SurfaceEdge
            Edge of surface ``adjacent_surf_u0`` that will be stitched. Default: ``None``
        other_edge_v0: SurfaceEdge
            Edge of surface ``adjacent_surf_u0`` that will be stitched. Default: ``None``
        other_edge_v1: SurfaceEdge
            Edge of surface ``adjacent_surf_u0`` that will be stitched. Default: ``None``
        f_u0_initial: float
            Initial value of the tangent proportionality factor across boundary :math:`u_0`. The final value
            selected by the optimizer will be different from this value. Default: ``1.0``
        f_u1_initial: float
            Initial value of the tangent proportionality factor across boundary :math:`u_1`. The final value
            selected by the optimizer will be different from this value. Default: ``1.0``
        f_v0_initial: float
            Initial value of the tangent proportionality factor across boundary :math:`v_0`. The final value
            selected by the optimizer will be different from this value. Default: ``1.0``
        f_v1_initial: float
            Initial value of the tangent proportionality factor across boundary :math:`v_1`. The final value
            selected by the optimizer will be different from this value. Default: ``1.0``
        n_deriv_points: int
            Number of discrete locations where the continuity error will be evaluated. Default: ``10``

        Returns
        -------
        OptimizeResult
            Result from the :math:`G^1` continuity error minimization problem solution
        """
        adjacent_surfs = (adjacent_surf_u0, adjacent_surf_u1, adjacent_surf_v0, adjacent_surf_v1)
        other_edges = (other_edge_u0, other_edge_u1, other_edge_v0, other_edge_v1)
        # Input validation
        if not any(adjacent_surfs):
            raise ValueError("Must specify at least one adjacent surface")
        if not any(other_edges):
            raise ValueError("Must specify at least one other edge")
        if len(adjacent_surfs) == 1:
            raise ValueError("For continuity enforcement with only one other surface, use 'enforce_g0g1' instead")
        if len(adjacent_surfs) != len(other_edges):
            raise ValueError("Must specify one 'other_edge' for every 'adjacent_surf'")

        # Create a mapping between the surfaces and edges
        surf_edge_mapping = {
            SurfaceEdge.u0: (adjacent_surf_u0, other_edge_u0, f_u0_initial),
            SurfaceEdge.u1: (adjacent_surf_u1, other_edge_u1, f_u1_initial),
            SurfaceEdge.v0: (adjacent_surf_v0, other_edge_v0, f_v0_initial),
            SurfaceEdge.v1: (adjacent_surf_v1, other_edge_v1, f_v1_initial)
        }
        for self_edge, other_data in surf_edge_mapping.items():
            if any(other_data) or all(other_data):
                continue
            raise ValueError("Must specify either both an 'adjacent_surf' and an 'other_edge' or neither for every "
                             "edge of the current surface")

        # Enforce G0 continuity with all surfaces
        for self_edge in surf_edge_mapping.keys():
            data = surf_edge_mapping[self_edge]
            if data[0] is None:
                continue
            self.enforce_g0(
                data[0], surface_edge=self_edge, other_surface_edge=data[1]
            )

        d1_other = {
            self_edge: data[0].get_first_derivs_along_edge(data[1], n_points=n_deriv_points) if data[0] else None
            for self_edge, data in surf_edge_mapping.items()
        }

        def get_point_ijs_to_update() -> typing.List[typing.Tuple[int]]:
            """Gets the indices of the points in the target surface that will be updated during the optimization"""
            point_ijs_to_update = []
            for surface_edge, _data in surf_edge_mapping.items():
                # Loop through all the points in the second row starting from the second point and ending at the
                # second-to-last point
                for row_index in range(1, self.get_parallel_n_points(surface_edge) - 1):
                    point_ij = self.get_point_ij(row_index, continuity_index=1, surface_edge=surface_edge)
                    if point_ij in point_ijs_to_update:
                        continue
                    point_ijs_to_update.append(point_ij)
            return point_ijs_to_update

        f_signs = {self_edge: self._evaluate_f_sign(self_edge, data[1])
                   for self_edge, data in surf_edge_mapping.items() if data is not None}
        f_vals = {self_edge: data[2] for self_edge, data in surf_edge_mapping.items() if data is not None}
        mod_ijs = get_point_ijs_to_update()
        mod_points = [self.points[i][j] for i, j in mod_ijs]
        x0 = np.array([p.as_array() for p in mod_points]).flatten()
        x0 = np.append(x0, np.array(list(f_vals.values())))

        def obj_fun_and_jac(x: np.ndarray) -> (float, np.ndarray):
            """
            Computes the objective function as the sum of the squares of the :math:`G^1` continuity error, along
            with the Jacobian

            Parameters
            ----------
            x: np.ndarray
                1-D array of design variable values

            Returns
            -------
            float, np.ndarray
                The objective function value and the Jacobian (a 1-D array of sensitivities)
            """
            x_reshaped = x[:3 * len(mod_ijs)].reshape((len(mod_points), 3))
            jac_arr = np.zeros(x.shape)
            # Update the points in-place
            for i in range(x_reshaped.shape[0]):
                mod_points[i].x.m = x_reshaped[i, 0]
                mod_points[i].y.m = x_reshaped[i, 1]
                mod_points[i].z.m = x_reshaped[i, 2]

            # Evaluate the objective function and Jacobian
            obj_fun_val = 0.0
            for edge_idx, (target_edge, multiface_data) in enumerate(surf_edge_mapping.items()):
                if surf_edge_mapping[target_edge][0] is None:
                    continue
                f = x[3 * len(mod_ijs) + edge_idx]
                f_sign = f_signs[target_edge]
                A = -f_sign * 1 / abs(f)
                dA = f_sign * 1 / f ** 2
                d1_self = self.get_first_derivs_along_edge(target_edge, n_points=n_deriv_points)

                # Objective function value update
                obj_fun_val += np.sum((d1_other[target_edge] + A * d1_self) ** 2)

                # Jacobian update loop
                start_xyz = 0
                for mod_ij in mod_ijs:
                    d1_sens_self = self.get_first_deriv_cp_sens_along_edge(
                        target_edge,
                        mod_ij[0],
                        mod_ij[1],
                        n_points=n_deriv_points
                    )
                    # Jacobian array update
                    for k in range(3):
                        jac_arr[start_xyz + k] += np.sum(
                            2 * (d1_other[target_edge][:, k] + A * d1_self[:, k]) * A * d1_sens_self[:, k]
                        )
                    start_xyz += 3
                jac_arr[3 * len(mod_ijs) + edge_idx] = np.sum(
                    2 * (d1_other[target_edge] + A * d1_self) * dA * d1_self
                )

            return obj_fun_val, jac_arr

        res = minimize(obj_fun_and_jac, x0, jac=True)
        return res

    def enforce_g0g1g2(self, other: "BezierSurface", f: float,
                       surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge):
        r"""
        First enforces :math:`G^0` and :math:`G^1` continuity, then curvature (:math:`G^2`) continuity is enforced
        according to the following equation:

        .. math::

            \mathcal{P}^{b,\mathcal{E}_b}_{k,2} = 2 \mathcal{P}^{b,\mathcal{E}_b}_{k,1} - \mathcal{P}^{b,\mathcal{E}_b}_{k,0} + f^2 \frac{p_{\perp}^{a,\mathcal{E}_a}(p_{\perp}^{a,\mathcal{E}_a}-1)}{p_{\perp}^{b,\mathcal{E}_b}(p_{\perp}^{b,\mathcal{E}_b}-1)} \left[ \mathcal{P}^{a,\mathcal{E}_a}_{k,0} - 2 \mathcal{P}^{a,\mathcal{E}_a}_{k,1} + \mathcal{P}^{a,\mathcal{E}_a}_{k,2} \right]  \text{ for }k=0,1,\ldots,p_{\parallel}^{b,\mathcal{E}_b}

        Here, :math:`b` corresponds to the current surface, and :math:`a` corresponds to the ``other`` surface.
        The control points of the surface from which this method is called are modified in-place, and the control
        points of ``other`` are left unchanged.

        .. seealso::

            :obj:`~aerocaps.geom.surfaces.BezierSurface.enforce_g0`
                Geometric point continuity enforcement (:math:`G^0`)
            :obj:`~aerocaps.geom.surfaces.BezierSurface.enforce_g0g1`
                Geometric tangent continuity enforcement (:math:`G^1`)
            :obj:`~aerocaps.geom.surfaces.BezierSurface.enforce_c0c1c2`
                Parametric continuity equivalent (:math:`C^2`)

        Parameters
        ----------
        other: BezierSurface
            Another Bézier surface along which an edge will be used for stitching
        f: float
            Tangent proportionality factor
        surface_edge: SurfaceEdge
            The edge of the current surface to modify
        other_surface_edge: SurfaceEdge
            Tool edge of surface ``other`` which determines the positions of control points along ``surface_edge``
            of the current surface
        """
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
        r"""
        Equivalent to calling :obj:`~aerocaps.geom.surfaces.BezierSurface.enforce_g0g1g2` with ``f=1.0``. See that
        method for more detailed documentation.

        Parameters
        ----------
        other: BezierSurface
            Another Bézier surface along which an edge will be used for stitching
        surface_edge: SurfaceEdge
            The edge of the current surface to modify
        other_surface_edge: SurfaceEdge
            Tool edge of surface ``other`` which determines the positions of control points along ``surface_edge``
            of the current surface
        """
        self.enforce_g0g1g2(other, 1.0, surface_edge, other_surface_edge)

    def enforce_g0g1g2_multiface(self,
                                 adjacent_surf_u0: "BezierSurface" = None,
                                 adjacent_surf_u1: "BezierSurface" = None,
                                 adjacent_surf_v0: "BezierSurface" = None,
                                 adjacent_surf_v1: "BezierSurface" = None,
                                 other_edge_u0: SurfaceEdge = None,
                                 other_edge_u1: SurfaceEdge = None,
                                 other_edge_v0: SurfaceEdge = None,
                                 other_edge_v1: SurfaceEdge = None,
                                 f_u0_initial: float = 1.0,
                                 f_u1_initial: float = 1.0,
                                 f_v0_initial: float = 1.0,
                                 f_v1_initial: float = 1.0,
                                 n_deriv_points: int = 10,
                                 ) -> OptimizeResult:
        r"""
        .. warning::

            This is an experimental feature and should not be used in production geometries

        Enforces :math:`G^0`, :math:`G^1`, and :math:`G^2` continuity across multiple adjacent boundaries of a surface,
        up to all four boundaries. This is done by first enforcing :math:`G^0` continuity at all required boundaries
        and then optimizing the locations of the second and third rows of control points to minimize
        :math:`G^1` and :math:`G^2` error at ``n_deriv_points`` locations along each of the boundary curves.
        The following is the cost function that is minimized:

        .. math::

            J(x_k) = \sum\limits_{l=1}^2 \sum\limits_{i=0}^{n_p n_{\mathcal{E}}} \left( \left. \frac{\partial^l \mathbf{S}_i^a(u,v)}{\partial \mu^l} \right|_{u=u_i,v=v_i} - \frac{f_{\text{sgn},i}}{f^l_i(x_k)}  \left. \frac{\partial^l \mathbf{S}^b(u,v,x_k)}{\partial \mu^l} \right|_{u=u_i,v=v_i} \right)^2

        where

        * :math:`x_k` is the set of design variables to be optimized including the internal control point locations and tangent proportionality factors across each boundary
        * :math:`n_p` is the number of discrete derivative calculations on each boundary (specified by ``n_deriv_points``)
        * :math:`n_\mathcal{E}` is the number of edges across which continuity is being enforced
        * :math:`\mathbf{S}_i^a(u,v)` is a surface specified by ``adjacent_surf_u0``, etc.
        * :math:`\mu` is equal to either :math:`u` or :math:`v`, determined by the parametric direction perpendicular to the edge
        * :math:`(u_i,v_i)` is the point along an edge where the derivative is being evaluated
        * :math:`f_{\text{sgn},i}` is the sign of the proportionality factor, :math:`-1` if both the target edge and other edge specified by :math:`i` end in :math:`0` or both end in :math:`1`, :math:`1` otherwise
        * :math:`f_i(x_k)` is the tangent proportionality factor
        * :math:`\mathbf{S}^b(u,v,x_k)` is the target surface (``self``)
        * :math:`l` is the derivative order

        For maximum performance of the optimizer, the exact Jacobian is calculated:

        .. math::

            \frac{\partial J(x_k)}{\partial x_k} = 2 \sum\limits_{l=1}^2 \sum\limits_{i=0}^{n_p n_{\mathcal{E}}} \left( \left. \frac{\partial^l \mathbf{S}_i^a(u,v)}{\partial \mu^l} \right|_{u=u_i,v=v_i} - \frac{f_{\text{sgn},i}}{f^l_i(x_k)}  \left. \frac{\partial^l \mathbf{S}^b(u,v,x_k)}{\partial \mu^l} \right|_{u=u_i,v=v_i} \right) \left[ -f_{\text{sgn},i} \frac{\partial}{\partial x_k} \left( \frac{1}{f^l_i} \right) \left( \left. \frac{\partial^l \mathbf{S}^b(u,v,x_k)}{\partial \mu^l} \right|_{u=u_i,v=v_i} \right) -\frac{f_{\text{sgn},i}}{f_i^l} \frac{\partial}{\partial x_k} \left(  \left. \frac{\partial^l \mathbf{S}^b(u,v,x_k)}{\partial \mu^l} \right|_{u=u_i,v=v_i} \right) \right]

        .. note::

            This method is reserved for the complex case where continuity is required across boundaries that
            share a surface corner. In the case of continuity with a single surface or a pair of surfaces
            with common boundaries on opposite sides of the surface (such as the :math:`v_0` and :math:`v_1`
            boundaries), the much simpler :obj:`~aerocaps.geom.surfaces.BezierSurface.enforce_g0g1g2` should
            be used.

        Parameters
        ----------
        adjacent_surf_u0: BezierSurface
            Surface sharing the :math:`u_0` boundary of ``target_surf``. Default: ``None``
        adjacent_surf_u1: BezierSurface
            Surface sharing the :math:`u_1` boundary of ``target_surf``. Default: ``None``
        adjacent_surf_v0: BezierSurface
            Surface sharing the :math:`v_0` boundary of ``target_surf``. Default: ``None``
        adjacent_surf_v1: BezierSurface
            Surface sharing the :math:`v_1` boundary of ``target_surf``. Default: ``None``
        other_edge_u0: SurfaceEdge
            Edge of surface ``adjacent_surf_u0`` that will be stitched. Default: ``None``
        other_edge_u1: SurfaceEdge
            Edge of surface ``adjacent_surf_u0`` that will be stitched. Default: ``None``
        other_edge_v0: SurfaceEdge
            Edge of surface ``adjacent_surf_u0`` that will be stitched. Default: ``None``
        other_edge_v1: SurfaceEdge
            Edge of surface ``adjacent_surf_u0`` that will be stitched. Default: ``None``
        f_u0_initial: float
            Initial value of the tangent proportionality factor across boundary :math:`u_0`. The final value
            selected by the optimizer will be different from this value. Default: ``1.0``
        f_u1_initial: float
            Initial value of the tangent proportionality factor across boundary :math:`u_1`. The final value
            selected by the optimizer will be different from this value. Default: ``1.0``
        f_v0_initial: float
            Initial value of the tangent proportionality factor across boundary :math:`v_0`. The final value
            selected by the optimizer will be different from this value. Default: ``1.0``
        f_v1_initial: float
            Initial value of the tangent proportionality factor across boundary :math:`v_1`. The final value
            selected by the optimizer will be different from this value. Default: ``1.0``
        n_deriv_points: int
            Number of discrete locations where the continuity error will be evaluated. Default: ``10``

        Returns
        -------
        OptimizeResult
            Result from the :math:`G^1`- and :math:`G^2`-continuity error minimization problem solution
        """
        adjacent_surfs = (adjacent_surf_u0, adjacent_surf_u1, adjacent_surf_v0, adjacent_surf_v1)
        other_edges = (other_edge_u0, other_edge_u1, other_edge_v0, other_edge_v1)
        # Input validation
        if not any(adjacent_surfs):
            raise ValueError("Must specify at least one adjacent surface")
        if not any(other_edges):
            raise ValueError("Must specify at least one other edge")
        if len(adjacent_surfs) == 1:
            raise ValueError("For continuity enforcement with only one other surface, use 'enforce_g0g1' instead")
        if len(adjacent_surfs) != len(other_edges):
            raise ValueError("Must specify one 'other_edge' for every 'adjacent_surf'")

        # Create a mapping between the surfaces and edges
        surf_edge_mapping = {
            SurfaceEdge.u0: (adjacent_surf_u0, other_edge_u0, f_u0_initial),
            SurfaceEdge.u1: (adjacent_surf_u1, other_edge_u1, f_u1_initial),
            SurfaceEdge.v0: (adjacent_surf_v0, other_edge_v0, f_v0_initial),
            SurfaceEdge.v1: (adjacent_surf_v1, other_edge_v1, f_v1_initial)
        }
        for self_edge, other_data in surf_edge_mapping.items():
            if any(other_data) or all(other_data):
                continue
            raise ValueError("Must specify either both an 'adjacent_surf' and an 'other_edge' or neither for every "
                             "edge of the current surface")

        # Enforce G0 continuity with all surfaces
        for self_edge in surf_edge_mapping.keys():
            data = surf_edge_mapping[self_edge]
            if data[0] is None:
                continue
            self.enforce_g0(
                data[0], surface_edge=self_edge, other_surface_edge=data[1]
            )

        d1_other = {
            self_edge: data[0].get_first_derivs_along_edge(data[1], n_points=n_deriv_points) if data[0] else None
            for self_edge, data in surf_edge_mapping.items()
        }
        d2_other = {
            self_edge: data[0].get_second_derivs_along_edge(data[1], n_points=n_deriv_points) if data[0] else None
            for self_edge, data in surf_edge_mapping.items()
        }

        def get_point_ijs_to_update() -> typing.List[typing.Tuple[int]]:
            """Gets the indices of the points in the target surface that will be updated during the optimization"""
            point_ijs_to_update = []
            for surface_edge, _data in surf_edge_mapping.items():
                # Loop through all the points in the second row starting from the second point and ending at the
                # second-to-last point
                for row_index in range(1, self.get_parallel_n_points(surface_edge) - 1):
                    point_ij = self.get_point_ij(row_index, continuity_index=1, surface_edge=surface_edge)
                    if point_ij in point_ijs_to_update:
                        continue
                    point_ijs_to_update.append(point_ij)
            for surface_edge, _data in surf_edge_mapping.items():
                for row_index in range(1, self.get_parallel_n_points(surface_edge) - 1):
                    point_ij_2 = self.get_point_ij(row_index, continuity_index=2, surface_edge=surface_edge)
                    if point_ij_2 in point_ijs_to_update:
                        continue
                    point_ijs_to_update.append(point_ij_2)
            return point_ijs_to_update

        f_signs = {self_edge: self._evaluate_f_sign(self_edge, data[1])
                   for self_edge, data in surf_edge_mapping.items() if data is not None}
        f_vals = {self_edge: data[2] for self_edge, data in surf_edge_mapping.items() if data is not None}
        mod_ijs = get_point_ijs_to_update()
        mod_points = [self.points[i][j] for i, j in mod_ijs]
        x0 = np.array([p.as_array() for p in mod_points]).flatten()
        x0 = np.append(x0, np.array(list(f_vals.values())))

        def obj_fun_and_jac(x: np.ndarray) -> (float, np.ndarray):
            """
            Computes the objective function as the sum of the squares of the :math:`G^1` continuity error, along
            with the Jacobian

            Parameters
            ----------
            x: np.ndarray
                1-D array of design variable values

            Returns
            -------
            float, np.ndarray
                The objective function value and the Jacobian (a 1-D array of sensitivities)
            """
            x_reshaped = x[:3 * len(mod_ijs)].reshape((len(mod_points), 3))
            jac_arr = np.zeros(x.shape)
            # Update the points in-place
            for i in range(x_reshaped.shape[0]):
                mod_points[i].x.m = x_reshaped[i, 0]
                mod_points[i].y.m = x_reshaped[i, 1]
                mod_points[i].z.m = x_reshaped[i, 2]

            # Evaluate the objective function and Jacobian
            obj_fun_val = 0.0
            for edge_idx, (target_edge, multiface_data) in enumerate(surf_edge_mapping.items()):
                if surf_edge_mapping[target_edge][0] is None:
                    continue
                f = x[3 * len(mod_ijs) + edge_idx]
                f_sign = f_signs[target_edge]
                A = -f_sign * 1 / abs(f)
                A2 = -f_sign * (1 / f) ** 2
                dA = f_sign * (1 / f) ** 2
                dA2 = 2 * f_sign * (1 / abs(f)) ** 3
                d1_self = self.get_first_derivs_along_edge(target_edge, n_points=n_deriv_points)
                d2_self = self.get_second_derivs_along_edge(target_edge, n_points=n_deriv_points)

                # Objective function value update
                obj_fun_val += np.sum((d1_other[target_edge] + A * d1_self) ** 2)
                obj_fun_val += np.sum((d2_other[target_edge] + A2 * d2_self) ** 2)

                # Jacobian update loop
                start_xyz = 0  # Jacobian array starting index
                for mod_ij_idx, mod_ij in enumerate(mod_ijs):
                    d1_sens_self = self.get_first_deriv_cp_sens_along_edge(
                        target_edge,
                        mod_ij[0],
                        mod_ij[1],
                        n_points=n_deriv_points
                    )
                    d2_sens_self = self.get_second_deriv_cp_sens_along_edge(
                        target_edge,
                        mod_ij[0],
                        mod_ij[1],
                        n_points=n_deriv_points
                    )
                    # Jacobian array update
                    for k in range(3):
                        jac_arr[start_xyz + k] += 2 * np.sum(
                            (d1_other[target_edge][:, k] + A * d1_self[:, k]) * A * d1_sens_self[:, k] + (
                                    d2_other[target_edge][:, k] + A2 * d2_self[:, k]
                            ) * A2 * d2_sens_self[:, k]
                        )
                    # Increment the starting Jacobian index
                    start_xyz += 3

                jac_arr[3 * len(mod_ijs) + edge_idx] = np.sum(
                    2 * (d1_other[target_edge] + A * d1_self) * dA * d1_self +
                    2 * (d2_other[target_edge] + A2 * d2_self) * dA2 * d2_self
                )

            return obj_fun_val, jac_arr

        # bounds = np.zeros((x0.shape[0], 2))
        # bounds[-4:, 0] = 0.05
        # bounds[-4:, 1] = 3.0
        # xyz_start_iii = 0
        # for iii in range(16):
        #     bounds[xyz_start_iii, 0] = 0.0
        #     bounds[xyz_start_iii, 1] = 1.0
        #     bounds[xyz_start_iii + 1, 0] = -1.0
        #     bounds[xyz_start_iii + 1, 1] = 0.0
        #     bounds[xyz_start_iii + 2, 0] = -0.3
        #     bounds[xyz_start_iii + 2, 1] = 0.3
        #     xyz_start_iii += 3

        # res = minimize(obj_fun_and_jac, x0, jac=True, bounds=bounds)
        res = minimize(obj_fun_and_jac, x0, jac=True)
        return res

    def get_u_or_v_given_uvxyz(self, u: float = None, v: float = None, uv_guess: float = 0.5,
                               x: Length = None, y: Length = None, z: Length = None) -> float:
        """
        Computes one parametric value given the other and a specified :math:`x`-, :math:`y`-, or :math:`z`-location.
        As an example, given a :obj:`~aerocaps.geom.surfaces.BezierSurface` object assigned to the variable ``surf``,
        the :math:`u`-parameter corresponding to :math:`y=1.4` along the :math:`v=0.8` isoparametric curve can be
        computed using

        .. code-block:: python

            u = surf.get_u_or_v_given_uvxyz(v=0.8, y=1.4)

        Note that the inputs are keyword arguments to avoid having to specify ``None`` for each of the arguments
        not used.

        Parameters
        ----------
        u: float or None
            Value of :math:`u` to solve for or specify. If left as ``None``, this parameter will be solved for.
            If ``None``, :math:`v` must be specified. Default: ``None``
        v: float or None
            Value of :math:`v` to solve for or specify. If left as ``None``, this parameter will be solved for.
            If ``None``, :math:`u` must be specified. Default: ``None``
        uv_guess: float
            Starting guess for the unsolved :math:`u` or :math:`v` parameter. Default: ``0.5``
        x: Length or None
            :math:`x`-location corresponding to the :math:`u` or :math:`v` parameter to be solved. If this value is
            outside the surface geometry, the root-finder will fail and an error will be raised. If unspecified,
            either :math:`y` or :math:`z` must be specified. Default: ``None``
        y: Length or None
            :math:`y`-location corresponding to the :math:`u` or :math:`v` parameter to be solved. If this value is
            outside the surface geometry, the root-finder will fail and an error will be raised. If unspecified,
            either :math:`x` or :math:`z` must be specified. Default: ``None``
        z: Length or None
            :math:`z`-location corresponding to the :math:`u` or :math:`v` parameter to be solved. If this value is
            outside the surface geometry, the root-finder will fail and an error will be raised. If unspecified,
            either :math:`x` or :math:`y` must be specified. Default: ``None``

        Returns
        -------
        float
            The value of :math:`u` if :math:`v` is specified or :math:`v` if :math:`u` is specified
        """
        # Validate inputs
        if u is None and v is None or (u is not None and v is not None):
            raise ValueError("Must specify exactly one of either u or v")
        xyz_spec = (x is not None, y is not None, z is not None)
        if len([xyz for xyz in xyz_spec if xyz]) != 1:
            raise ValueError("Must specify exactly one of x, y, or z")

        if x is not None:
            xyz, xyz_val = "x", x.m
        elif y is not None:
            xyz, xyz_val = "y", y.m
        elif z is not None:
            xyz, xyz_val = "z", z.m
        else:
            raise ValueError("Did not detect an x, y, or z input")

        def root_find_func_u(u_current):
            point = self.evaluate_point3d(u_current, v)
            return np.array([getattr(point, xyz).m - xyz_val])

        def root_find_func_v(v_current):
            point = self.evaluate_point3d(u, v_current)
            return np.array([getattr(point, xyz).m - xyz_val])

        if v is not None:
            return fsolve(root_find_func_u, x0=np.array([uv_guess]))[0]
        if u is not None:
            return fsolve(root_find_func_v, x0=np.array([uv_guess]))[0]
        raise ValueError("Did not detect a u or v input")

    def split_at_u(self, u0: float) -> ("BezierSurface", "BezierSurface"):
        """
        Splits the Bézier surface at :math:`u=u_0` along the :math:`v`-parametric direction.
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
            [de_casteljau(i=0, j=i, k=k) for i in range(self.n_points_u)] for k in range(self.n_points_v)
        ])
        bez_surf_split_2_P = np.array([
            [de_casteljau(i=i, j=self.degree_u - i, k=k) for i in range(self.n_points_u)] for k in
            range(self.n_points_v)
        ])

        return (
            BezierSurface(
                np.transpose(
                    bez_surf_split_1_P, (1, 0, 2)
                )
            ),
            BezierSurface(
                np.transpose(
                    bez_surf_split_2_P, (1, 0, 2)
                )
            )
        )

    def split_at_v(self, v0: float) -> ("BezierSurface", "BezierSurface"):
        """
        Splits the Bézier surface at :math:`v=v_0` along the :math:`u`-parametric direction.
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
            [de_casteljau(i=0, j=i, k=k) for i in range(self.n_points_v)] for k in range(self.n_points_u)
        ])
        bez_surf_split_2_P = np.array([
            [de_casteljau(i=i, j=self.degree_v - i, k=k) for i in range(self.n_points_v)] for k in
            range(self.n_points_u)
        ])

        return (
            BezierSurface(bez_surf_split_1_P),
            BezierSurface(bez_surf_split_2_P)
        )

    def transform(self, **transformation_kwargs) -> "BezierSurface":
        """
        Creates a transformed copy of the surface by transforming each of the control points

        Parameters
        ----------
        transformation_kwargs
            Keyword arguments passed to :obj:`~aerocaps.geom.transformation.Transformation3D`

        Returns
        -------
        BezierSurface
            Transformed surface
        """
        transformation = Transformation3D(**transformation_kwargs)
        initial_control_points = self.get_control_point_array()
        return BezierSurface(
            np.array([transformation.transform(p_arr) for p_arr in initial_control_points]),
            name=self.name, construction=self.construction
        )

    def generate_control_point_net(self) -> (typing.List[Point3D], typing.List[Line3D]):
        """
        Generates a list of :obj:`~aerocaps.geom.point.Point3D` and :obj:`~aerocaps.geom.curves.Line3D` objects
        representing the Bézier surface's control points and connections between them

        Returns
        -------
        typing.List[Point3D], typing.List[Line3D]
            Control points and lines between adjacent control points in flattened lists
        """
        points = []
        lines = []
        control_points = self.get_control_point_array()

        for i in range(self.n_points_u):
            for j in range(self.n_points_v):
                points.append(Point3D.from_array(control_points[i, j, :]))

        for i in range(self.n_points_u - 1):
            for j in range(self.n_points_v - 1):
                point_obj_1 = Point3D.from_array(control_points[i, j, :])
                point_obj_2 = Point3D.from_array(control_points[i + 1, j, :])
                point_obj_3 = Point3D.from_array(control_points[i, j + 1, :])

                line_1 = Line3D(p0=point_obj_1, p1=point_obj_2)
                line_2 = Line3D(p0=point_obj_1, p1=point_obj_3)
                lines.extend([line_1, line_2])

                if i < self.n_points_u - 2 and j < self.n_points_v - 2:
                    continue

                point_obj_4 = Point3D.from_array(control_points[i + 1, j + 1, :])
                line_3 = Line3D(p0=point_obj_3, p1=point_obj_4)
                line_4 = Line3D(p0=point_obj_2, p1=point_obj_4)
                lines.extend([line_3, line_4])

        return points, lines

    def plot_surface(self, plot: pv.Plotter, Nu: int = 50, Nv: int = 50, **mesh_kwargs) -> pv.StructuredGrid:
        """
        Plots the Bézier surface using the `pyvista <https://pyvista.org/>`_ library

        Parameters
        ----------
        plot:
            :obj:`pyvista.Plotter` instance
        Nu: int
            Number of points to evaluate in the :math:`u`-parametric direction. Default: ``50``
        Nv: int
            Number of points to evaluate in the :math:`v`-parametric direction. Default: ``50``
        mesh_kwargs:
            Keyword arguments to pass to :obj:`pyvista.Plotter.add_mesh`

        Returns
        -------
        pyvista.core.pointset.StructuredGrid
            The evaluated Bézier surface
        """
        XYZ = self.evaluate_grid(Nu, Nv)
        grid = pv.StructuredGrid(XYZ[:, :, 0], XYZ[:, :, 1], XYZ[:, :, 2])
        plot.add_mesh(grid, **mesh_kwargs)

        return grid

    def plot_control_point_mesh_lines(self, plot: pv.Plotter, **line_kwargs) -> pv.Actor:
        """
        Plots the network of lines connecting the Bézier surface control points using the
        `pyvista <https://pyvista.org/>`_ library

        Parameters
        ----------
        plot:
            :obj:`pyvista.Plotter` instance
        line_kwargs:
            Keyword arguments to pass to the :obj:`pyvista.Plotter.add_lines`

        Returns
        -------
        pv.Actor
            The lines actor
        """
        _, line_objs = self.generate_control_point_net()
        line_arr = np.array([[line_obj.p0.as_array(), line_obj.p1.as_array()] for line_obj in line_objs])
        line_arr = line_arr.reshape((len(line_objs) * 2, 3))
        line_actor = plot.add_lines(line_arr, **line_kwargs)
        return line_actor

    def plot_control_points(self, plot: pv.Plotter, **point_kwargs) -> pv.Actor:
        """
        Plots the Bézier surface control points using the `pyvista <https://pyvista.org/>`_ library

        Parameters
        ----------
        plot:
            :obj:`pyvista.Plotter` instance
        point_kwargs:
            Keyword arguments to pass to the :obj:`pyvista.Plotter.add_points`

        Returns
        -------
        pv.Actor
            The points actor
        """
        point_objs, _ = self.generate_control_point_net()
        point_arr = np.array([point_obj.as_array() for point_obj in point_objs])
        point_actor = plot.add_points(point_arr, **point_kwargs)
        return point_actor

    def __repr__(self):
        return (f"{self.name}: {self.degree_u} x {self.degree_v} {self.__class__.__name__} "
                f"({self.degree_u + 1} x {self.degree_v + 1} control points)")


class RationalBezierSurface(Surface):
    """
    Rational Bézier surface class. A NURBS surface with no internal knot vectors.
    """
    def __init__(self,
                 points: typing.List[typing.List[Point3D]] or np.ndarray,
                 weights: np.ndarray,
                 name: str = "RationalBezierSurface",
                 construction: bool = False
                 ):
        """

        Parameters
        ----------
        points
        weights
        name: str
            Name of the geometric object. May be re-assigned a unique name when added to a
            :obj:`~aerocaps.geom.geometry_container.GeometryContainer`. Default: 'RationalBezierSurface'
        construction: bool
            Whether this is a geometry used only for construction of other geometries. If ``True``, this
            geometry will not be exported or plotted. Default: ``False``
        """
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

        self._knots_u = knots_u
        self._knots_v = knots_v
        self.weights = deepcopy(weights)
        super().__init__(name=name, construction=construction)

    @property
    def n_points_u(self) -> int:
        """Number of control points in the :math:`u`-parametric direction"""
        return len(self.points)

    @property
    def n_points_v(self) -> int:
        """Number of control points in the :math:`v`-parametric direction"""
        return len(self.points[0])

    @property
    def degree_u(self) -> int:
        """Surface degree in the :math:`u`-parametric direction"""
        return self.n_points_u - 1

    @property
    def degree_v(self) -> int:
        """Surface degree in the :math:`v`-parametric direction"""
        return self.n_points_v - 1

    @property
    def n(self) -> int:
        """
        Shorthand for :obj:`~aerocaps.geom.surfaces.RationalBezierSurface.degree_u`

        Returns
        -------
        int
            Surface degree in the :math:`u`-parametric direction
        """
        return self.degree_u

    @property
    def m(self) -> int:
        """
        Shorthand for :obj:`~aerocaps.geom.surfaces.RationalBezierSurface.degree_v`

        Returns
        -------
        int
            Surface degree in the :math:`v`-parametric direction
        """
        return self.degree_v

    @property
    def knots_u(self) -> np.ndarray:
        """Knots in the :math:`u`-direction"""
        return self._knots_u

    @property
    def knots_v(self) -> np.ndarray:
        """Knots in the :math:`v`-direction"""
        return self._knots_v

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
        """
        Converts the nested list of control points to a 3-D :obj:`~numpy.ndarray`.

        Returns
        -------
        numpy.ndarray
            3-D array
        """
        return np.array([np.array([p.as_array() for p in p_arr]) for p_arr in self.points])

    def get_homogeneous_control_points(self) -> np.ndarray:
        r"""
        Gets the array of control points in homogeneous coordinates, :math:`\mathbf{P}_{i,j} \cdot w_{i,j}`

        Returns
        -------
        numpy.ndarray
            Array of size :math:`(n + 1) \times (m + 1) \times 4`,
            where :math:`n` is the surface degree in the :math:`u`-direction and :math:`m` is the surface
            degree in the :math:`v`-direction. The four elements of the last array dimension are, in order,
            the :math:`x`-coordinate, :math:`y`-coordinate, :math:`z`-coordinate, and weight of each
            control point.
        """
        return np.dstack((
            self.get_control_point_array() * np.repeat(self.weights[:, :, np.newaxis], 3, axis=2),
            self.weights
        ))

    @staticmethod
    def project_homogeneous_control_points(homogeneous_points: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Projects the homogeneous coordinates onto the :math:`w=1` hyperplane.

        Returns
        -------
        numpy.ndarray, numpy.ndarray
            The projected coordinates in three-dimensional space followed by the weight array
        """
        P = homogeneous_points[:, :, :3] / np.repeat(homogeneous_points[:, :, -1][:, :, np.newaxis], 3, axis=2)
        w = homogeneous_points[:, :, -1]
        return P, w

    def elevate_degree_u(self) -> "RationalBezierSurface":
        """
        Elevates the degree of the rational Bézier surface in the :math:`u`-parametric direction.

        Returns
        -------
        RationalBezierSurface
            A new rational Bézier surface with identical shape to the current one but with one additional row of
            control points in the :math:`u`-parametric direction
        """
        n = self.degree_u
        m = self.degree_v
        Pw = self.get_homogeneous_control_points()

        # New array has one additional control point (current array only has n+1 control points)
        new_Pw = np.zeros((Pw.shape[0] + 1, Pw.shape[1], Pw.shape[2]))

        # Set starting and ending control points to what they already were
        new_Pw[0, :, :] = Pw[0, :, :]
        new_Pw[-1, :, :] = Pw[-1, :, :]

        # Update all the other control points
        for i in range(1, n + 1):  # 1 <= i <= n
            for j in range(0, m + 1):  # for all j
                new_Pw[i, j, :] = i / (n + 1) * Pw[i - 1, j, :] + (1 - i / (n + 1)) * Pw[i, j, :]

        # Extract projected control points and weights from array
        new_P, new_w = self.project_homogeneous_control_points(new_Pw)

        return RationalBezierSurface(new_P, new_w)

    def elevate_degree_v(self) -> "RationalBezierSurface":
        """
        Elevates the degree of the rational Bézier surface in the :math:`v`-parametric direction.

        Returns
        -------
        RationalBezierSurface
            A new rational Bézier surface with identical shape to the current one but with one additional row of
            control points in the :math:`v`-parametric direction
        """
        n = self.degree_u
        m = self.degree_v
        Pw = self.get_homogeneous_control_points()

        # New array has one additional control point (current array only has n+1 control points)
        new_Pw = np.zeros((Pw.shape[0], Pw.shape[1] + 1, Pw.shape[2]))

        # Set starting and ending control points to what they already were
        new_Pw[:, 0, :] = Pw[:, 0, :]
        new_Pw[:, -1, :] = Pw[:, -1, :]

        # Update all the other control points
        for i in range(0, n + 1):  # for all i
            for j in range(1, m + 1):  # 1 <= j <= m
                new_Pw[i, j, :] = j / (m + 1) * Pw[i, j - 1, :] + (1 - j / (m + 1)) * Pw[i, j, :]

        # Extract projected control points and weights from array
        new_P, new_w = self.project_homogeneous_control_points(new_Pw)

        return RationalBezierSurface(new_P, new_w)

    @classmethod
    def from_bezier_revolve(cls, bezier: BezierCurve3D, axis: Line3D, start_angle: Angle, end_angle: Angle):
        """
        Creates a rational Bézier surface from the revolution of a Bézier curve about an axis.

        Parameters
        ----------
        bezier: BezierCurve3D
            Bézier curve to revolve
        axis: Line3D
            Axis of revolution
        start_angle: Angle
            Starting angle for the revolve
        end_angle: Angle
            Ending angle for the revolve

        Returns
        -------
        RationalBezierSurface
            Surface of revolution
        """

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

    @staticmethod
    def fill_surface_from_four_boundaries(left_curve: BezierCurve3D or RationalBezierCurve3D,
                                          right_curve: BezierCurve3D or RationalBezierCurve3D,
                                          top_curve: BezierCurve3D or RationalBezierCurve3D,
                                          bottom_curve: BezierCurve3D or RationalBezierCurve3D,
                                          displacement_degree: int = 3) -> "RationalBezierSurface":
        """
        Creates a fill surface from four boundary curves by linearly interpolating the ``left_curve`` and
        ``right_curve`` and displacing the edges created by the interpolation to form the ``top_curve``
        and ``bottom_curve`` boundaries.

        .. figure:: ../images/fill_surface.*
            :width: 600
            :align: center

        Fill surface from four curve boundaries

        .. warning::

            While this method works for non-planar sets of boundary curves, the primary intended use is for
            co-planar boundary curves. Undesirable surface shapes may result if using non-planar curves.

        Parameters
        ----------
        left_curve: BezierCurve3D or RationalBezierCurve3D
            Left boundary curve
        right_curve: Bezier3D or RationalBezierCurve3D
            Right boundary curve
        top_curve: Bezier3D or RationalBezierCurve3D
            Top boundary curve
        bottom_curve: Bezier3D or RationalBezierCurve3D
            Bottom boundary curve
        displacement_degree: int
            Degree of function used to displace the surface control points to accommodate the top and bottom surfaces.
            Default: 3

        Returns
        -------
        RationalBezierSurface
            Fill surface
        """
        # Convert the boundary curves to rational Bézier curves if they are non-rational
        if isinstance(left_curve, BezierCurve3D):
            left_curve = left_curve.to_rational_bezier_curve()
        if isinstance(right_curve, BezierCurve3D):
            right_curve = right_curve.to_rational_bezier_curve()
        if isinstance(top_curve, BezierCurve3D):
            top_curve = top_curve.to_rational_bezier_curve()
        if isinstance(bottom_curve, BezierCurve3D):
            bottom_curve = bottom_curve.to_rational_bezier_curve()

        # Ensure the boundary curve loop is closed
        left_cps = left_curve.get_homogeneous_control_points()
        right_cps = right_curve.get_homogeneous_control_points()
        top_cps = top_curve.get_homogeneous_control_points()
        bottom_cps = bottom_curve.get_homogeneous_control_points()
        corners = np.array([
            left_cps[0],
            left_cps[-1],
            right_cps[0],
            right_cps[-1],
            top_cps[0],
            top_cps[-1],
            bottom_cps[0],
            bottom_cps[-1]
        ])
        unique_corners = unique_with_tolerance(corners)
        if len(unique_corners) > 4:
            raise ValueError("Boundary curve loop is not closed")

        # Re-orient the curves if necessary
        if not (
                all(np.isclose(left_cps[-1], top_cps[0])) or
                all(np.isclose(left_cps[0], top_cps[0]))
        ):
            top_curve = top_curve.reverse()
            if not (
                    all(np.isclose(left_cps[-1], top_cps[-1])) or
                    all(np.isclose(left_cps[0], top_cps[-1]))
            ):
                raise ValueError("Top curve and left curve are not connected")
        else:
            if not (
                    all(np.isclose(right_cps[-1], top_cps[-1])) or
                    all(np.isclose(right_cps[0], top_cps[-1])) or
                    all(np.isclose(right_cps[-1], top_cps[0])) or
                    all(np.isclose(right_cps[0], top_cps[0]))
            ):
                raise ValueError("Top curve and right curve are not connected")
        if not (
                all(np.isclose(left_cps[-1], bottom_cps[0])) or
                all(np.isclose(left_cps[0], bottom_cps[0]))
        ):
            bottom_curve = bottom_curve.reverse()
            if not (
                    all(np.isclose(left_cps[-1], bottom_cps[-1])) or
                    all(np.isclose(left_cps[0], bottom_cps[-1]))
            ):
                raise ValueError("Bottom curve and left curve are not connected")
        else:
            if not (
                    all(np.isclose(right_cps[-1], bottom_cps[-1])) or
                    all(np.isclose(right_cps[0], bottom_cps[-1])) or
                    all(np.isclose(right_cps[-1], bottom_cps[0])) or
                    all(np.isclose(right_cps[0], bottom_cps[0]))
            ):
                raise ValueError("Bottom curve and right curve are not connected")

        # Elevate the curve degrees of the curve pairs so that they match internally
        if left_curve.degree != right_curve.degree:
            if left_curve.degree < right_curve.degree:
                while left_curve.degree < right_curve.degree:
                    left_curve = left_curve.elevate_degree()
            else:
                while right_curve.degree < left_curve.degree:
                    right_curve = right_curve.elevate_degree()
        if top_curve.degree != bottom_curve.degree:
            if top_curve.degree < bottom_curve.degree:
                while top_curve.degree < bottom_curve.degree:
                    top_curve = top_curve.elevate_degree()
            else:
                while bottom_curve.degree < top_curve.degree:
                    bottom_curve = bottom_curve.elevate_degree()

        # Retrieve the new homogeneous control points for each (possibly modified) curve
        left_cps = left_curve.get_homogeneous_control_points()
        right_cps = right_curve.get_homogeneous_control_points()
        top_cps = top_curve.get_homogeneous_control_points()
        bottom_cps = bottom_curve.get_homogeneous_control_points()

        # Linearly interpolate between the left and right curves, length equal to number of control points
        # in the top/bottom direction
        arrays_to_stack = []
        for t in np.linspace(0.0, 1.0, top_curve.degree + 1):
            arrays_to_stack.append(left_cps + t * (right_cps - left_cps))
        Pw = np.array(arrays_to_stack)

        # Displace the control points associated with the top side of the surface
        for Pw_slice, curve_Pw in zip(Pw[1:-1], bottom_cps[1:-1]):
            displacement = curve_Pw - Pw_slice[0]
            for i in range(left_curve.degree):
                Pw_slice[i] += ((left_curve.degree - i) / left_curve.degree) ** displacement_degree * displacement

        # Displace the control points associated with the bottom side of the surface
        for Pw_slice, curve_Pw in zip(Pw[1:-1], top_cps[1:-1]):
            displacement = curve_Pw - Pw_slice[-1]
            for i in range(left_curve.degree):
                Pw_slice[-1 - i] += ((left_curve.degree - i) / left_curve.degree) ** displacement_degree * displacement

        # Project the new homogeneous control points onto the w=1 hyperplane
        new_points, new_weights = RationalBezierSurface.project_homogeneous_control_points(Pw)

        return RationalBezierSurface(new_points, new_weights)

    def evaluate(self, u: float, v: float) -> np.ndarray:
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
        return np.array(rational_bezier_surf_eval(P, self.weights, u, v))

    def evaluate_point3d(self, u: float, v: float) -> Point3D:
        r"""
        Evaluates the rational Bézier surface at a single :math:`(u,v)` parameter pair and returns a point object.

        Parameters
        ----------
        u: float
            Position along :math:`u` in parametric space. Normally in the range :math:`[0,1]`
        v: float
            Position along :math:`v` in parametric space. Normally in the range :math:`[0,1]`

        Returns
        -------
        Point3D
            Point object corresponding to the :math:`(u,v)` pair
        """
        return Point3D.from_array(self.evaluate(u, v))

    def evaluate_grid(self, Nu: int, Nv: int) -> np.ndarray:
        r"""
        Evaluates the rational Bézier surface on a uniform :math:`N_u \times N_v` grid of parameter values.

        Parameters
        ----------
        Nu: int
            Number of uniformly spaced parameter values in the :math:`u`-direction
        Nv: int
            Number of uniformly spaced parameter values in the :math:`v`-direction

        Returns
        -------
        numpy.ndarray
            Array of size :math:`N_u \times N_v \times 3`
        """
        P = self.get_control_point_array()
        return np.array(rational_bezier_surf_eval_grid(P, self.weights, Nu, Nv))

    def extract_edge_curve(self, surface_edge: SurfaceEdge) -> RationalBezierCurve3D:
        """
        Extracts the control points and weights from one of the four edges of the rational Bézier surface and
        outputs a rational Bézier curve with these control points and weights

        Parameters
        ----------
        surface_edge: SurfaceEdge
            Edge along which to extract the curve

        Returns
        -------
        RationalBezierCurve3D
            Rational Bézier curve with control points corresponding to the control points along the edge of the surface
        """
        P = self.get_control_point_array()
        w = self.weights

        if surface_edge == SurfaceEdge.u0:
            return RationalBezierCurve3D(P[0, :, :], w[0, :])
        if surface_edge == SurfaceEdge.u1:
            return RationalBezierCurve3D(P[-1, :, :], w[-1, :])
        if surface_edge == SurfaceEdge.v0:
            return RationalBezierCurve3D(P[:, 0, :], w[:, 0])
        if surface_edge == SurfaceEdge.v1:
            return RationalBezierCurve3D(P[:, -1, :], w[:, -1])

        raise ValueError(f"Invalid surface edge {surface_edge}")

    def reverse_u(self) -> "RationalBezierSurface":
        """Reverses the surface in the :math:`u`-direction"""
        P = self.get_control_point_array()[::-1, :, :]
        w = self.weights[::-1, :]
        return RationalBezierSurface(P, w)

    def reverse_v(self) -> "RationalBezierSurface":
        """Reverses the surface in the :math:`v`-direction"""
        P = self.get_control_point_array()[:, ::-1, :]
        w = self.weights[:, ::-1]
        return RationalBezierSurface(P, w)

    def get_parallel_degree(self, surface_edge: SurfaceEdge):
        r"""
        Gets the degree of the curve corresponding to the input surface edge.

        Parameters
        ----------
        surface_edge: SurfaceEdge
            Edge along which the parallel degree is evaluated

        Returns
        -------
        int
            Degree parallel to the edge
        """
        if surface_edge in [SurfaceEdge.v1, SurfaceEdge.v0]:
            return self.degree_u
        return self.degree_v

    def get_perpendicular_degree(self, surface_edge: SurfaceEdge):
        r"""
        Gets the degree of the curve in the parametric direction perpendicular to the input surface edge.

        Parameters
        ----------
        surface_edge: SurfaceEdge
            Edge along which the perpendicular degree is evaluated

        Returns
        -------
        int
            Degree perpendicular to the edge
        """
        if surface_edge in [SurfaceEdge.v1, SurfaceEdge.v0]:
            return self.degree_v
        return self.degree_u

    def get_parallel_n_points(self, surface_edge: SurfaceEdge) -> int:
        r"""
        Gets the number of control points in the parametric direction parallel to the input surface edge.

        Parameters
        ----------
        surface_edge: SurfaceEdge
            Edge along which the parallel number of control points is evaluated

        Returns
        -------
        int
            Number of control points parallel to the edge
        """
        if surface_edge in (SurfaceEdge.v1, SurfaceEdge.v0):
            return self.n_points_u
        return self.n_points_v

    def get_perpendicular_n_points(self, surface_edge: SurfaceEdge) -> int:
        r"""
        Gets the number of control points in the parametric direction perpendicular to the input surface edge.

        Parameters
        ----------
        surface_edge: SurfaceEdge
            Edge along which the perpendicular number of control points is evaluated

        Returns
        -------
        int
            Number of control points perpendicular to the edge
        """
        if surface_edge in (SurfaceEdge.v1, SurfaceEdge.v0):
            return self.n_points_v
        return self.n_points_u

    def get_corner_index(self, surface_corner: SurfaceCorner) -> (int, int):
        """
        Gets the :math:`i`- and :math:`j`-indices of the control point corresponding to the input corner

        Parameters
        ----------
        surface_corner: SurfaceCorner
            Corner from which to retrieve the index

        Returns
        -------
        int, int
            :math:`i`-index and :math:`j`-index, respectively
        """
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
        r"""
        Gets the point corresponding to a particular index along the edge curve with perpendicular index
        corresponding to the level of continuity being applied. For example, for a :math:`6 \times 5` rational
        Bézier surface, the following code

        .. code-block:: python

            p = surf.get_point(2, 1, ac.SurfaceEdge.v0)

        returns the point :math:`\mathbf{P}_{2,1}` and

        .. code-block:: python

            p = surf.get_point(2, 1, ac.SurfaceEdge.u1)

        returns the point :math:`\mathbf{P}_{6-1,2} = \mathbf{P}_{5,2}`.

        .. seealso::

            :obj:`~aerocaps.geom.surfaces.RationalBezierSurface.set_point`
                Setter equivalent of this method

        Parameters
        ----------
        row_index: int
            Index along the surface edge control points
        continuity_index: int
            Index in the parametric direction perpendicular to the surface edge. Normally either ``0``, ``1``, or ``2``
        surface_edge: SurfaceEdge
            Edge of the surface along which to retrieve the control point

        Returns
        -------
        Point3D
            Point used to enforce :math:`G^x` continuity, where :math:`x` is the value of ``continuity_index``
        """
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

    def get_point_ij(self, row_index: int, continuity_index: int, surface_edge: SurfaceEdge) -> (int, int):
        r"""
        Gets the point indices corresponding to a particular index along the edge curve with perpendicular index
        corresponding to the level of continuity being applied.

        Parameters
        ----------
        row_index: int
            Index along the surface edge control points
        continuity_index: int
            Index in the parametric direction perpendicular to the surface edge. Normally either ``0``, ``1``, or ``2``
        surface_edge: SurfaceEdge
            Edge of the surface along which to retrieve the control point

        Returns
        -------
        int, int
            Point indices used to enforce :math:`G^x` continuity, where :math:`x` is the value of ``continuity_index``
        """
        if surface_edge == SurfaceEdge.v1:
            return row_index, len(self.points[0]) - (continuity_index + 1)
        elif surface_edge == SurfaceEdge.v0:
            return row_index, continuity_index
        elif surface_edge == SurfaceEdge.u1:
            return len(self.points) - (continuity_index + 1), row_index
        elif surface_edge == SurfaceEdge.u0:
            return continuity_index, row_index
        else:
            raise ValueError("Invalid surface_edge value")

    def set_point(self, point: Point3D, row_index: int, continuity_index: int, surface_edge: SurfaceEdge):
        r"""
        Sets the point corresponding to a particular index along the edge curve with perpendicular index
        corresponding to the level of continuity being applied. For example, for a :math:`6 \times 5`
        rational Bézier surface, the following code

        .. code-block:: python

            p = ac.Point3D.from_array(np.array([3.0, 4.0, 5.0]))
            surf.set_point(p, 2, 1, ac.SurfaceEdge.v0)

        sets the value of point :math:`\mathbf{P}_{2,1}` to :math:`[3,4,5]^T` and

        .. code-block:: python

            p = ac.Point3D.from_array(np.array([3.0, 4.0, 5.0]))
            surf.set_point(p, 2, 1, ac.SurfaceEdge.u1)

        sets the value of point :math:`\mathbf{P}_{6-1,2} = \mathbf{P}_{5,2}` to :math:`[3,4,5]^T`.

        .. seealso::

            :obj:`~aerocaps.geom.surfaces.RationalBezierSurface.get_point`
                Getter equivalent of this method

        Parameters
        ----------
        point: Point3D
            Point object to apply at the specified indices
        row_index: int
            Index along the surface edge control points
        continuity_index: int
            Index in the parametric direction perpendicular to the surface edge. Normally either ``0``, ``1``, or ``2``
        surface_edge: SurfaceEdge
            Edge of the surface along which to retrieve the control point
        """
        if surface_edge == SurfaceEdge.v1:
            self.points[row_index][-(continuity_index + 1)].x.m = point.x.m
            self.points[row_index][-(continuity_index + 1)].y.m = point.y.m
            self.points[row_index][-(continuity_index + 1)].z.m = point.z.m
        elif surface_edge == SurfaceEdge.v0:
            self.points[row_index][continuity_index].x.m = point.x.m
            self.points[row_index][continuity_index].y.m = point.y.m
            self.points[row_index][continuity_index].z.m = point.z.m
        elif surface_edge == SurfaceEdge.u1:
            self.points[-(continuity_index + 1)][row_index].x.m = point.x.m
            self.points[-(continuity_index + 1)][row_index].y.m = point.y.m
            self.points[-(continuity_index + 1)][row_index].z.m = point.z.m
        elif surface_edge == SurfaceEdge.u0:
            self.points[continuity_index][row_index].x.m = point.x.m
            self.points[continuity_index][row_index].y.m = point.y.m
            self.points[continuity_index][row_index].z.m = point.z.m
        else:
            raise ValueError("Invalid surface_edge value")

    def get_weight(self, row_index: int, continuity_index: int, surface_edge: SurfaceEdge) -> float:
        r"""
        Gets the weight corresponding to a particular index along the edge curve with perpendicular index
        corresponding to the level of continuity being applied. For example, for a :math:`6 \times 5` rational
        Bézier surface, the following code

        .. code-block:: python

            w = surf.get_weight(2, 1, ac.SurfaceEdge.v0)

        returns the weight :math:`w_{2,1}` and

        .. code-block:: python

            w = surf.get_weight(2, 1, ac.SurfaceEdge.u1)

        returns the weight :math:`w_{6-1,2} = w_{5,2}`.

        .. seealso::

            :obj:`~aerocaps.geom.surfaces.RationalBezierSurface.set_weight`
                Setter equivalent of this method

        Parameters
        ----------
        row_index: int
            Index along the surface edge weights
        continuity_index: int
            Index in the parametric direction perpendicular to the surface edge. Normally either ``0``, ``1``, or ``2``
        surface_edge: SurfaceEdge
            Edge of the surface along which to retrieve the weight

        Returns
        -------
        float
            Weight used to enforce :math:`G^x` continuity, where :math:`x` is the value of ``continuity_index``
        """
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
        r"""
        Sets the weight corresponding to a particular index along the edge curve with perpendicular index
        corresponding to the level of continuity being applied. For example, for a :math:`6 \times 5`
        rational Bézier surface, the following code

        .. code-block:: python

            surf.set_weight(0.9, 2, 1, ac.SurfaceEdge.v0)

        sets the value of weight :math:`w_{2,1}` to :math:`0.9` and

        .. code-block:: python

            surf.set_weight(1.1, 2, 1, ac.SurfaceEdge.u1)

        sets the value of weight :math:`w_{6-1,2} = w_{5,2}` to :math:`1.1`.

        .. seealso::

            :obj:`~aerocaps.geom.surfaces.RationalBezierSurface.get_weight`
                Getter equivalent of this method

        Parameters
        ----------
        weight: float
            Weight to apply at the specified indices
        row_index: int
            Index along the surface edge weights
        continuity_index: int
            Index in the parametric direction perpendicular to the surface edge. Normally either ``0``, ``1``, or ``2``
        surface_edge: SurfaceEdge
            Edge of the surface along which to retrieve the weight
        """
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

    @staticmethod
    def _evaluate_f_sign(surf_edge_1: SurfaceEdge, surf_edge_2: SurfaceEdge) -> float:
        """
        Evaluates the sign of the tangent proportionality factor across an edge pair

        Parameters
        ----------
        surf_edge_1: SurfaceEdge
            First surface edge
        surf_edge_2: SurfaceEdge
            Second surface edge

        Returns
        -------
        float
            ``-1.0`` if both surface edges end in 0 or both surface edges end in 1, ``1.0`` otherwise
        """
        surf_edges_0 = (SurfaceEdge.u0, SurfaceEdge.v0)
        surf_edges_1 = (SurfaceEdge.u1, SurfaceEdge.v1)
        if surf_edge_1 in surf_edges_0 and surf_edge_2 in surf_edges_0:
            return -1.0
        if surf_edge_1 in surf_edges_1 and surf_edge_2 in surf_edges_1:
            return -1.0
        return 1.0

    def enforce_g0(self, other: "RationalBezierSurface",
                   surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge):
        r"""
        Enforces :math:`G^0` continuity along the input ``surface_edge`` by equating the control points
        and weights along this edge to the corresponding control points and weights along the ``other_surface_edge``
        of the rational Bézier surface given by ``other``.
        The control points of the surface from which this method is called are modified in-place, and the control
        points of ``other`` are left unchanged.

        .. important::

            The parallel degree of the current surface along ``surface_edge`` must be equal to the parallel degree
            of the ``other`` surface along ``other_surface_edge``, otherwise an ``AssertionError`` will be raised.
            If these degrees are not equal, first elevate the degree of the surface with the lower parallel degree
            until the degrees match using either :obj:`~aerocaps.geom.surfaces.RationalBezierSurface.elevate_degree_u`
            or :obj:`~aerocaps.geom.surfaces.RationalBezierSurface.elevate_degree_v`, whichever is appropriate.

        .. seealso::

            :obj:`~aerocaps.geom.surfaces.RationalBezierSurface.enforce_c0`
                Parametric continuity equivalent (:math:`C^0`)

        Parameters
        ----------
        other: RationalBezierSurface
            Another rational Bézier surface along which an edge will be used for stitching
        surface_edge: SurfaceEdge
            The edge of the current surface to modify
        other_surface_edge: SurfaceEdge
            Tool edge of surface ``other`` which determines the positions of control points along ``surface_edge``
            of the current surface
        """
        # P^b[:, 0] = P^a[:, -1]
        self_parallel_degree = self.get_parallel_degree(surface_edge)
        other_parallel_degree = other.get_parallel_degree(other_surface_edge)
        if self_parallel_degree != other_parallel_degree:
            raise ValueError(f"Degree parallel to the edge of the input surface ({self_parallel_degree}) does "
                             f"not match the degree parallel to the edge of the other surface "
                             f"({other_parallel_degree})")
        for row_index in range(self.get_parallel_degree(surface_edge) + 1):
            self.set_point(other.get_point(row_index, 0, other_surface_edge), row_index, 0, surface_edge)
            self.set_weight(other.get_weight(row_index, 0, other_surface_edge), row_index, 0, surface_edge)

    def enforce_c0(self, other: "RationalBezierSurface", surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge):
        """
        For zeroth-degree continuity, there is no difference between geometric (:math:`G^0`) and parametric
        (:math:`C^0`) continuity. Because this method is simply a convenience method that calls
        :obj:`~aerocaps.geom.surfaces.RationalBezierSurface.enforce_g0`, see the documentation for that method for more
        detailed documentation.

        Parameters
        ----------
        other: RationalBezierSurface
            Another rational Bézier surface along which an edge will be used for stitching
        surface_edge: SurfaceEdge
            The edge of the current surface to modify
        other_surface_edge: SurfaceEdge
            Tool edge of surface ``other`` which determines the positions of control points along ``surface_edge``
            of the current surface
        """
        self.enforce_g0(other, surface_edge, other_surface_edge)

    def enforce_g0g1(self, other: "RationalBezierSurface", f: float or np.ndarray,
                     surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge):
        r"""
        First enforces :math:`G^0` continuity, then tangent (:math:`G^1`) continuity is enforced according to
        the following equations:

        .. math::

            \mathcal{W}^{b,\mathcal{E}_b}_{k,1} = \mathcal{W}^{b,\mathcal{E}_b}_{k,0} + f \frac{p_{\perp}^{a,\mathcal{E}_a}}{p_{\perp}^{b,\mathcal{E}_b}} \left( \mathcal{W}^{a,\mathcal{E}_a}_{k,0} - \mathcal{W}^{a,\mathcal{E}_a}_{k,1} \right) \text{ for }k=0,1,\ldots,p_{\parallel}^{b,\mathcal{E}_b}

        .. math::

            \mathcal{P}^{b,\mathcal{E}_b}_{k,1} = \frac{\mathcal{W}^{b,\mathcal{E}_b}_{k,0}}{\mathcal{W}^{b,\mathcal{E}_b}_{k,1}} \mathcal{P}^{b,\mathcal{E}_b}_{k,0} + f \frac{p_{\perp}^{a,\mathcal{E}_a}}{p_{\perp}^{b,\mathcal{E}_b}} \frac{1}{\mathcal{W}^{b,\mathcal{E}_b}_{k,1}} \left[\mathcal{W}^{a,\mathcal{E}_a}_{k,0} \mathcal{P}^{a,\mathcal{E}_a}_{k,0} - \mathcal{P}^{a,\mathcal{E}_a}_{k,1} \mathcal{W}^{a,\mathcal{E}_a}_{k,1} \right] \text{ for }k=0,1,\ldots,p_{\parallel}^{b,\mathcal{E}_b}

        Here, :math:`b` corresponds to the current surface, and :math:`a` corresponds to the ``other`` surface.
        The control points of the surface from which this method is called are modified in-place, and the control
        points of ``other`` are left unchanged.

        .. seealso::

            :obj:`~aerocaps.geom.surfaces.RationalBezierSurface.enforce_g0`
                Geometric point continuity enforcement (:math:`G^0`)
            :obj:`~aerocaps.geom.surfaces.RationalBezierSurface.enforce_c0c1`
                Parametric continuity equivalent (:math:`C^1`)

        Parameters
        ----------
        other: RationalBezierSurface
            Another rational Bézier surface along which an edge will be used for stitching
        f: float
            Tangent proportionality factor
        surface_edge: SurfaceEdge
            The edge of the current surface to modify
        other_surface_edge: SurfaceEdge
            Tool edge of surface ``other`` which determines the positions of control points along ``surface_edge``
            of the current surface
        """
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
        r"""
        Equivalent to calling :obj:`~aerocaps.geom.surfaces.RationalBezierSurface.enforce_g0g1` with ``f=1.0``. See that
        method for more detailed documentation.

        Parameters
        ----------
        other: RationalBezierSurface
            Another rational Bézier surface along which an edge will be used for stitching
        surface_edge: SurfaceEdge
            The edge of the current surface to modify
        other_surface_edge: SurfaceEdge
            Tool edge of surface ``other`` which determines the positions of control points along ``surface_edge``
            of the current surface
        """
        self.enforce_g0g1(other, 1.0, surface_edge, other_surface_edge)

    def enforce_g0g1g2(self, other: "RationalBezierSurface", f: float or np.ndarray,
                       surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge):
        r"""
        First enforces :math:`G^0` and :math:`G^1` continuity, then curvature (:math:`G^2`) continuity is enforced
        according to the following equations:

        .. math::

            \mathcal{W}^{b,\mathcal{E}_b}_{k,2} = 2 \mathcal{W}^{b,\mathcal{E}_b}_{k,1} - \mathcal{W}^{b,\mathcal{E}_b}_{k,0} + f^2 \frac{p_{\perp}^{a,\mathcal{E}_a}(p_{\perp}^{a,\mathcal{E}_a}-1)}{p_{\perp}^{b,\mathcal{E}_b}(p_{\perp}^{b,\mathcal{E}_b}-1)} \left[ \mathcal{W}^{a,\mathcal{E}_a}_{k,0} - 2 \mathcal{W}^{a,\mathcal{E}_a}_{k,1} + \mathcal{W}^{a,\mathcal{E}_a}_{k,2} \right]  \text{ for }k=0,1,\ldots,p_{\parallel}^{b,\mathcal{E}_b}

        .. math::

            \mathcal{P}^{b,\mathcal{E}_b}_{k,2} = 2 \frac{\mathcal{W}^{b,\mathcal{E}_b}_{k,1}}{\mathcal{W}^{b,\mathcal{E}_b}_{k,2}} \mathcal{P}^{b,\mathcal{E}_b}_{k,1} - \frac{\mathcal{W}^{b,\mathcal{E}_b}_{k,0}}{\mathcal{W}^{b,\mathcal{E}_b}_{k,2}} \mathcal{P}^{b,\mathcal{E}_b}_{k,0} + f^2 \frac{p_{\perp}^{a,\mathcal{E}_a}(p_{\perp}^{a,\mathcal{E}_a}-1)}{p_{\perp}^{b,\mathcal{E}_b}(p_{\perp}^{b,\mathcal{E}_b}-1)} \frac{1}{\mathcal{W}^{b,\mathcal{E}_b}_{k,2}} \left[ \mathcal{W}^{a,\mathcal{E}_a}_{k,1} \mathcal{P}^{a,\mathcal{E}_a}_{k,0} - 2 \mathcal{W}^{a,\mathcal{E}_a}_{k,1} \mathcal{P}^{a,\mathcal{E}_a}_{k,1} + \mathcal{W}^{a,\mathcal{E}_a}_{k,2} \mathcal{P}^{a,\mathcal{E}_a}_{k,2} \right]  \text{ for }k=0,1,\ldots,p_{\parallel}^{b,\mathcal{E}_b}

        Here, :math:`b` corresponds to the current surface, and :math:`a` corresponds to the ``other`` surface.
        The control points of the surface from which this method is called are modified in-place, and the control
        points of ``other`` are left unchanged.

        .. seealso::

         :obj:`~aerocaps.geom.surfaces.RationalBezierSurface.enforce_g0`
             Geometric point continuity enforcement (:math:`G^0`)
         :obj:`~aerocaps.geom.surfaces.RationalBezierSurface.enforce_g0g1`
             Geometric tangent continuity enforcement (:math:`G^1`)
         :obj:`~aerocaps.geom.surfaces.RationalBezierSurface.enforce_c0c1c2`
             Parametric continuity equivalent (:math:`C^2`)

        Parameters
        ----------
        other: RationalBezierSurface
            Another rational Bézier surface along which an edge will be used for stitching
        f: float
            Tangent proportionality factor
        surface_edge: SurfaceEdge
            The edge of the current surface to modify
        other_surface_edge: SurfaceEdge
            Tool edge of surface ``other`` which determines the positions of control points along ``surface_edge``
            of the current surface
        """
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
        r"""
        Equivalent to calling :obj:`~aerocaps.geom.surfaces.RationalBezierSurface.enforce_g0g1g2` with ``f=1.0``.
        See that method for more detailed documentation.

        Parameters
        ----------
        other: RationalBezierSurface
            Another rational Bézier surface along which an edge will be used for stitching
        surface_edge: SurfaceEdge
            The edge of the current surface to modify
        other_surface_edge: SurfaceEdge
            Tool edge of surface ``other`` which determines the positions of control points along ``surface_edge``
            of the current surface
        """
        self.enforce_g0g1g2(other, 1.0, surface_edge, other_surface_edge)

    def enforce_g0g1g2_multiface(self,
                                 adjacent_surf_u0: "RationalBezierSurface" = None,
                                 adjacent_surf_u1: "RationalBezierSurface" = None,
                                 adjacent_surf_v0: "RationalBezierSurface" = None,
                                 adjacent_surf_v1: "RationalBezierSurface" = None,
                                 other_edge_u0: SurfaceEdge = None,
                                 other_edge_u1: SurfaceEdge = None,
                                 other_edge_v0: SurfaceEdge = None,
                                 other_edge_v1: SurfaceEdge = None,
                                 f_u0_initial: float = 1.0,
                                 f_u1_initial: float = 1.0,
                                 f_v0_initial: float = 1.0,
                                 f_v1_initial: float = 1.0,
                                 n_deriv_points: int = 10,
                                 ) -> OptimizeResult:
        r"""
        .. warning::

            This is an experimental feature and should not be used in production geometries

        Enforces :math:`G^0`, :math:`G^1`, and :math:`G^2` continuity across multiple adjacent boundaries of a surface,
        up to all four boundaries. This is done by first enforcing :math:`G^0` continuity at all required boundaries
        and then optimizing the locations of the second and third rows of control points to minimize
        :math:`G^1` and :math:`G^2` error at ``n_deriv_points`` locations along each of the boundary curves.
        The following is the cost function that is minimized:

        .. math::

            J(x_k) = \sum\limits_{l=1}^2 \sum\limits_{i=0}^{n_p n_{\mathcal{E}}} \left( \left. \frac{\partial^l \mathbf{S}_i^a(u,v)}{\partial \mu^l} \right|_{u=u_i,v=v_i} - \frac{f_{\text{sgn},i}}{f^l_i(x_k)}  \left. \frac{\partial^l \mathbf{S}^b(u,v,x_k)}{\partial \mu^l} \right|_{u=u_i,v=v_i} \right)^2

        where

        * :math:`x_k` is the set of design variables to be optimized including the internal control point locations and tangent proportionality factors across each boundary
        * :math:`n_p` is the number of discrete derivative calculations on each boundary (specified by ``n_deriv_points``)
        * :math:`n_\mathcal{E}` is the number of edges across which continuity is being enforced
        * :math:`\mathbf{S}_i^a(u,v)` is a surface specified by ``adjacent_surf_u0``, etc.
        * :math:`\mu` is equal to either :math:`u` or :math:`v`, determined by the parametric direction perpendicular to the edge
        * :math:`(u_i,v_i)` is the point along an edge where the derivative is being evaluated
        * :math:`f_{\text{sgn},i}` is the sign of the proportionality factor, :math:`-1` if both the target edge and other edge specified by :math:`i` end in :math:`0` or both end in :math:`1`, :math:`1` otherwise
        * :math:`f_i(x_k)` is the tangent proportionality factor
        * :math:`\mathbf{S}^b(u,v,x_k)` is the target surface (``self``)
        * :math:`l` is the derivative order

        For maximum performance of the optimizer, the exact Jacobian is calculated:

        .. math::

            \frac{\partial J(x_k)}{\partial x_k} = 2 \sum\limits_{l=1}^2 \sum\limits_{i=0}^{n_p n_{\mathcal{E}}} \left( \left. \frac{\partial^l \mathbf{S}_i^a(u,v)}{\partial \mu^l} \right|_{u=u_i,v=v_i} - \frac{f_{\text{sgn},i}}{f^l_i(x_k)}  \left. \frac{\partial^l \mathbf{S}^b(u,v,x_k)}{\partial \mu^l} \right|_{u=u_i,v=v_i} \right) \left[ -f_{\text{sgn},i} \frac{\partial}{\partial x_k} \left( \frac{1}{f^l_i} \right) \left( \left. \frac{\partial^l \mathbf{S}^b(u,v,x_k)}{\partial \mu^l} \right|_{u=u_i,v=v_i} \right) -\frac{f_{\text{sgn},i}}{f_i^l} \frac{\partial}{\partial x_k} \left(  \left. \frac{\partial^l \mathbf{S}^b(u,v,x_k)}{\partial \mu^l} \right|_{u=u_i,v=v_i} \right) \right]

        .. note::

            This method is reserved for the complex case where continuity is required across boundaries that
            share a surface corner. In the case of continuity with a single surface or a pair of surfaces
            with common boundaries on opposite sides of the surface (such as the :math:`v_0` and :math:`v_1`
            boundaries), the much simpler :obj:`~aerocaps.geom.surfaces.BezierSurface.enforce_g0g1g2` should
            be used.

        .. figure:: ../images/bezier_enforce_g0g1g2_multiface.*
            :align: center
            :width: 600

            Multi-face :math:`G^0`, :math:`G^1`, and :math:`G^2` continuity enforcement

        Parameters
        ----------
        adjacent_surf_u0: BezierSurface
            Surface sharing the :math:`u_0` boundary of ``target_surf``. Default: ``None``
        adjacent_surf_u1: BezierSurface
            Surface sharing the :math:`u_1` boundary of ``target_surf``. Default: ``None``
        adjacent_surf_v0: BezierSurface
            Surface sharing the :math:`v_0` boundary of ``target_surf``. Default: ``None``
        adjacent_surf_v1: BezierSurface
            Surface sharing the :math:`v_1` boundary of ``target_surf``. Default: ``None``
        other_edge_u0: SurfaceEdge
            Edge of surface ``adjacent_surf_u0`` that will be stitched. Default: ``None``
        other_edge_u1: SurfaceEdge
            Edge of surface ``adjacent_surf_u0`` that will be stitched. Default: ``None``
        other_edge_v0: SurfaceEdge
            Edge of surface ``adjacent_surf_u0`` that will be stitched. Default: ``None``
        other_edge_v1: SurfaceEdge
            Edge of surface ``adjacent_surf_u0`` that will be stitched. Default: ``None``
        f_u0_initial: float
            Initial value of the tangent proportionality factor across boundary :math:`u_0`. The final value
            selected by the optimizer will be different from this value. Default: ``1.0``
        f_u1_initial: float
            Initial value of the tangent proportionality factor across boundary :math:`u_1`. The final value
            selected by the optimizer will be different from this value. Default: ``1.0``
        f_v0_initial: float
            Initial value of the tangent proportionality factor across boundary :math:`v_0`. The final value
            selected by the optimizer will be different from this value. Default: ``1.0``
        f_v1_initial: float
            Initial value of the tangent proportionality factor across boundary :math:`v_1`. The final value
            selected by the optimizer will be different from this value. Default: ``1.0``
        n_deriv_points: int
            Number of discrete locations where the continuity error will be evaluated. Default: ``10``

        Returns
        -------
        OptimizeResult
            Result from the :math:`G^1`- and :math:`G^2`-continuity error minimization problem solution
        """
        adjacent_surfs = (adjacent_surf_u0, adjacent_surf_u1, adjacent_surf_v0, adjacent_surf_v1)
        other_edges = (other_edge_u0, other_edge_u1, other_edge_v0, other_edge_v1)
        # Input validation
        if not any(adjacent_surfs):
            raise ValueError("Must specify at least one adjacent surface")
        if not any(other_edges):
            raise ValueError("Must specify at least one other edge")
        if len(adjacent_surfs) == 1:
            raise ValueError("For continuity enforcement with only one other surface, use 'enforce_g0g1' instead")
        if len(adjacent_surfs) != len(other_edges):
            raise ValueError("Must specify one 'other_edge' for every 'adjacent_surf'")

        # Create a mapping between the surfaces and edges
        surf_edge_mapping = {
            SurfaceEdge.u0: (adjacent_surf_u0, other_edge_u0, f_u0_initial),
            SurfaceEdge.u1: (adjacent_surf_u1, other_edge_u1, f_u1_initial),
            SurfaceEdge.v0: (adjacent_surf_v0, other_edge_v0, f_v0_initial),
            SurfaceEdge.v1: (adjacent_surf_v1, other_edge_v1, f_v1_initial)
        }
        for self_edge, other_data in surf_edge_mapping.items():
            if any(other_data) or all(other_data):
                continue
            raise ValueError("Must specify either both an 'adjacent_surf' and an 'other_edge' or neither for every "
                             "edge of the current surface")

        # Enforce G0 continuity with all surfaces
        for self_edge in surf_edge_mapping.keys():
            data = surf_edge_mapping[self_edge]
            if data[0] is None:
                continue
            self.enforce_g0(
                data[0], surface_edge=self_edge, other_surface_edge=data[1]
            )

        d1_other = {
            self_edge: data[0].get_first_derivs_along_edge(data[1], n_points=n_deriv_points) if data[0] else None
            for self_edge, data in surf_edge_mapping.items()
        }
        d2_other = {
            self_edge: data[0].get_second_derivs_along_edge(data[1], n_points=n_deriv_points) if data[0] else None
            for self_edge, data in surf_edge_mapping.items()
        }

        def get_point_ijs_to_update() -> typing.List[typing.Tuple[int]]:
            """Gets the indices of the points in the target surface that will be updated during the optimization"""
            point_ijs_to_update = []
            for surface_edge, _data in surf_edge_mapping.items():
                # Loop through all the points in the second row starting from the second point and ending at the
                # second-to-last point
                for row_index in range(1, self.get_parallel_n_points(surface_edge) - 1):
                    point_ij = self.get_point_ij(row_index, continuity_index=1, surface_edge=surface_edge)
                    if point_ij in point_ijs_to_update:
                        continue
                    point_ijs_to_update.append(point_ij)
            for surface_edge, _data in surf_edge_mapping.items():
                for row_index in range(1, self.get_parallel_n_points(surface_edge) - 1):
                    point_ij_2 = self.get_point_ij(row_index, continuity_index=2, surface_edge=surface_edge)
                    if point_ij_2 in point_ijs_to_update:
                        continue
                    point_ijs_to_update.append(point_ij_2)
            return point_ijs_to_update

        f_signs = {self_edge: self._evaluate_f_sign(self_edge, data[1])
                   for self_edge, data in surf_edge_mapping.items() if data is not None}
        f_vals = {self_edge: data[2] for self_edge, data in surf_edge_mapping.items() if data is not None}
        mod_ijs = get_point_ijs_to_update()
        mod_points = [self.points[i][j] for i, j in mod_ijs]
        x0 = np.array([p.as_array() for p in mod_points]).flatten()
        x0 = np.append(x0, np.array(list(f_vals.values())))

        def obj_fun_and_jac(x: np.ndarray) -> (float, np.ndarray):
            """
            Computes the objective function as the sum of the squares of the :math:`G^1` continuity error, along
            with the Jacobian

            Parameters
            ----------
            x: np.ndarray
                1-D array of design variable values

            Returns
            -------
            float, np.ndarray
                The objective function value and the Jacobian (a 1-D array of sensitivities)
            """
            x_reshaped = x[:3 * len(mod_ijs)].reshape((len(mod_points), 3))
            jac_arr = np.zeros(x.shape)
            # Update the points in-place
            for i in range(x_reshaped.shape[0]):
                mod_points[i].x.m = x_reshaped[i, 0]
                mod_points[i].y.m = x_reshaped[i, 1]
                mod_points[i].z.m = x_reshaped[i, 2]

            # Evaluate the objective function and Jacobian
            obj_fun_val = 0.0
            for edge_idx, (target_edge, multiface_data) in enumerate(surf_edge_mapping.items()):
                if surf_edge_mapping[target_edge][0] is None:
                    continue
                f = x[3 * len(mod_ijs) + edge_idx]
                f_sign = f_signs[target_edge]
                A = -f_sign * 1 / abs(f)
                A2 = -f_sign * (1 / f) ** 2
                dA = f_sign * (1 / f) ** 2
                dA2 = 2 * f_sign * (1 / abs(f)) ** 3
                d1_self = self.get_first_derivs_along_edge(target_edge, n_points=n_deriv_points)
                d2_self = self.get_second_derivs_along_edge(target_edge, n_points=n_deriv_points)

                # Objective function value update
                obj_fun_val += np.sum((d1_other[target_edge] + A * d1_self) ** 2)
                obj_fun_val += np.sum((d2_other[target_edge] + A2 * d2_self) ** 2)

                # Jacobian update loop
                start_xyz = 0  # Jacobian array starting index
                for mod_ij_idx, mod_ij in enumerate(mod_ijs):
                    d1_sens_self = self.get_first_deriv_cp_sens_along_edge(
                        target_edge,
                        mod_ij[0],
                        mod_ij[1],
                        n_points=n_deriv_points
                    )
                    d2_sens_self = self.get_second_deriv_cp_sens_along_edge(
                        target_edge,
                        mod_ij[0],
                        mod_ij[1],
                        n_points=n_deriv_points
                    )
                    # Jacobian array update
                    for k in range(3):
                        jac_arr[start_xyz + k] += 2 * np.sum(
                            (d1_other[target_edge][:, k] + A * d1_self[:, k]) * A * d1_sens_self[:, k] + (
                                    d2_other[target_edge][:, k] + A2 * d2_self[:, k]
                            ) * A2 * d2_sens_self[:, k]
                        )
                    # Increment the starting Jacobian index
                    start_xyz += 3

                jac_arr[3 * len(mod_ijs) + edge_idx] = np.sum(
                    2 * (d1_other[target_edge] + A * d1_self) * dA * d1_self +
                    2 * (d2_other[target_edge] + A2 * d2_self) * dA2 * d2_self
                )

            return obj_fun_val, jac_arr

        res = minimize(obj_fun_and_jac, x0, jac=True)
        return res

    def dSdu(self, u: float, v: float) -> np.ndarray:
        r"""
        Evaluates the first derivative with respect to :math:`u` at a single :math:`(u,v)` pair

        Parameters
        ----------
        u: float
            Position along :math:`u` in parametric space. Normally in the range :math:`[0,1]`
        v: float
            Position along :math:`v` in parametric space. Normally in the range :math:`[0,1]`

        Returns
        -------
        np.ndarray
            1-D array containing the :math:`x`-, :math:`y`-, and :math:`z`-components of the second derivative
        """
        P = self.get_control_point_array()
        return np.array(rational_bezier_surf_dsdu(P, self.weights, u, v))

    def dSdu_grid(self, Nu: int, Nv: int) -> np.ndarray:
        r"""
        Evaluates the first derivative with respect to :math:`u` on a linearly-spaced grid of :math:`u`- and
        :math:`v`-values.

        Parameters
        ----------
        Nu: int
            Number of evenly spaced :math:`u` values
        Nv: int
            Number of evenly spaced :math:`v` values

        Returns
        -------
        np.ndarray
            Array of size :math:`N_u \times N_v \times 3`
        """
        P = self.get_control_point_array()
        return np.array(rational_bezier_surf_dsdu_grid(P, self.weights, Nu, Nv))

    def dSdu_uvvecs(self, u: np.ndarray, v: np.ndarray):
        r"""
        Evaluates the first derivative of the surface with respect to :math:`u` at arbitrary vectors of
        :math:`u` and :math:`v`-values.

        Parameters
        ----------
        u: np.ndarray
            1-D array of :math:`u`-parameter values
        v: np.ndarray
            1-D array of :math:`v`-parameter values

        Returns
        -------
        np.ndarray
            Array of size :math:`\text{len}(u) \times \text{len}(v) \times 3`
        """
        P = self.get_control_point_array()
        return np.array(rational_bezier_surf_dsdu_uvvecs(P, self.weights, u, v))

    def dSdv(self, u: float or np.ndarray, v: float or np.ndarray):
        r"""
        Evaluates the first derivative with respect to :math:`v` at a single :math:`(u,v)` pair

        Parameters
        ----------
        u: float
            Position along :math:`u` in parametric space. Normally in the range :math:`[0,1]`
        v: float
            Position along :math:`v` in parametric space. Normally in the range :math:`[0,1]`

        Returns
        -------
        np.ndarray
            1-D array containing the :math:`x`-, :math:`y`-, and :math:`z`-components of the second derivative
        """
        P = self.get_control_point_array()
        return np.array(rational_bezier_surf_dsdv(P, self.weights, u, v))

    def dSdv_grid(self, Nu: int, Nv: int) -> np.ndarray:
        r"""
        Evaluates the first derivative with respect to :math:`v` on a linearly-spaced grid of :math:`u`- and
        :math:`v`-values.

        Parameters
        ----------
        Nu: int
            Number of evenly spaced :math:`u` values
        Nv: int
            Number of evenly spaced :math:`v` values

        Returns
        -------
        np.ndarray
            Array of size :math:`N_u \times N_v \times 3`
        """
        P = self.get_control_point_array()
        return np.array(rational_bezier_surf_dsdv_grid(P, self.weights, Nu, Nv))

    def dSdv_uvvecs(self, u: np.ndarray, v: np.ndarray):
        r"""
        Evaluates the first derivative of the surface with respect to :math:`v` at arbitrary vectors of
        :math:`u` and :math:`v`-values.

        Parameters
        ----------
        u: np.ndarray
            1-D array of :math:`u`-parameter values
        v: np.ndarray
            1-D array of :math:`v`-parameter values

        Returns
        -------
        np.ndarray
            Array of size :math:`\text{len}(u) \times \text{len}(v) \times 3`
        """
        P = self.get_control_point_array()
        return np.array(rational_bezier_surf_dsdv_uvvecs(P, self.weights, u, v))

    def d2Sdu2(self, u: float, v: float) -> np.ndarray:
        r"""
        Evaluates the second derivative with respect to :math:`u` at a single :math:`(u,v)` pair

        Parameters
        ----------
        u: float
            Position along :math:`u` in parametric space. Normally in the range :math:`[0,1]`
        v: float
            Position along :math:`v` in parametric space. Normally in the range :math:`[0,1]`

        Returns
        -------
        np.ndarray
            1-D array containing the :math:`x`-, :math:`y`-, and :math:`z`-components of the second derivative
        """
        P = self.get_control_point_array()
        return np.array(rational_bezier_surf_d2sdu2(P, self.weights, u, v))

    def d2Sdu2_grid(self, Nu: int, Nv: int) -> np.ndarray:
        r"""
        Evaluates the second derivative with respect to :math:`u` on a linearly-spaced grid of :math:`u`- and
        :math:`v`-values.

        Parameters
        ----------
        Nu: int
            Number of evenly spaced :math:`u` values
        Nv: int
            Number of evenly spaced :math:`v` values

        Returns
        -------
        np.ndarray
            Array of size :math:`N_u \times N_v \times 3`
        """
        P = self.get_control_point_array()
        return np.array(rational_bezier_surf_d2sdu2_grid(P, self.weights, Nu, Nv))

    def d2Sdu2_uvvecs(self, u: np.ndarray, v: np.ndarray):
        r"""
        Evaluates the second derivative of the surface with respect to :math:`u` at arbitrary vectors of
        :math:`u` and :math:`v`-values.

        Parameters
        ----------
        u: np.ndarray
            1-D array of :math:`u`-parameter values
        v: np.ndarray
            1-D array of :math:`v`-parameter values

        Returns
        -------
        np.ndarray
            Array of size :math:`\text{len}(u) \times \text{len}(v) \times 3`
        """
        P = self.get_control_point_array()
        return np.array(rational_bezier_surf_d2sdu2_uvvecs(P, self.weights, u, v))

    def d2Sdv2(self, u: float or np.ndarray, v: float or np.ndarray):
        r"""
        Evaluates the second derivative with respect to :math:`v` at a single :math:`(u,v)` pair

        Parameters
        ----------
        u: float
            Position along :math:`u` in parametric space. Normally in the range :math:`[0,1]`
        v: float
            Position along :math:`v` in parametric space. Normally in the range :math:`[0,1]`

        Returns
        -------
        np.ndarray
            1-D array containing the :math:`x`-, :math:`y`-, and :math:`z`-components of the second derivative
        """
        P = self.get_control_point_array()
        return np.array(rational_bezier_surf_d2sdv2(P, self.weights, u, v))

    def d2Sdv2_grid(self, Nu: int, Nv: int) -> np.ndarray:
        r"""
        Evaluates the second derivative with respect to :math:`v` on a linearly-spaced grid of :math:`u`- and
        :math:`v`-values.

        Parameters
        ----------
        Nu: int
            Number of evenly spaced :math:`u` values
        Nv: int
            Number of evenly spaced :math:`v` values

        Returns
        -------
        np.ndarray
            Array of size :math:`N_u \times N_v \times 3`
        """
        P = self.get_control_point_array()
        return np.array(rational_bezier_surf_d2sdv2_grid(P, self.weights, Nu, Nv))

    def d2Sdv2_uvvecs(self, u: np.ndarray, v: np.ndarray):
        r"""
        Evaluates the second derivative of the surface with respect to :math:`v` at arbitrary vectors of
        :math:`u` and :math:`v`-values.

        Parameters
        ----------
        u: np.ndarray
            1-D array of :math:`u`-parameter values
        v: np.ndarray
            1-D array of :math:`v`-parameter values

        Returns
        -------
        np.ndarray
            Array of size :math:`\text{len}(u) \times \text{len}(v) \times 3`
        """
        P = self.get_control_point_array()
        return np.array(rational_bezier_surf_d2sdv2_uvvecs(P, self.weights, u, v))

    def get_edge(self, edge: SurfaceEdge, n_points: int = 10) -> np.ndarray:
        r"""
        Evaluates the surface at ``n_points`` parameter locations along a given edge.

        Parameters
        ----------
        edge: SurfaceEdge
            Edge along which to evaluate
        n_points: int
            Number of evenly-spaced parameter locations at which to evaluate the edge curve. Default: 10

        Returns
        -------
        numpy.ndarray
            2-D array of size :math:`n_\text{points} \times 3`
        """
        P = self.get_control_point_array()
        if edge == SurfaceEdge.v1:
            return np.array(rational_bezier_surf_eval_iso_v(P, self.weights, n_points, 1.0))
        elif edge == SurfaceEdge.v0:
            return np.array(rational_bezier_surf_eval_iso_v(P, self.weights, n_points, 0.0))
        elif edge == SurfaceEdge.u1:
            return np.array(rational_bezier_surf_eval_iso_u(P, self.weights, 1.0, n_points))
        elif edge == SurfaceEdge.u0:
            return np.array(rational_bezier_surf_eval_iso_u(P, self.weights, 0.0, n_points))
        else:
            raise ValueError(f"No edge called {edge}")

    def get_first_derivs_along_edge(self, edge: SurfaceEdge, n_points: int = 10, perp: bool = True) -> np.ndarray:
        r"""
        Evaluates the parallel or perpendicular derivative along a surface edge at ``n_points`` parameter locations.
        The derivative represents either :math:`\frac{\partial \mathbf{S}(u,v)}{\partial u}` or
        :math:`\frac{\partial \mathbf{S}(u,v)}{\partial v}` depending on which edge is selected and which value is
        assigned to ``perp``.

        Parameters
        ----------
        edge: SurfaceEdge
            Edge along which to evaluate
        n_points: int
            Number of evenly-spaced parameter locations at which to evaluate the derivative. Default: 10
        perp: bool
            Whether to evaluate the cross-derivative. If ``False``, the derivative along the parameter direction
            parallel to the edge will be evaluated instead. Default: ``True``

        Returns
        -------
        numpy.ndarray
            2-D array of size :math:`n_\text{points} \times 3`
        """
        P = self.get_control_point_array()
        if edge == SurfaceEdge.v1:
            return np.array(rational_bezier_surf_dsdv_iso_v(P, self.weights, n_points, 1.0)) if perp else np.array(
                rational_bezier_surf_dsdu_iso_v(P, self.weights, n_points, 1.0))
        elif edge == SurfaceEdge.v0:
            return np.array(rational_bezier_surf_dsdv_iso_v(P, self.weights, n_points, 0.0)) if perp else np.array(
                rational_bezier_surf_dsdu_iso_v(P, self.weights, n_points, 0.0))
        elif edge == SurfaceEdge.u1:
            return np.array(rational_bezier_surf_dsdu_iso_u(P, self.weights, 1.0, n_points)) if perp else np.array(
                rational_bezier_surf_dsdv_iso_u(P, self.weights, 1.0, n_points))
        elif edge == SurfaceEdge.u0:
            return np.array(rational_bezier_surf_dsdu_iso_u(P, self.weights, 0.0, n_points)) if perp else np.array(
                rational_bezier_surf_dsdv_iso_u(P, self.weights, 0.0, n_points))
        else:
            raise ValueError(f"No edge called {edge}")

    def get_first_deriv_cp_sens_along_edge(self, edge: SurfaceEdge, i: int, j: int, n_points: int = 10,
                                           perp: bool = True) -> np.ndarray:
        r"""
        Gets the sensitivity of the first :math:`u`- or :math:`v`-derivative along an edge with respect to
        control point :math:`\mathbf{P}_{i,j}`

        Parameters
        ----------
        edge: SurfaceEdge
            Edge along which to evaluate
        i: int
            :math:`i`-index of the control point
        j: int
            :math:`j`-index of the control point
        n_points: int
            Number of evenly-spaced parameter locations at which to evaluate the derivative. Default: 10
        perp: bool
            Whether to evaluate the cross-derivative. If ``False``, the derivative along the parameter direction
            parallel to the edge will be evaluated instead. Default: ``True``

        Returns
        -------
        numpy.ndarray
            2-D array of size :math:`n_\text{points} \times 3`
        """
        w = self.weights
        if edge == SurfaceEdge.v1:
            return np.array(
                rational_bezier_surf_dsdv_dp_iso_v(w, i, j, self.n, self.m, 3, n_points, 1.0)) if perp else np.array(
                rational_bezier_surf_dsdu_dp_iso_v(w, i, j, self.n, self.m, 3, n_points, 1.0))
        elif edge == SurfaceEdge.v0:
            return np.array(
                rational_bezier_surf_dsdv_dp_iso_v(w, i, j, self.n, self.m, 3, n_points, 0.0)) if perp else np.array(
                rational_bezier_surf_dsdu_dp_iso_v(w, i, j, self.n, self.m, 3, n_points, 0.0))
        elif edge == SurfaceEdge.u1:
            return np.array(
                rational_bezier_surf_dsdu_dp_iso_u(w, i, j, self.n, self.m, 3, 1.0, n_points)) if perp else np.array(
                rational_bezier_surf_dsdv_dp_iso_u(w, i, j, self.n, self.m, 3, 1.0, n_points))
        elif edge == SurfaceEdge.u0:
            return np.array(
                rational_bezier_surf_dsdu_dp_iso_u(w, i, j, self.n, self.m, 3, 0.0, n_points)) if perp else np.array(
                rational_bezier_surf_dsdv_dp_iso_u(w, i, j, self.n, self.m, 3, 0.0, n_points))
        else:
            raise ValueError(f"No edge called {edge}")

    def get_second_derivs_along_edge(self, edge: SurfaceEdge, n_points: int = 10, perp: bool = True) -> np.ndarray:
        r"""
        Evaluates the parallel or perpendicular second derivative along a surface edge at ``n_points`` parameter
        locations. The derivative represents either :math:`\frac{\partial^2 \mathbf{S}(u,v)}{\partial u^2}` or
        :math:`\frac{\partial^2 \mathbf{S}(u,v)}{\partial v^2}` depending on which edge is selected and which value is
        assigned to ``perp``.

        Parameters
        ----------
        edge: SurfaceEdge
            Edge along which to evaluate
        n_points: int
            Number of evenly-spaced parameter locations at which to evaluate the second derivative. Default: 10
        perp: bool
            Whether to evaluate the cross-derivative. If ``False``, the second derivative along the parameter direction
            parallel to the edge will be evaluated instead. Default: ``True``

        Returns
        -------
        numpy.ndarray
            2-D array of size :math:`n_\text{points} \times 3`
        """
        P = self.get_control_point_array()
        if edge == SurfaceEdge.v1:
            return np.array(rational_bezier_surf_d2sdv2_iso_v(P, self.weights, n_points, 1.0)) if perp else np.array(
                rational_bezier_surf_d2sdu2_iso_v(P, self.weights, n_points, 1.0))
        elif edge == SurfaceEdge.v0:
            return np.array(rational_bezier_surf_d2sdv2_iso_v(P, self.weights, n_points, 0.0)) if perp else np.array(
                rational_bezier_surf_d2sdu2_iso_v(P, self.weights, n_points, 0.0))
        elif edge == SurfaceEdge.u1:
            return np.array(rational_bezier_surf_d2sdu2_iso_u(P, self.weights, 1.0, n_points)) if perp else np.array(
                rational_bezier_surf_d2sdv2_iso_u(P, self.weights, 1.0, n_points))
        elif edge == SurfaceEdge.u0:
            return np.array(rational_bezier_surf_d2sdu2_iso_u(P, self.weights, 0.0, n_points)) if perp else np.array(
                rational_bezier_surf_d2sdv2_iso_u(P, self.weights, 0.0, n_points))
        else:
            raise ValueError(f"No edge called {edge}")

    def get_second_deriv_cp_sens_along_edge(self, edge: SurfaceEdge, i: int, j: int, n_points: int = 10,
                                            perp: bool = True) -> np.ndarray:
        r"""
        Gets the sensitivity of the second :math:`u`- or :math:`v`-derivative along an edge with respect to
        control point :math:`\mathbf{P}_{i,j}`

        Parameters
        ----------
        edge: SurfaceEdge
            Edge along which to evaluate
        i: int
            :math:`i`-index of the control point
        j: int
            :math:`j`-index of the control point
        n_points: int
            Number of evenly-spaced parameter locations at which to evaluate the derivative. Default: 10
        perp: bool
            Whether to evaluate the cross-derivative. If ``False``, the derivative along the parameter direction
            parallel to the edge will be evaluated instead. Default: ``True``

        Returns
        -------
        numpy.ndarray
            2-D array of size :math:`n_\text{points} \times 3`
        """
        w = self.weights
        if edge == SurfaceEdge.v1:
            return np.array(
                rational_bezier_surf_d2sdv2_dp_iso_v(w, i, j, self.n, self.m, 3, n_points, 1.0)) if perp else np.array(
                rational_bezier_surf_d2sdu2_dp_iso_v(w, i, j, self.n, self.m, 3, n_points, 1.0))
        elif edge == SurfaceEdge.v0:
            return np.array(
                rational_bezier_surf_d2sdv2_dp_iso_v(w, i, j, self.n, self.m, 3, n_points, 0.0)) if perp else np.array(
                rational_bezier_surf_d2sdu2_dp_iso_v(w, i, j, self.n, self.m, 3, n_points, 0.0))
        elif edge == SurfaceEdge.u1:
            return np.array(
                rational_bezier_surf_d2sdu2_dp_iso_u(w, i, j, self.n, self.m, 3, 1.0, n_points)) if perp else np.array(
                rational_bezier_surf_d2sdv2_dp_iso_u(w, i, j, self.n, self.m, 3, 1.0, n_points))
        elif edge == SurfaceEdge.u0:
            return np.array(
                rational_bezier_surf_d2sdu2_dp_iso_u(w, i, j, self.n, self.m, 3, 0.0, n_points)) if perp else np.array(
                rational_bezier_surf_d2sdv2_dp_iso_u(w, i, j, self.n, self.m, 3, 0.0, n_points))
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
        #print(f"{self_perp_edge_derivs=},{other_perp_edge_derivs=}")
        self_perp_edge_derivs[np.absolute(self_perp_edge_derivs) < 1e-6] = 0.0
        other_perp_edge_derivs[np.absolute(other_perp_edge_derivs) < 1e-6] = 0.0

        ratios_other_self = other_perp_edge_derivs / self_perp_edge_derivs
        #print(f"{ratios_other_self=}")
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
                               x: Length = None, y: Length = None, z: Length = None):
        """
        Computes one parametric value given the other and a specified :math:`x`-, :math:`y`-, or :math:`z`-location.
        As an example, given a :obj:`~aerocaps.geom.surfaces.RationalBezierSurface` object
        assigned to the variable ``surf``,
        the :math:`u`-parameter corresponding to :math:`y=1.4` along the :math:`v=0.8` isoparametric curve can be
        computed using

        .. code-block:: python

            u = surf.get_u_or_v_given_uvxyz(v=0.8, y=1.4)

        Note that the inputs are keyword arguments to avoid having to specify ``None`` for each of the arguments
        not used.

        Parameters
        ----------
        u: float or None
            Value of :math:`u` to solve for or specify. If left as ``None``, this parameter will be solved for.
            If ``None``, :math:`v` must be specified. Default: ``None``
        v: float or None
            Value of :math:`v` to solve for or specify. If left as ``None``, this parameter will be solved for.
            If ``None``, :math:`u` must be specified. Default: ``None``
        uv_guess: float
            Starting guess for the unsolved :math:`u` or :math:`v` parameter. Default: ``0.5``
        x: Length or None
            :math:`x`-location corresponding to the :math:`u` or :math:`v` parameter to be solved. If this value is
            outside the surface geometry, the root-finder will fail and an error will be raised. If unspecified,
            either :math:`y` or :math:`z` must be specified. Default: ``None``
        y: Length or None
            :math:`y`-location corresponding to the :math:`u` or :math:`v` parameter to be solved. If this value is
            outside the surface geometry, the root-finder will fail and an error will be raised. If unspecified,
            either :math:`x` or :math:`z` must be specified. Default: ``None``
        z: Length or None
            :math:`z`-location corresponding to the :math:`u` or :math:`v` parameter to be solved. If this value is
            outside the surface geometry, the root-finder will fail and an error will be raised. If unspecified,
            either :math:`x` or :math:`y` must be specified. Default: ``None``

        Returns
        -------
        float
            The value of :math:`u` if :math:`v` is specified or :math:`v` if :math:`u` is specified
        """
        # Validate inputs
        if u is None and v is None or (u is not None and v is not None):
            raise ValueError("Must specify exactly one of either u or v")
        xyz_spec = (x is not None, y is not None, z is not None)
        if len([xyz for xyz in xyz_spec if xyz]) != 1:
            raise ValueError("Must specify exactly one of x, y, or z")

        if x is not None:
            xyz, xyz_val = "x", x.m
        elif y is not None:
            xyz, xyz_val = "y", y.m
        elif z is not None:
            xyz, xyz_val = "z", z.m
        else:
            raise ValueError("Did not detect an x, y, or z input")

        def root_find_func_u(u_current):
            point = self.evaluate_point3d(u_current, v)
            return np.array([getattr(point, xyz).m - xyz_val])

        def root_find_func_v(v_current):
            point = self.evaluate_point3d(u, v_current)
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
        Pw = self.get_homogeneous_control_points()

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
            [de_casteljau(i=0, j=i, k=k) for i in range(self.n_points_u)] for k in range(self.n_points_v)
        ])
        bez_surf_split_2_Pw = np.array([
            [de_casteljau(i=i, j=self.degree_u - i, k=k) for i in range(self.n_points_u)] for k in
            range(self.n_points_v)
        ])

        transposed_Pw_1 = np.transpose(bez_surf_split_1_Pw, (1, 0, 2))
        transposed_Pw_2 = np.transpose(bez_surf_split_2_Pw, (1, 0, 2))

        P1, w1 = self.project_homogeneous_control_points(transposed_Pw_1)
        P2, w2 = self.project_homogeneous_control_points(transposed_Pw_2)

        return (
            RationalBezierSurface(P1, w1),
            RationalBezierSurface(P2, w2)
        )

    def split_at_v(self, v0: float) -> ("BezierSurface", "BezierSurface"):
        """
        Splits the rational Bezier surface at :math:`v=v_0` along the :math:`u`-parametric direction.
        """
        Pw = self.get_homogeneous_control_points()

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
            [de_casteljau(i=0, j=i, k=k) for i in range(self.n_points_v)] for k in range(self.n_points_u)
        ])
        bez_surf_split_2_Pw = np.array([
            [de_casteljau(i=i, j=self.degree_v - i, k=k) for i in range(self.n_points_v)] for k in
            range(self.n_points_u)
        ])

        P1, w1 = self.project_homogeneous_control_points(bez_surf_split_1_Pw)
        P2, w2 = self.project_homogeneous_control_points(bez_surf_split_2_Pw)

        return (
            RationalBezierSurface(P1, w1),
            RationalBezierSurface(P2, w2)
        )

    def transform(self, **transformation_kwargs) -> "RationalBezierSurface":
        """
        Creates a transformed copy of the surface by transforming each of the control points

        Parameters
        ----------
        transformation_kwargs
            Keyword arguments passed to :obj:`~aerocaps.geom.transformation.Transformation3D`

        Returns
        -------
        RationalBezierSurface
            Transformed surface
        """
        transformation = Transformation3D(**transformation_kwargs)
        initial_control_points = self.get_control_point_array()
        return RationalBezierSurface(
            np.array([transformation.transform(p_arr) for p_arr in initial_control_points]),
            weights=deepcopy(self.weights),
            name=self.name, construction=self.construction
        )

    def generate_control_point_net(self) -> (typing.List[Point3D], typing.List[Line3D]):
        """
        Generates a list of :obj:`~aerocaps.geom.point.Point3D` and :obj:`~aerocaps.geom.curves.Line3D` objects
        representing the rational Bézier surface's control points and connections between them

        Returns
        -------
        typing.List[Point3D], typing.List[Line3D]
            Control points and lines between adjacent control points in flattened lists
        """
        control_points = self.get_control_point_array()
        points = []
        lines = []

        for i in range(self.n_points_u):
            for j in range(self.n_points_v):
                points.append(Point3D.from_array(control_points[i, j, :]))

        for i in range(self.n_points_u - 1):
            for j in range(self.n_points_v - 1):
                point_obj_1 = Point3D.from_array(control_points[i, j, :])
                point_obj_2 = Point3D.from_array(control_points[i + 1, j, :])
                point_obj_3 = Point3D.from_array(control_points[i, j + 1, :])

                line_1 = Line3D(p0=point_obj_1, p1=point_obj_2)
                line_2 = Line3D(p0=point_obj_1, p1=point_obj_3)
                lines.extend([line_1, line_2])

                if i < self.n_points_u - 2 and j < self.n_points_v - 2:
                    continue

                point_obj_4 = Point3D.from_array(control_points[i + 1, j + 1, :])
                line_3 = Line3D(p0=point_obj_3, p1=point_obj_4)
                line_4 = Line3D(p0=point_obj_2, p1=point_obj_4)
                lines.extend([line_3, line_4])

        return points, lines

    def plot_surface(self, plot: pv.Plotter, Nu: int = 50, Nv: int = 50, **mesh_kwargs):
        """
        Plots the rational Bézier surface using the `pyvista <https://pyvista.org/>`_ library

        Parameters
        ----------
        plot:
            :obj:`pyvista.Plotter` instance
        Nu: int
            Number of points to evaluate in the :math:`u`-parametric direction. Default: ``50``
        Nv: int
            Number of points to evaluate in the :math:`v`-parametric direction. Default: ``50``
        mesh_kwargs:
            Keyword arguments to pass to :obj:`pyvista.Plotter.add_mesh`

        Returns
        -------
        pyvista.core.pointset.StructuredGrid
            The evaluated rational Bézier surface
        """
        XYZ = self.evaluate_grid(Nu, Nv)
        grid = pv.StructuredGrid(XYZ[:, :, 0], XYZ[:, :, 1], XYZ[:, :, 2])
        plot.add_mesh(grid, **mesh_kwargs)
        return grid

    def plot_control_point_mesh_lines(self, plot: pv.Plotter, **line_kwargs) -> pv.Actor:
        """
        Plots the network of lines connecting the rational Bézier surface control points using the
        `pyvista <https://pyvista.org/>`_ library

        Parameters
        ----------
        plot:
            :obj:`pyvista.Plotter` instance
        line_kwargs:
            Keyword arguments to pass to the :obj:`pyvista.Plotter.add_lines`

        Returns
        -------
        pv.Actor
            The lines actor
        """
        _, line_objs = self.generate_control_point_net()
        line_arr = np.array([[line_obj.p0.as_array(), line_obj.p1.as_array()] for line_obj in line_objs])
        line_arr = line_arr.reshape((len(line_objs) * 2, 3))
        line_actor = plot.add_lines(line_arr, **line_kwargs)
        return line_actor

    def plot_control_points(self, plot: pv.Plotter, **point_kwargs) -> pv.Actor:
        """
        Plots the rational Bézier surface control points using the `pyvista <https://pyvista.org/>`_ library

        Parameters
        ----------
        plot:
            :obj:`pyvista.Plotter` instance
        point_kwargs:
            Keyword arguments to pass to the :obj:`pyvista.Plotter.add_points`

        Returns
        -------
        pv.Actor
            The points actor
        """
        point_objs, _ = self.generate_control_point_net()
        point_arr = np.array([point_obj.as_array() for point_obj in point_objs])
        point_actor = plot.add_points(point_arr, **point_kwargs)
        return point_actor

    def __repr__(self):
        return (f"{self.name}: {self.degree_u} x {self.degree_v} {self.__class__.__name__} "
                f"({self.degree_u + 1} x {self.degree_v + 1} control points)")


class BSplineSurface(Surface):
    """
    B-spline surface class
    """

    def __init__(self,
                 points: typing.List[typing.List[Point3D]] or np.ndarray,
                 knots_u: np.ndarray,
                 knots_v: np.ndarray,
                 name: str = "BSplineSurface",
                 construction: bool = False
                 ):
        """

        Parameters
        ----------
        points
        knots_u
        knots_v
        name: str
            Name of the geometric object. May be re-assigned a unique name when added to a
            :obj:`~aerocaps.geom.geometry_container.GeometryContainer`. Default: 'BSplineSurface'
        construction: bool
            Whether this is a geometry used only for construction of other geometries. If ``True``, this
            geometry will not be exported or plotted. Default: ``False``
        """
        if isinstance(points, np.ndarray):
            points = [[Point3D.from_array(pt_row) for pt_row in pt_mat] for pt_mat in points]
        self.points = points
        assert knots_u.ndim == 1
        assert knots_v.ndim == 1

        self.knots_u = deepcopy(knots_u)
        self.knots_v = deepcopy(knots_v)

        self._weights = np.ones((len(points), len(points[0])))
        super().__init__(name=name, construction=construction)

    @property
    def n_points_u(self) -> int:
        """Number of control points in the :math:`u`-parametric direction"""
        return len(self.points)

    @property
    def n_points_v(self) -> int:
        """Number of control points in the :math:`v`-parametric direction"""
        return len(self.points[0])

    @property
    def degree_u(self) -> int:
        """Surface degree in the :math:`u`-parametric direction"""
        return len(self.knots_u) - self.n_points_u - 1

    @property
    def degree_v(self) -> int:
        """Surface degree in the :math:`v`-parametric direction"""
        return len(self.knots_v) - self.n_points_v - 1

    @property
    def n(self) -> int:
        """
        Shorthand for :obj:`~aerocaps.geom.surfaces.BSplineSurface.degree_u`

        Returns
        -------
        int
            Surface degree in the :math:`u`-parametric direction
        """
        return self.degree_u

    @property
    def m(self) -> int:
        """
        Shorthand for :obj:`~aerocaps.geom.surfaces.BSplineSurface.degree_v`

        Returns
        -------
        int
            Surface degree in the :math:`v`-parametric direction
        """
        return self.degree_v

    @property
    def weights(self) -> np.ndarray:
        """Weight matrix (all ones for this surface type)"""
        return self._weights

    def to_iges(self, *args, **kwargs) -> aerocaps.iges.entity.IGESEntity:
        """
        Exports the NURBS surface to an IGES entity
        """
        return aerocaps.iges.surfaces.RationalBSplineSurfaceIGES(
            control_points=self.get_control_point_array(),
            knots_u=self.knots_u,
            knots_v=self.knots_v,
            weights=self.weights,
            degree_u=self.degree_u,
            degree_v=self.degree_v
        )

    def get_control_point_array(self) -> np.ndarray:
        r"""
        Gets the control points in float array form.

        Returns
        -------
        numpy.ndarray
            Array of size :math:`N_u \times N_v \times 3`
        """
        return np.array([np.array([p.as_array() for p in p_arr]) for p_arr in self.points])

    def evaluate(self, u: float, v: float) -> np.ndarray:
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
        return np.array(bspline_surf_eval(P, self.knots_u, self.knots_v, u, v))

    def evaluate_point3d(self, u: float, v: float) -> Point3D:
        r"""
        Evaluates the B-spline surface at a single :math:`(u,v)` parameter pair and returns a point object.

        Parameters
        ----------
        u: float
            Position along :math:`u` in parametric space. Normally in the range :math:`[0,1]`
        v: float
            Position along :math:`v` in parametric space. Normally in the range :math:`[0,1]`

        Returns
        -------
        Point3D
            Point object corresponding to the :math:`(u,v)` pair
        """
        return Point3D.from_array(self.evaluate(u, v))

    def evaluate_grid(self, Nu: int, Nv: int) -> np.ndarray:
        r"""
        Evaluates the B-spline surface on a uniform :math:`N_u \times N_v` grid of parameter values.

        Parameters
        ----------
        Nu: int
            Number of uniformly spaced parameter values in the :math:`u`-direction
        Nv: int
            Number of uniformly spaced parameter values in the :math:`v`-direction

        Returns
        -------
        numpy.ndarray
            Array of size :math:`N_u \times N_v \times 3`
        """
        P = self.get_control_point_array()
        return np.array(bspline_surf_eval_grid(P, self.knots_u, self.knots_v, Nu, Nv))

    def get_parallel_control_point_length(self, surface_edge: SurfaceEdge) -> int:
        r"""
        Gets the number of control points of the curve corresponding to the input surface edge.

        Parameters
        ----------
        surface_edge: SurfaceEdge
            Edge along which the number of control points is computed

        Returns
        -------
        int
            Number of control points parallel to the edge
        """
        if surface_edge in [SurfaceEdge.v1, SurfaceEdge.v0]:
            return self.n_points_u
        return self.n_points_v

    def get_perpendicular_control_point_length(self, surface_edge: SurfaceEdge) -> int:
        r"""
        Gets the number of control points in the parametric direction perpendicular to the input surface edge.

        Parameters
        ----------
        surface_edge: SurfaceEdge
            Edge along which the number of perpendicular control points is computed

        Returns
        -------
        int
            Number of control points perpendicular to the edge
        """
        if surface_edge in [SurfaceEdge.v1, SurfaceEdge.v0]:
            return self.n_points_v
        return self.n_points_u

    def get_parallel_degree(self, surface_edge: SurfaceEdge) -> int:
        r"""
        Gets the degree of the curve corresponding to the input surface edge.

        Parameters
        ----------
        surface_edge: SurfaceEdge
            Edge along which the parallel degree is evaluated

        Returns
        -------
        int
            Degree parallel to the edge
        """
        if surface_edge in [SurfaceEdge.v1, SurfaceEdge.v0]:
            return self.degree_u
        return self.degree_v

    def get_perpendicular_degree(self, surface_edge: SurfaceEdge) -> int:
        r"""
        Gets the degree of the curve in the parametric direction perpendicular to the input surface edge.

        Parameters
        ----------
        surface_edge: SurfaceEdge
            Edge along which the perpendicular degree is evaluated

        Returns
        -------
        int
            Degree perpendicular to the edge
        """
        if surface_edge in [SurfaceEdge.v1, SurfaceEdge.v0]:
            return self.degree_v
        return self.degree_u

    def get_parallel_knots(self, surface_edge: SurfaceEdge) -> np.ndarray:
        r"""
        Gets the knots in the parametric direction parallel to the input surface edge.

        Parameters
        ----------
        surface_edge: SurfaceEdge
            Edge along which the parallel knots are returned

        Returns
        -------
        numpy.ndarray
            Knots parallel to the edge
        """
        if surface_edge in [SurfaceEdge.v1, SurfaceEdge.v0]:
            return self.knots_u
        return self.knots_v

    def get_perpendicular_knots(self, surface_edge: SurfaceEdge) -> np.ndarray:
        r"""
        Gets the knots in the parametric direction perpendicular to the input surface edge.

        Parameters
        ----------
        surface_edge: SurfaceEdge
            Edge along which the perpendicular knots are returned

        Returns
        -------
        numpy.ndarray
            Knots perpendicular to the edge
        """
        if surface_edge in [SurfaceEdge.v1, SurfaceEdge.v0]:
            return self.knots_v
        return self.knots_u

    def get_point(self, row_index: int, continuity_index: int, surface_edge: SurfaceEdge) -> Point3D:
        r"""
        Gets the point corresponding to a particular index along the edge curve with perpendicular index
        corresponding to the level of continuity being applied. For example, for a :math:`6 \times 5` B-spline surface,
        the following code

        .. code-block:: python

            p = surf.get_point(2, 1, ac.SurfaceEdge.v0)

        returns the point :math:`\mathbf{P}_{2,1}` and

        .. code-block:: python

            p = surf.get_point(2, 1, ac.SurfaceEdge.u1)

        returns the point :math:`\mathbf{P}_{6-1,2} = \mathbf{P}_{5,2}` if there are no internal knot vectors.
        If the B-spline surface has internal knot vectors, the actual :math:`i`-index of the point may be different,
        but the second-to-last point in the third row of control points will still be returned.

        .. seealso::

            :obj:`~aerocaps.geom.surfaces.BSplineSurface.set_point`
                Setter equivalent of this method

        Parameters
        ----------
        row_index: int
            Index along the surface edge control points
        continuity_index: int
            Index in the parametric direction perpendicular to the surface edge. Normally either ``0``, ``1``, or ``2``
        surface_edge: SurfaceEdge
            Edge of the surface along which to retrieve the control point

        Returns
        -------
        Point3D
            Point used to enforce :math:`G^x` continuity, where :math:`x` is the value of ``continuity_index``
        """
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
        r"""
        Sets the point corresponding to a particular index along the edge curve with perpendicular index
        corresponding to the level of continuity being applied. For example, for a :math:`6 \times 5` B-spline surface,
        the following code

        .. code-block:: python

            p = ac.Point3D.from_array(np.array([3.0, 4.0, 5.0]))
            surf.set_point(p, 2, 1, ac.SurfaceEdge.v0)

        sets the value of point :math:`\mathbf{P}_{2,1}` to :math:`[3,4,5]^T` and

        .. code-block:: python

            p = ac.Point3D.from_array(np.array([3.0, 4.0, 5.0]))
            surf.get_point(p, 2, 1, ac.SurfaceEdge.u1)

        sets the value of point :math:`\mathbf{P}_{6-1,2} = \mathbf{P}_{5,2}` to :math:`[3,4,5]^T` if there are no
        internal knot vectors.

        .. seealso::

            :obj:`~aerocaps.geom.surfaces.BSplineSurface.get_point`
                Getter equivalent of this method

        Parameters
        ----------
        point: Point3D
            Point object to apply at the specified indices
        row_index: int
            Index along the surface edge control points
        continuity_index: int
            Index in the parametric direction perpendicular to the surface edge. Normally either ``0``, ``1``, or ``2``
        surface_edge: SurfaceEdge
            Edge of the surface along which to retrieve the control point
        """
        if surface_edge == SurfaceEdge.v1:
            self.points[row_index][-(continuity_index + 1)].x.m = point.x.m
            self.points[row_index][-(continuity_index + 1)].y.m = point.y.m
            self.points[row_index][-(continuity_index + 1)].z.m = point.z.m
        elif surface_edge == SurfaceEdge.v0:
            self.points[row_index][continuity_index].x.m = point.x.m
            self.points[row_index][continuity_index].y.m = point.y.m
            self.points[row_index][continuity_index].z.m = point.z.m
        elif surface_edge == SurfaceEdge.u1:
            self.points[-(continuity_index + 1)][row_index].x.m = point.x.m
            self.points[-(continuity_index + 1)][row_index].y.m = point.y.m
            self.points[-(continuity_index + 1)][row_index].z.m = point.z.m
        elif surface_edge == SurfaceEdge.u0:
            self.points[continuity_index][row_index].x.m = point.x.m
            self.points[continuity_index][row_index].y.m = point.y.m
            self.points[continuity_index][row_index].z.m = point.z.m
        else:
            raise ValueError("Invalid surface_edge value")

    def extract_edge_curve(self, surface_edge: SurfaceEdge) -> BSplineCurve3D:
        """
        Extracts the control points, weights, and knots from one of the four edges of the B-spline surface and
        outputs a B-spline curve with these control points and weights

        Parameters
        ----------
        surface_edge: SurfaceEdge
            Edge along which to extract the curve

        Returns
        -------
        BSplineCurve3D
            B-spline curve with control points and knots corresponding to the control points and knots
            along the edge of the surface
        """
        P = self.get_control_point_array()

        if surface_edge == SurfaceEdge.u0:
            return BSplineCurve3D(P[0, :, :], self.knots_v, self.degree_v)
        if surface_edge == SurfaceEdge.u1:
            return BSplineCurve3D(P[-1, :, :], self.knots_v, self.degree_v)
        if surface_edge == SurfaceEdge.v0:
            return BSplineCurve3D(P[:, 0, :], self.knots_u, self.degree_u)
        if surface_edge == SurfaceEdge.v1:
            return BSplineCurve3D(P[:, -1, :], self.knots_u, self.degree_u)

        raise ValueError(f"Invalid surface edge {surface_edge}")

    def enforce_g0(self, other: "BSplineSurface",
                   surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge):
        r"""
        Enforces :math:`G^0` continuity along the input ``surface_edge`` by equating the control points
        along this edge to the corresponding control points and weights along the ``other_surface_edge``
        of the B-spline surface given by ``other``.
        The control points of the surface from which this method is called are modified in-place, and the control
        points of ``other`` are left unchanged.

        .. important::

            The parallel degree of the current surface along ``surface_edge`` must be equal to the parallel degree
            of the ``other`` surface along ``other_surface_edge``, otherwise an ``AssertionError`` will be raised.
            Additionally, the knot vector along the ``surface_edge`` of the current surface must be equal
            to the knot vector along the ``other_surface_edge`` of the other surface.

        .. seealso::

            :obj:`~aerocaps.geom.surfaces.BSplineSurface.enforce_c0`
                Parametric continuity equivalent (:math:`C^0`)

        Parameters
        ----------
        other: BSplineSurface
            Another B-spline surface along which an edge will be used for stitching
        surface_edge: SurfaceEdge
            The edge of the current surface to modify
        other_surface_edge: SurfaceEdge
            Tool edge of surface ``other`` which determines the positions of control points along ``surface_edge``
            of the current surface
        """
        # P^b[:, 0] = P^a[:, -1]
        self_parallel_knots = self.get_parallel_knots(surface_edge)
        other_parallel_knots = other.get_parallel_knots(other_surface_edge)
        self_parallel_degree = self.get_parallel_degree(surface_edge)
        other_parallel_degree = other.get_parallel_degree(other_surface_edge)
        if len(self_parallel_knots) != len(other_parallel_knots):
            raise ValueError(f"Length of the knot vector parallel to the edge of the input surface "
                             f"({len(self_parallel_knots)}) is not equal to the length of the knot vector "
                             f"parallel to the edge of the other surface ({len(other_parallel_knots)})")
        if any([k1 != k2 for k1, k2 in zip(self_parallel_knots, other_parallel_knots)]):
            raise ValueError(f"Knots parallel to the edge of the input surface ({self_parallel_knots}) "
                             f"are not equal to the knots in the direction parallel to the edge of the other "
                             f"surface ({other_parallel_knots})")
        if self_parallel_degree != other_parallel_degree:
            raise ValueError(f"Degree parallel to the edge of the input surface ({self_parallel_degree}) does "
                             f"not match the degree parallel to the edge of the other surface "
                             f"({other_parallel_degree})")
        for row_index in range(self.get_parallel_control_point_length(surface_edge)):
            self.set_point(other.get_point(row_index, 0, other_surface_edge), row_index, 0, surface_edge)

    def enforce_c0(self, other: "BSplineSurface", surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge):
        """
        For zeroth-degree continuity, there is no difference between geometric (:math:`G^0`) and parametric
        (:math:`C^0`) continuity. Because this method is simply a convenience method that calls
        :obj:`~aerocaps.geom.surfaces.BSplineSurface.enforce_g0`, see the documentation for that method for more
        detailed documentation.

        Parameters
        ----------
        other: BSplineSurface
            Another B-spline surface along which an edge will be used for stitching
        surface_edge: SurfaceEdge
            The edge of the current surface to modify
        other_surface_edge: SurfaceEdge
            Tool edge of surface ``other`` which determines the positions of control points along ``surface_edge``
            of the current surface
        """
        self.enforce_g0(other, surface_edge, other_surface_edge)

    def enforce_g0g1(self, other: "BSplineSurface", f: float,
                     surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge):
        r"""
        First enforces :math:`G^0` continuity, then tangent (:math:`G^1`) continuity is enforced according to
        the following equation:

        .. math::

            \mathcal{P}^{b,\mathcal{E}_b}_{k,1} = \mathcal{P}^{b,\mathcal{E}_b}_{k,0} + f \frac{p_{\perp}^{a,\mathcal{E}_a}}{p_{\perp}^{b,\mathcal{E}_b}} \left[\mathcal{P}^{a,\mathcal{E}_a}_{k,0} - \mathcal{P}^{a,\mathcal{E}_a}_{k,1} \right] \text{ for }k=0,1,\ldots,p_{\parallel}^{b,\mathcal{E}_b}

        Here, :math:`b` corresponds to the current surface, and :math:`a` corresponds to the ``other`` surface.
        The control points of the surface from which this method is called are modified in-place, and the control
        points of ``other`` are left unchanged.

        .. seealso::

            :obj:`~aerocaps.geom.surfaces.BSplineSurface.enforce_g0`
                Geometric point continuity enforcement (:math:`G^0`)
            :obj:`~aerocaps.geom.surfaces.BSplineSurface.enforce_c0c1`
                Parametric continuity equivalent (:math:`C^1`)

        Parameters
        ----------
        other: BSplineSurface
            Another B-spline surface along which an edge will be used for stitching
        f: float
            Tangent proportionality factor
        surface_edge: SurfaceEdge
            The edge of the current surface to modify
        other_surface_edge: SurfaceEdge
            Tool edge of surface ``other`` which determines the positions of control points along ``surface_edge``
            of the current surface
        """
        self.enforce_g0(other, surface_edge, other_surface_edge)
        n_ratio = other.get_perpendicular_degree(other_surface_edge) / self.get_perpendicular_degree(surface_edge)
        for row_index in range(self.get_parallel_control_point_length(surface_edge)):
            P_i0_b = self.get_point(row_index, 0, surface_edge)
            P_im_a = other.get_point(row_index, 0, other_surface_edge)
            P_im1_a = other.get_point(row_index, 1, other_surface_edge)

            P_i1_b = P_i0_b + f * n_ratio * (P_im_a - P_im1_a)
            self.set_point(P_i1_b, row_index, 1, surface_edge)

    def enforce_c0c1(self, other: "BSplineSurface",
                     surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge):
        r"""
        Equivalent to calling :obj:`~aerocaps.geom.surfaces.BSplineSurface.enforce_g0g1` with ``f=1.0``. See that
        method for more detailed documentation.

        Parameters
        ----------
        other: BSplineSurface
            Another B-spline surface along which an edge will be used for stitching
        surface_edge: SurfaceEdge
            The edge of the current surface to modify
        other_surface_edge: SurfaceEdge
            Tool edge of surface ``other`` which determines the positions of control points along ``surface_edge``
            of the current surface
        """
        self.enforce_g0g1(other, 1.0, surface_edge, other_surface_edge)

    def enforce_g0g1g2(self, other: "BSplineSurface", f: float,
                       surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge):
        r"""
        First enforces :math:`G^0` and :math:`G^1` continuity, then curvature (:math:`G^2`) continuity is enforced
        according to the following equation:

        .. math::

            \mathcal{P}^{b,\mathcal{E}_b}_{k,2} = 2 \mathcal{P}^{b,\mathcal{E}_b}_{k,1} - \mathcal{P}^{b,\mathcal{E}_b}_{k,0} + f^2 \frac{p_{\perp}^{a,\mathcal{E}_a}(p_{\perp}^{a,\mathcal{E}_a}-1)}{p_{\perp}^{b,\mathcal{E}_b}(p_{\perp}^{b,\mathcal{E}_b}-1)} \left[ \mathcal{P}^{a,\mathcal{E}_a}_{k,0} - 2 \mathcal{P}^{a,\mathcal{E}_a}_{k,1} + \mathcal{P}^{a,\mathcal{E}_a}_{k,2} \right]  \text{ for }k=0,1,\ldots,p_{\parallel}^{b,\mathcal{E}_b}

        Here, :math:`b` corresponds to the current surface, and :math:`a` corresponds to the ``other`` surface.
        The control points of the surface from which this method is called are modified in-place, and the control
        points of ``other`` are left unchanged.

        .. seealso::

         :obj:`~aerocaps.geom.surfaces.BSplineSurface.enforce_g0`
             Geometric point continuity enforcement (:math:`G^0`)
         :obj:`~aerocaps.geom.surfaces.BSplineSurface.enforce_g0g1`
             Geometric tangent continuity enforcement (:math:`G^1`)
         :obj:`~aerocaps.geom.surfaces.BSplineSurface.enforce_c0c1c2`
             Parametric continuity equivalent (:math:`C^2`)

        Parameters
        ----------
        other: BSplineSurface
            Another B-spline surface along which an edge will be used for stitching
        f: float
            Tangent proportionality factor
        surface_edge: SurfaceEdge
            The edge of the current surface to modify
        other_surface_edge: SurfaceEdge
            Tool edge of surface ``other`` which determines the positions of control points along ``surface_edge``
            of the current surface
        """
        self.enforce_g0g1(other, f, surface_edge, other_surface_edge)
        p_perp_a = other.get_perpendicular_degree(other_surface_edge)
        p_perp_b = self.get_perpendicular_degree(surface_edge)
        n_ratio = (p_perp_a ** 2 - p_perp_a) / (p_perp_b ** 2 - p_perp_b)
        for row_index in range(self.get_parallel_control_point_length(surface_edge)):
            P_i0_b = self.get_point(row_index, 0, surface_edge)
            P_i1_b = self.get_point(row_index, 1, surface_edge)
            P_im_a = other.get_point(row_index, 0, other_surface_edge)
            P_im1_a = other.get_point(row_index, 1, other_surface_edge)
            P_im2_a = other.get_point(row_index, 2, other_surface_edge)

            P_i2_b = (2.0 * P_i1_b - P_i0_b) + f ** 2 * n_ratio * (P_im_a - 2.0 * P_im1_a + P_im2_a)
            self.set_point(P_i2_b, row_index, 2, surface_edge)

    def enforce_c0c1c2(self, other: "BSplineSurface",
                       surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge):
        r"""
        Equivalent to calling :obj:`~aerocaps.geom.surfaces.BSplineSurface.enforce_g0g1g2` with ``f=1.0``.
        See that method for more detailed documentation.

        Parameters
        ----------
        other: BSplineSurface
            Another B-spline surface along which an edge will be used for stitching
        surface_edge: SurfaceEdge
            The edge of the current surface to modify
        other_surface_edge: SurfaceEdge
            Tool edge of surface ``other`` which determines the positions of control points along ``surface_edge``
            of the current surface
        """
        self.enforce_g0g1g2(other, 1.0, surface_edge, other_surface_edge)

    def dSdu(self, u: float, v: float) -> np.ndarray:
        r"""
        Evaluates the first derivative with respect to :math:`u` at a single :math:`(u,v)` pair

        Parameters
        ----------
        u: float
            Position along :math:`u` in parametric space. Normally in the range :math:`[0,1]`
        v: float
            Position along :math:`v` in parametric space. Normally in the range :math:`[0,1]`

        Returns
        -------
        np.ndarray
            1-D array containing the :math:`x`-, :math:`y`-, and :math:`z`-components of the second derivative
        """
        P = self.get_control_point_array()
        return np.array(bspline_surf_dsdu(P, self.knots_u, self.knots_v, u, v))

    def dSdu_grid(self, Nu: int, Nv: int) -> np.ndarray:
        r"""
        Evaluates the first derivative with respect to :math:`u` on a linearly-spaced grid of :math:`u`- and
        :math:`v`-values.

        Parameters
        ----------
        Nu: int
            Number of evenly spaced :math:`u` values
        Nv: int
            Number of evenly spaced :math:`v` values

        Returns
        -------
        np.ndarray
            Array of size :math:`N_u \times N_v \times 3`
        """
        P = self.get_control_point_array()
        return np.array(bspline_surf_dsdu_grid(P, self.knots_u, self.knots_v, Nu, Nv))

    def dSdu_uvvecs(self, u: np.ndarray, v: np.ndarray):
        r"""
        Evaluates the first derivative of the surface with respect to :math:`u` at arbitrary vectors of
        :math:`u` and :math:`v`-values.

        Parameters
        ----------
        u: np.ndarray
            1-D array of :math:`u`-parameter values
        v: np.ndarray
            1-D array of :math:`v`-parameter values

        Returns
        -------
        np.ndarray
            Array of size :math:`\text{len}(u) \times \text{len}(v) \times 3`
        """
        P = self.get_control_point_array()
        return np.array(bspline_surf_dsdu_uvvecs(P, self.knots_u, self.knots_v, u, v))

    def dSdv(self, u: float or np.ndarray, v: float or np.ndarray):
        r"""
        Evaluates the first derivative with respect to :math:`v` at a single :math:`(u,v)` pair

        Parameters
        ----------
        u: float
            Position along :math:`u` in parametric space. Normally in the range :math:`[0,1]`
        v: float
            Position along :math:`v` in parametric space. Normally in the range :math:`[0,1]`

        Returns
        -------
        np.ndarray
            1-D array containing the :math:`x`-, :math:`y`-, and :math:`z`-components of the second derivative
        """
        P = self.get_control_point_array()
        return np.array(bspline_surf_dsdv(P, self.knots_u, self.knots_v, u, v))

    def dSdv_grid(self, Nu: int, Nv: int) -> np.ndarray:
        r"""
        Evaluates the first derivative with respect to :math:`v` on a linearly-spaced grid of :math:`u`- and
        :math:`v`-values.

        Parameters
        ----------
        Nu: int
            Number of evenly spaced :math:`u` values
        Nv: int
            Number of evenly spaced :math:`v` values

        Returns
        -------
        np.ndarray
            Array of size :math:`N_u \times N_v \times 3`
        """
        P = self.get_control_point_array()
        return np.array(bspline_surf_dsdv_grid(P, self.knots_u, self.knots_v, Nu, Nv))

    def dSdv_uvvecs(self, u: np.ndarray, v: np.ndarray):
        r"""
        Evaluates the first derivative of the surface with respect to :math:`v` at arbitrary vectors of
        :math:`u` and :math:`v`-values.

        Parameters
        ----------
        u: np.ndarray
            1-D array of :math:`u`-parameter values
        v: np.ndarray
            1-D array of :math:`v`-parameter values

        Returns
        -------
        np.ndarray
            Array of size :math:`\text{len}(u) \times \text{len}(v) \times 3`
        """
        P = self.get_control_point_array()
        return np.array(bspline_surf_dsdv_uvvecs(P, self.knots_u, self.knots_v, u, v))

    def d2Sdu2(self, u: float, v: float) -> np.ndarray:
        r"""
        Evaluates the second derivative with respect to :math:`u` at a single :math:`(u,v)` pair

        Parameters
        ----------
        u: float
            Position along :math:`u` in parametric space. Normally in the range :math:`[0,1]`
        v: float
            Position along :math:`v` in parametric space. Normally in the range :math:`[0,1]`

        Returns
        -------
        np.ndarray
            1-D array containing the :math:`x`-, :math:`y`-, and :math:`z`-components of the second derivative
        """
        P = self.get_control_point_array()
        return np.array(bspline_surf_d2sdu2(P, self.knots_u, self.knots_v, u, v))

    def d2Sdu2_grid(self, Nu: int, Nv: int) -> np.ndarray:
        r"""
        Evaluates the second derivative with respect to :math:`u` on a linearly-spaced grid of :math:`u`- and
        :math:`v`-values.

        Parameters
        ----------
        Nu: int
            Number of evenly spaced :math:`u` values
        Nv: int
            Number of evenly spaced :math:`v` values

        Returns
        -------
        np.ndarray
            Array of size :math:`N_u \times N_v \times 3`
        """
        P = self.get_control_point_array()
        return np.array(bspline_surf_d2sdu2_grid(P, self.knots_u, self.knots_v, Nu, Nv))

    def d2Sdu2_uvvecs(self, u: np.ndarray, v: np.ndarray):
        r"""
        Evaluates the second derivative of the surface with respect to :math:`u` at arbitrary vectors of
        :math:`u` and :math:`v`-values.

        Parameters
        ----------
        u: np.ndarray
            1-D array of :math:`u`-parameter values
        v: np.ndarray
            1-D array of :math:`v`-parameter values

        Returns
        -------
        np.ndarray
            Array of size :math:`\text{len}(u) \times \text{len}(v) \times 3`
        """
        P = self.get_control_point_array()
        return np.array(bspline_surf_d2sdu2_uvvecs(P, self.knots_u, self.knots_v, u, v))

    def d2Sdv2(self, u: float or np.ndarray, v: float or np.ndarray):
        r"""
        Evaluates the second derivative with respect to :math:`v` at a single :math:`(u,v)` pair

        Parameters
        ----------
        u: float
            Position along :math:`u` in parametric space. Normally in the range :math:`[0,1]`
        v: float
            Position along :math:`v` in parametric space. Normally in the range :math:`[0,1]`

        Returns
        -------
        np.ndarray
            1-D array containing the :math:`x`-, :math:`y`-, and :math:`z`-components of the second derivative
        """
        P = self.get_control_point_array()
        return np.array(bspline_surf_d2sdv2(P, self.knots_u, self.knots_v, u, v))

    def d2Sdv2_grid(self, Nu: int, Nv: int) -> np.ndarray:
        r"""
        Evaluates the second derivative with respect to :math:`v` on a linearly-spaced grid of :math:`u`- and
        :math:`v`-values.

        Parameters
        ----------
        Nu: int
            Number of evenly spaced :math:`u` values
        Nv: int
            Number of evenly spaced :math:`v` values

        Returns
        -------
        np.ndarray
            Array of size :math:`N_u \times N_v \times 3`
        """
        P = self.get_control_point_array()
        return np.array(bspline_surf_d2sdv2_grid(P, self.knots_u, self.knots_v, Nu, Nv))

    def d2Sdv2_uvvecs(self, u: np.ndarray, v: np.ndarray):
        r"""
        Evaluates the second derivative of the surface with respect to :math:`v` at arbitrary vectors of
        :math:`u` and :math:`v`-values.

        Parameters
        ----------
        u: np.ndarray
            1-D array of :math:`u`-parameter values
        v: np.ndarray
            1-D array of :math:`v`-parameter values

        Returns
        -------
        np.ndarray
            Array of size :math:`\text{len}(u) \times \text{len}(v) \times 3`
        """
        P = self.get_control_point_array()
        return np.array(bspline_surf_d2sdv2_uvvecs(P, self.knots_u, self.knots_v, u, v))

    def get_edge(self, edge: SurfaceEdge, n_points: int = 10) -> np.ndarray:
        r"""
        Evaluates the surface at ``n_points`` parameter locations along a given edge.

        Parameters
        ----------
        edge: SurfaceEdge
            Edge along which to evaluate
        n_points: int
            Number of evenly-spaced parameter locations at which to evaluate the edge curve. Default: 10

        Returns
        -------
        numpy.ndarray
            2-D array of size :math:`n_\text{points} \times 3`
        """
        P = self.get_control_point_array()
        if edge == SurfaceEdge.v1:
            return np.array(bspline_surf_eval_iso_v(P, self.knots_u, self.knots_v, n_points, 1.0))
        elif edge == SurfaceEdge.v0:
            return np.array(bspline_surf_eval_iso_v(P, self.knots_u, self.knots_v, n_points, 0.0))
        elif edge == SurfaceEdge.u1:
            return np.array(bspline_surf_eval_iso_u(P, self.knots_u, self.knots_v, 1.0, n_points))
        elif edge == SurfaceEdge.u0:
            return np.array(bspline_surf_eval_iso_u(P, self.knots_u, self.knots_v, 0.0, n_points))
        else:
            raise ValueError(f"No edge called {edge}")

    def get_first_derivs_along_edge(self, edge: SurfaceEdge, n_points: int = 10, perp: bool = True) -> np.ndarray:
        r"""
        Evaluates the parallel or perpendicular derivative along a surface edge at ``n_points`` parameter locations.
        The derivative represents either :math:`\frac{\partial \mathbf{S}(u,v)}{\partial u}` or
        :math:`\frac{\partial \mathbf{S}(u,v)}{\partial v}` depending on which edge is selected and which value is
        assigned to ``perp``.

        Parameters
        ----------
        edge: SurfaceEdge
            Edge along which to evaluate
        n_points: int
            Number of evenly-spaced parameter locations at which to evaluate the derivative. Default: 10
        perp: bool
            Whether to evaluate the cross-derivative. If ``False``, the derivative along the parameter direction
            parallel to the edge will be evaluated instead. Default: ``True``

        Returns
        -------
        numpy.ndarray
            2-D array of size :math:`n_\text{points} \times 3`
        """
        P = self.get_control_point_array()
        if edge == SurfaceEdge.v1:
            return np.array(bspline_surf_dsdv_iso_v(
                P, self.knots_u, self.knots_v, n_points, 1.0)) if perp else np.array(
                bspline_surf_dsdu_iso_v(P, self.knots_u, self.knots_v, n_points, 1.0))
        elif edge == SurfaceEdge.v0:
            return np.array(bspline_surf_dsdv_iso_v(
                P, self.knots_u, self.knots_v, n_points, 0.0)) if perp else np.array(
                bspline_surf_dsdu_iso_v(P, self.knots_u, self.knots_v, n_points, 0.0))
        elif edge == SurfaceEdge.u1:
            return np.array(bspline_surf_dsdu_iso_u(
                P, self.knots_u, self.knots_v, 1.0, n_points)) if perp else np.array(
                bspline_surf_dsdv_iso_u(P, self.knots_u, self.knots_v, 1.0, n_points))
        elif edge == SurfaceEdge.u0:
            return np.array(bspline_surf_dsdu_iso_u(
                P, self.knots_u, self.knots_v, 0.0, n_points)) if perp else np.array(
                bspline_surf_dsdv_iso_u(P, self.knots_u, self.knots_v, 0.0, n_points))
        else:
            raise ValueError(f"No edge called {edge}")

    def get_second_derivs_along_edge(self, edge: SurfaceEdge, n_points: int = 10, perp: bool = True) -> np.ndarray:
        r"""
        Evaluates the parallel or perpendicular second derivative along a surface edge at ``n_points`` parameter
        locations. The derivative represents either :math:`\frac{\partial^2 \mathbf{S}(u,v)}{\partial u^2}` or
        :math:`\frac{\partial^2 \mathbf{S}(u,v)}{\partial v^2}` depending on which edge is selected and which value is
        assigned to ``perp``.

        Parameters
        ----------
        edge: SurfaceEdge
            Edge along which to evaluate
        n_points: int
            Number of evenly-spaced parameter locations at which to evaluate the second derivative. Default: 10
        perp: bool
            Whether to evaluate the cross-derivative. If ``False``, the second derivative along the parameter direction
            parallel to the edge will be evaluated instead. Default: ``True``

        Returns
        -------
        numpy.ndarray
            2-D array of size :math:`n_\text{points} \times 3`
        """
        P = self.get_control_point_array()
        if edge == SurfaceEdge.v1:
            return np.array(bspline_surf_d2sdv2_iso_v(
                P, self.knots_u, self.knots_v, n_points, 1.0)) if perp else np.array(
                bspline_surf_d2sdu2_iso_v(P, self.knots_u, self.knots_v, n_points, 1.0))
        elif edge == SurfaceEdge.v0:
            return np.array(bspline_surf_d2sdv2_iso_v(
                P, self.knots_u, self.knots_v, n_points, 0.0)) if perp else np.array(
                bspline_surf_d2sdu2_iso_v(P, self.knots_u, self.knots_v, n_points, 0.0))
        elif edge == SurfaceEdge.u1:
            return np.array(bspline_surf_d2sdu2_iso_u(
                P, self.knots_u, self.knots_v, 1.0, n_points)) if perp else np.array(
                bspline_surf_d2sdv2_iso_u(P, self.knots_u, self.knots_v, 1.0, n_points))
        elif edge == SurfaceEdge.u0:
            return np.array(bspline_surf_d2sdu2_iso_u(
                P, self.knots_u, self.knots_v, 0.0, n_points)) if perp else np.array(
                bspline_surf_d2sdv2_iso_u(P, self.knots_u, self.knots_v, 0.0, n_points))
        else:
            raise ValueError(f"No edge called {edge}")

    def verify_g0(self, other: "BSplineSurface", surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge,
                  n_points: int = 10):
        """ Verifies that two NURBS Surfaces are G0 continuous along their shared edge"""
        self_edge = self.get_edge(surface_edge, n_points=n_points)
        other_edge = other.get_edge(other_surface_edge, n_points=n_points)
        assert np.array_equal(self_edge, other_edge)

    def verify_g1(self, other: "BSplineSurface", surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge,
                  n_points: int = 10):
        """
        Verifies that two NURBSSurfaces are G1 continuous along their shared edge
        """
        # Get the first derivatives at the boundary and perpendicular to the boundary for each surface,
        # evaluated at "n_points" locations along the boundary
        self_perp_edge_derivs = self.get_first_derivs_along_edge(surface_edge, n_points=n_points, perp=True)
        other_perp_edge_derivs = other.get_first_derivs_along_edge(other_surface_edge, n_points=n_points, perp=True)
        print(f"{self_perp_edge_derivs = }")
        print(f"{other_perp_edge_derivs = }")
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

    def verify_g2(self, other: "BSplineSurface", surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge,
                  n_points: int = 10):
        """
        Verifies that two B-spline surfaces are G2 continuous along their shared edge
        """
        # Get the first derivatives at the boundary and perpendicular to the boundary for each surface,
        # evaluated at "n_points" locations along the boundary
        self_perp_edge_derivs = self.get_second_derivs_along_edge(surface_edge, n_points=n_points, perp=True)
        other_perp_edge_derivs = other.get_second_derivs_along_edge(other_surface_edge, n_points=n_points, perp=True)
        print(f"{self_perp_edge_derivs=},{other_perp_edge_derivs=}")
        self_perp_edge_derivs[np.absolute(self_perp_edge_derivs) < 1e-6] = 0.0
        other_perp_edge_derivs[np.absolute(other_perp_edge_derivs) < 1e-6] = 0.0

        ratios_other_self = other_perp_edge_derivs / self_perp_edge_derivs
        #print(f"{ratios_other_self=}")
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

    def transform(self, **transformation_kwargs) -> "BSplineSurface":
        """
        Creates a transformed copy of the surface by transforming each of the control points

        Parameters
        ----------
        transformation_kwargs
            Keyword arguments passed to :obj:`~aerocaps.geom.transformation.Transformation3D`

        Returns
        -------
        BSplineSurface
            Transformed surface
        """
        transformation = Transformation3D(**transformation_kwargs)
        initial_control_points = self.get_control_point_array()
        return BSplineSurface(
            np.array([transformation.transform(p_arr) for p_arr in initial_control_points]),
            knots_u=deepcopy(self.knots_u),
            knots_v=deepcopy(self.knots_v),
            name=self.name, construction=self.construction
        )

    def generate_control_point_net(self) -> (typing.List[Point3D], typing.List[Line3D]):
        """
        Generates a list of :obj:`~aerocaps.geom.point.Point3D` and :obj:`~aerocaps.geom.curves.Line3D` objects
        representing the NURBS surface's control points and connections between them

        Returns
        -------
        typing.List[Point3D], typing.List[Line3D]
            Control points and lines between adjacent control points in flattened lists
        """
        control_points = self.get_control_point_array()
        points = []
        lines = []

        for i in range(self.n_points_u):
            for j in range(self.n_points_v):
                points.append(Point3D.from_array(control_points[i, j, :]))

        for i in range(self.n_points_u - 1):
            for j in range(self.n_points_v - 1):
                point_obj_1 = Point3D.from_array(control_points[i, j, :])
                point_obj_2 = Point3D.from_array(control_points[i + 1, j, :])
                point_obj_3 = Point3D.from_array(control_points[i, j + 1, :])

                line_1 = Line3D(p0=point_obj_1, p1=point_obj_2)
                line_2 = Line3D(p0=point_obj_1, p1=point_obj_3)
                lines.extend([line_1, line_2])

                if i < self.n_points_u - 2 and j < self.n_points_v - 2:
                    continue

                point_obj_4 = Point3D.from_array(control_points[i + 1, j + 1, :])
                line_3 = Line3D(p0=point_obj_3, p1=point_obj_4)
                line_4 = Line3D(p0=point_obj_2, p1=point_obj_4)
                lines.extend([line_3, line_4])

        return points, lines

    def plot_surface(self, plot: pv.Plotter, Nu: int = 50, Nv: int = 50, **mesh_kwargs):
        """
        Plots the B-spline surface using the `pyvista <https://pyvista.org/>`_ library

        Parameters
        ----------
        plot:
            :obj:`pyvista.Plotter` instance
        Nu: int
            Number of points to evaluate in the :math:`u`-parametric direction. Default: ``50``
        Nv: int
            Number of points to evaluate in the :math:`v`-parametric direction. Default: ``50``
        mesh_kwargs:
            Keyword arguments to pass to :obj:`pyvista.Plotter.add_mesh`

        Returns
        -------
        pyvista.core.pointset.StructuredGrid
            The evaluated B-spline surface
        """
        XYZ = self.evaluate_grid(Nu, Nv)
        grid = pv.StructuredGrid(XYZ[:, :, 0], XYZ[:, :, 1], XYZ[:, :, 2])
        plot.add_mesh(grid, **mesh_kwargs)
        return grid

    def plot_control_point_mesh_lines(self, plot: pv.Plotter, **line_kwargs) -> pv.Actor:
        """
        Plots the network of lines connecting the B-spline surface control points using the
        `pyvista <https://pyvista.org/>`_ library

        Parameters
        ----------
        plot:
            :obj:`pyvista.Plotter` instance
        line_kwargs:
            Keyword arguments to pass to the :obj:`pyvista.Plotter.add_lines`

        Returns
        -------
        pv.Actor
            The lines actor
        """
        _, line_objs = self.generate_control_point_net()
        line_arr = np.array([[line_obj.p0.as_array(), line_obj.p1.as_array()] for line_obj in line_objs])
        line_arr = line_arr.reshape((len(line_objs) * 2, 3))
        line_actor = plot.add_lines(line_arr, **line_kwargs)
        return line_actor

    def plot_control_points(self, plot: pv.Plotter, **point_kwargs) -> pv.Actor:
        """
        Plots the B-spline surface control points using the `pyvista <https://pyvista.org/>`_ library

        Parameters
        ----------
        plot:
            :obj:`pyvista.Plotter` instance
        point_kwargs:
            Keyword arguments to pass to the :obj:`pyvista.Plotter.add_points`

        Returns
        -------
        pv.Actor
            The points actor
        """
        point_objs, _ = self.generate_control_point_net()
        point_arr = np.array([point_obj.as_array() for point_obj in point_objs])
        point_actor = plot.add_points(point_arr, **point_kwargs)
        return point_actor

    def __repr__(self):
        return (f"{self.name}: {self.degree_u} x {self.degree_v} {self.__class__.__name__} "
                f"({self.n_points_u} x {self.n_points_v} control points)")


class NURBSSurface(Surface):
    """
    NURBS surface class
    """

    def __init__(self,
                 points: typing.List[typing.List[Point3D]] or np.ndarray,
                 knots_u: np.ndarray,
                 knots_v: np.ndarray,
                 weights: np.ndarray,
                 name: str = "NURBSSurface",
                 construction: bool = False
                 ):
        """

        Parameters
        ----------
        points
        knots_u
        knots_v
        weights
        name: str
            Name of the geometric object. May be re-assigned a unique name when added to a
            :obj:`~aerocaps.geom.geometry_container.GeometryContainer`. Default: 'NURBSSurface'
        construction: bool
            Whether this is a geometry used only for construction of other geometries. If ``True``, this
            geometry will not be exported or plotted. Default: ``False``
        """
        if isinstance(points, np.ndarray):
            points = [[Point3D.from_array(pt_row) for pt_row in pt_mat] for pt_mat in points]
        self.points = points
        assert knots_u.ndim == 1
        assert knots_v.ndim == 1
        assert weights.ndim == 2
        assert len(points) == weights.shape[0]
        assert len(points[0]) == weights.shape[1]

        # Negative weight check
        for weight_row in weights:
            for weight in weight_row:
                if weight < 0:
                    raise NegativeWeightError("All weights must be non-negative")

        self.knots_u = deepcopy(knots_u)
        self.knots_v = deepcopy(knots_v)
        self.weights = deepcopy(weights)
        super().__init__(name=name, construction=construction)

    @property
    def n_points_u(self) -> int:
        """Number of control points in the :math:`u`-parametric direction"""
        return len(self.points)

    @property
    def n_points_v(self) -> int:
        """Number of control points in the :math:`v`-parametric direction"""
        return len(self.points[0])

    @property
    def degree_u(self) -> int:
        """Surface degree in the :math:`u`-parametric direction"""
        return len(self.knots_u) - self.n_points_u - 1

    @property
    def degree_v(self) -> int:
        """Surface degree in the :math:`v`-parametric direction"""
        return len(self.knots_v) - self.n_points_v - 1

    @property
    def n(self) -> int:
        """
        Shorthand for :obj:`~aerocaps.geom.surfaces.NURBSSurface.degree_u`

        Returns
        -------
        int
            Surface degree in the :math:`u`-parametric direction
        """
        return self.degree_u

    @property
    def m(self) -> int:
        """
        Shorthand for :obj:`~aerocaps.geom.surfaces.NURBSSurface.degree_v`

        Returns
        -------
        int
            Surface degree in the :math:`v`-parametric direction
        """
        return self.degree_v

    def to_iges(self, *args, **kwargs) -> aerocaps.iges.entity.IGESEntity:
        """
        Exports the NURBS surface to an IGES entity
        """
        return aerocaps.iges.surfaces.RationalBSplineSurfaceIGES(
            control_points=self.get_control_point_array(),
            knots_u=self.knots_u,
            knots_v=self.knots_v,
            weights=self.weights,
            degree_u=self.degree_u,
            degree_v=self.degree_v
        )

    def get_control_point_array(self) -> np.ndarray:
        r"""
        Gets the control points in float array form.

        Returns
        -------
        numpy.ndarray
            Array of size :math:`N_u \times N_v \times 3`
        """
        return np.array([np.array([p.as_array() for p in p_arr]) for p_arr in self.points])

    def get_homogeneous_control_points(self) -> np.ndarray:
        r"""
        Gets the array of control points in homogeneous coordinates, :math:`\mathbf{P}_{i,j} \cdot w_{i,j}`

        Returns
        -------
        numpy.ndarray
            Array of size :math:`N_u \times N_v \times 4`.
            The four elements of the last array dimension are, in order,
            the :math:`x`-coordinate, :math:`y`-coordinate, :math:`z`-coordinate, and weight of each
            control point.
        """
        return np.dstack((
            self.get_control_point_array() * np.repeat(self.weights[:, :, np.newaxis], 3, axis=2),
            self.weights
        ))

    @staticmethod
    def project_homogeneous_control_points(homogeneous_points: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Projects the homogeneous coordinates onto the :math:`w=1` hyperplane.

        Returns
        -------
        numpy.ndarray, numpy.ndarray
            The projected coordinates in three-dimensional space followed by the weight array
        """
        P = homogeneous_points[:, :, :3] / np.repeat(homogeneous_points[:, :, -1][:, :, np.newaxis], 3, axis=2)
        w = homogeneous_points[:, :, -1]
        return P, w

    @classmethod
    def from_bezier_revolve(cls, bezier: BezierCurve3D, axis: Line3D,
                            start_angle: Angle, end_angle: Angle) -> "NURBSSurface":
        """
        Creates a NURBS surface from the revolution of a Bézier curve about an axis.

        Parameters
        ----------
        bezier: BezierCurve3D
            Bézier curve to revolve
        axis: Line3D
            Axis of revolution
        start_angle: Angle
            Starting angle for the revolve
        end_angle: Angle
            Ending angle for the revolve

        Returns
        -------
        NURBSSurface
            Surface of revolution
        """

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

        knots_v = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        n_knots_to_insert = len(angles) - 3
        if n_knots_to_insert > 0:
            delta = 1.0 / (n_knots_to_insert / 2 + 1)
            for idx in range(n_knots_to_insert):
                new_knot = knots_v[2 + idx] if idx % 2 else knots_v[2 + idx] + delta
                knots_v = np.insert(knots_v, 3 + idx, new_knot)

        knots_u = np.array([0.0 for _ in bezier.control_points] + [1.0 for _ in bezier.control_points])

        return cls(control_points, knots_u, knots_v, weights)

    def evaluate(self, u: float, v: float) -> np.ndarray:
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
        return np.array(nurbs_surf_eval(P, self.weights, self.knots_u, self.knots_v, u, v))

    def evaluate_point3d(self, u: float, v: float) -> Point3D:
        r"""
        Evaluates the NURBS surface at a single :math:`(u,v)` parameter pair and returns a point object.

        Parameters
        ----------
        u: float
            Position along :math:`u` in parametric space. Normally in the range :math:`[0,1]`
        v: float
            Position along :math:`v` in parametric space. Normally in the range :math:`[0,1]`

        Returns
        -------
        Point3D
            Point object corresponding to the :math:`(u,v)` pair
        """
        return Point3D.from_array(self.evaluate(u, v))

    def evaluate_grid(self, Nu: int, Nv: int) -> np.ndarray:
        r"""
        Evaluates the NURBS surface on a uniform :math:`N_u \times N_v` grid of parameter values.

        Parameters
        ----------
        Nu: int
            Number of uniformly spaced parameter values in the :math:`u`-direction
        Nv: int
            Number of uniformly spaced parameter values in the :math:`v`-direction

        Returns
        -------
        numpy.ndarray
            Array of size :math:`N_u \times N_v \times 3`
        """
        P = self.get_control_point_array()
        return np.array(nurbs_surf_eval_grid(P, self.weights, self.knots_u, self.knots_v, Nu, Nv))

    def get_parallel_control_point_length(self, surface_edge: SurfaceEdge) -> int:
        r"""
        Gets the number of control points of the curve corresponding to the input surface edge.

        Parameters
        ----------
        surface_edge: SurfaceEdge
            Edge along which the number of control points is computed

        Returns
        -------
        int
            Number of control points parallel to the edge
        """
        if surface_edge in [SurfaceEdge.v1, SurfaceEdge.v0]:
            return self.n_points_u
        return self.n_points_v

    def get_perpendicular_control_point_length(self, surface_edge: SurfaceEdge) -> int:
        r"""
        Gets the number of control points in the parametric direction perpendicular to the input surface edge.

        Parameters
        ----------
        surface_edge: SurfaceEdge
            Edge along which the number of perpendicular control points is computed

        Returns
        -------
        int
            Number of control points perpendicular to the edge
        """
        if surface_edge in [SurfaceEdge.v1, SurfaceEdge.v0]:
            return self.n_points_v
        return self.n_points_u

    def get_parallel_degree(self, surface_edge: SurfaceEdge) -> int:
        r"""
        Gets the degree of the curve corresponding to the input surface edge.

        Parameters
        ----------
        surface_edge: SurfaceEdge
            Edge along which the parallel degree is evaluated

        Returns
        -------
        int
            Degree parallel to the edge
        """
        if surface_edge in [SurfaceEdge.v1, SurfaceEdge.v0]:
            return self.degree_u
        return self.degree_v

    def get_perpendicular_degree(self, surface_edge: SurfaceEdge) -> int:
        r"""
        Gets the degree of the curve in the parametric direction perpendicular to the input surface edge.

        Parameters
        ----------
        surface_edge: SurfaceEdge
            Edge along which the perpendicular degree is evaluated

        Returns
        -------
        int
            Degree perpendicular to the edge
        """
        if surface_edge in [SurfaceEdge.v1, SurfaceEdge.v0]:
            return self.degree_v
        return self.degree_u

    def get_parallel_knots(self, surface_edge: SurfaceEdge) -> np.ndarray:
        r"""
        Gets the knots in the parametric direction parallel to the input surface edge.

        Parameters
        ----------
        surface_edge: SurfaceEdge
            Edge along which the parallel knots are returned

        Returns
        -------
        numpy.ndarray
            Knots parallel to the edge
        """
        if surface_edge in [SurfaceEdge.v1, SurfaceEdge.v0]:
            return self.knots_u
        return self.knots_v

    def get_perpendicular_knots(self, surface_edge: SurfaceEdge) -> np.ndarray:
        r"""
        Gets the knots in the parametric direction perpendicular to the input surface edge.

        Parameters
        ----------
        surface_edge: SurfaceEdge
            Edge along which the perpendicular knots are returned

        Returns
        -------
        numpy.ndarray
            Knots perpendicular to the edge
        """
        if surface_edge in [SurfaceEdge.v1, SurfaceEdge.v0]:
            return self.knots_v
        return self.knots_u

    def get_point(self, row_index: int, continuity_index: int, surface_edge: SurfaceEdge) -> Point3D:
        r"""
        Gets the point corresponding to a particular index along the edge curve with perpendicular index
        corresponding to the level of continuity being applied. For example, for a :math:`6 \times 5` NURBS surface,
        the following code

        .. code-block:: python

            p = surf.get_point(2, 1, ac.SurfaceEdge.v0)

        returns the point :math:`\mathbf{P}_{2,1}` and

        .. code-block:: python

            p = surf.get_point(2, 1, ac.SurfaceEdge.u1)

        returns the point :math:`\mathbf{P}_{6-1,2} = \mathbf{P}_{5,2}` if there are no internal knot vectors.
        If the NURBS surface has internal knot vectors, the actual :math:`i`-index of the point may be different,
        but the second-to-last point in the third row of control points will still be returned.

        .. seealso::

            :obj:`~aerocaps.geom.surfaces.NURBSSurface.set_point`
                Setter equivalent of this method

        Parameters
        ----------
        row_index: int
            Index along the surface edge control points
        continuity_index: int
            Index in the parametric direction perpendicular to the surface edge. Normally either ``0``, ``1``, or ``2``
        surface_edge: SurfaceEdge
            Edge of the surface along which to retrieve the control point

        Returns
        -------
        Point3D
            Point used to enforce :math:`G^x` continuity, where :math:`x` is the value of ``continuity_index``
        """
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
        r"""
        Sets the point corresponding to a particular index along the edge curve with perpendicular index
        corresponding to the level of continuity being applied. For example, for a :math:`6 \times 5` NURBS surface,
        the following code

        .. code-block:: python

            p = ac.Point3D.from_array(np.array([3.0, 4.0, 5.0]))
            surf.set_point(p, 2, 1, ac.SurfaceEdge.v0)

        sets the value of point :math:`\mathbf{P}_{2,1}` to :math:`[3,4,5]^T` and

        .. code-block:: python

            p = ac.Point3D.from_array(np.array([3.0, 4.0, 5.0]))
            surf.get_point(p, 2, 1, ac.SurfaceEdge.u1)

        sets the value of point :math:`\mathbf{P}_{6-1,2} = \mathbf{P}_{5,2}` to :math:`[3,4,5]^T` if there are no
        internal knot vectors.

        .. seealso::

            :obj:`~aerocaps.geom.surfaces.NURBSSurface.get_point`
                Getter equivalent of this method

        Parameters
        ----------
        point: Point3D
            Point object to apply at the specified indices
        row_index: int
            Index along the surface edge control points
        continuity_index: int
            Index in the parametric direction perpendicular to the surface edge. Normally either ``0``, ``1``, or ``2``
        surface_edge: SurfaceEdge
            Edge of the surface along which to retrieve the control point
        """
        if surface_edge == SurfaceEdge.v1:
            self.points[row_index][-(continuity_index + 1)].x.m = point.x.m
            self.points[row_index][-(continuity_index + 1)].y.m = point.y.m
            self.points[row_index][-(continuity_index + 1)].z.m = point.z.m
        elif surface_edge == SurfaceEdge.v0:
            self.points[row_index][continuity_index].x.m = point.x.m
            self.points[row_index][continuity_index].y.m = point.y.m
            self.points[row_index][continuity_index].z.m = point.z.m
        elif surface_edge == SurfaceEdge.u1:
            self.points[-(continuity_index + 1)][row_index].x.m = point.x.m
            self.points[-(continuity_index + 1)][row_index].y.m = point.y.m
            self.points[-(continuity_index + 1)][row_index].z.m = point.z.m
        elif surface_edge == SurfaceEdge.u0:
            self.points[continuity_index][row_index].x.m = point.x.m
            self.points[continuity_index][row_index].y.m = point.y.m
            self.points[continuity_index][row_index].z.m = point.z.m
        else:
            raise ValueError("Invalid surface_edge value")

    def get_weight(self, row_index: int, continuity_index: int, surface_edge: SurfaceEdge):
        r"""
        Gets the weight corresponding to a particular index along the edge curve with perpendicular index
        corresponding to the level of continuity being applied. For example, for a :math:`6 \times 5` NURBS surface,
        the following code

        .. code-block:: python

            w = surf.get_weight(2, 1, ac.SurfaceEdge.v0)

        returns the weight :math:`w_{2,1}` and

        .. code-block:: python

            w = surf.get_weight(2, 1, ac.SurfaceEdge.u1)

        returns the weight :math:`w_{6-1,2} = w_{5,2}`.

        .. seealso::

            :obj:`~aerocaps.geom.surfaces.NURBSSurface.set_weight`
                Setter equivalent of this method

        Parameters
        ----------
        row_index: int
            Index along the surface edge weights
        continuity_index: int
            Index in the parametric direction perpendicular to the surface edge. Normally either ``0``, ``1``, or ``2``
        surface_edge: SurfaceEdge
            Edge of the surface along which to retrieve the weight

        Returns
        -------
        float
            Weight used to enforce :math:`G^x` continuity, where :math:`x` is the value of ``continuity_index``
        """
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
        r"""
        Sets the weight corresponding to a particular index along the edge curve with perpendicular index
        corresponding to the level of continuity being applied. For example, for a :math:`6 \times 5`
        NURBS surface, the following code

        .. code-block:: python

            surf.set_weight(0.9, 2, 1, ac.SurfaceEdge.v0)

        sets the value of weight :math:`w_{2,1}` to :math:`0.9` and

        .. code-block:: python

            surf.set_weight(1.1, 2, 1, ac.SurfaceEdge.u1)

        sets the value of weight :math:`w_{6-1,2} = w_{5,2}` to :math:`1.1`.

        .. seealso::

            :obj:`~aerocaps.geom.surfaces.NURBSSurface.get_weight`
                Getter equivalent of this method

        Parameters
        ----------
        weight: float
            Weight to apply at the specified indices
        row_index: int
            Index along the surface edge weights
        continuity_index: int
            Index in the parametric direction perpendicular to the surface edge. Normally either ``0``, ``1``, or ``2``
        surface_edge: SurfaceEdge
            Edge of the surface along which to retrieve the weight
        """
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

    def extract_edge_curve(self, surface_edge: SurfaceEdge) -> NURBSCurve3D:
        """
        Extracts the control points, weights, and knots from one of the four edges of the NURBS surface and
        outputs a NURBS curve with these control points and weights

        Parameters
        ----------
        surface_edge: SurfaceEdge
            Edge along which to extract the curve

        Returns
        -------
        NURBSCurve3D
            NURBS curve with control points, weights, and knots corresponding to the control points, weights, and knots
            along the edge of the surface
        """
        P = self.get_control_point_array()
        w = self.weights

        if surface_edge == SurfaceEdge.u0:
            return NURBSCurve3D(P[0, :, :], w[0, :], self.knots_v, self.degree_v)
        if surface_edge == SurfaceEdge.u1:
            return NURBSCurve3D(P[-1, :, :], w[-1, :], self.knots_v, self.degree_v)
        if surface_edge == SurfaceEdge.v0:
            return NURBSCurve3D(P[:, 0, :], w[:, 0], self.knots_u, self.degree_u)
        if surface_edge == SurfaceEdge.v1:
            return NURBSCurve3D(P[:, -1, :], w[:, -1], self.knots_u, self.degree_u)

        raise ValueError(f"Invalid surface edge {surface_edge}")

    def is_clamped(self, surface_edge: SurfaceEdge) -> bool:
        """
        Checks if the NURBS surface is clamped along an edge

        Parameters
        ----------
        surface_edge: SurfaceEdge
            Edge where the perpendicular knots will be inspected

        Returns
        -------
        bool
            Whether the surface is clamped at the given edge
        """
        p = self.get_perpendicular_degree(surface_edge)
        knots = self.get_perpendicular_knots(surface_edge)
        if surface_edge in (SurfaceEdge.u0, SurfaceEdge.v0):
            start_knot = knots[0]
            if np.all(knots[:(p + 1)] == start_knot):
                return True
            return False
        end_knot = knots[-1]
        if np.all(knots[-(p + 1):] == end_knot):
            return True
        return False

    def has_internal_knots(self, direction: str) -> bool:
        """
        Whether the surface has internal knots in the specified direction

        Parameters
        ----------
        direction: str
            Either 'u' or 'v'

        Returns
        -------
        bool
        """
        if direction == "u":
            return len(set(self.knots_u)) > 2
        elif direction == "v":
            return len(set(self.knots_v)) > 2
        else:
            raise ValueError(f"Invalid direction {direction}. Must be either 'u' or 'v'")

    def enforce_g0(self, other: "NURBSSurface",
                   surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge):
        r"""
        Enforces :math:`G^0` continuity along the input ``surface_edge`` by equating the control points
        and weights along this edge to the corresponding control points and weights along the ``other_surface_edge``
        of the NURBS surface given by ``other``.
        The control points of the surface from which this method is called are modified in-place, and the control
        points of ``other`` are left unchanged.

        .. important::

            The parallel degree of the current surface along ``surface_edge`` must be equal to the parallel degree
            of the ``other`` surface along ``other_surface_edge``, otherwise an ``AssertionError`` will be raised.
            Additionally, the knot vector along the ``surface_edge`` of the current surface must be equal
            to the knot vector along the ``other_surface_edge`` of the other surface.

        .. seealso::

            :obj:`~aerocaps.geom.surfaces.NURBSSurface.enforce_c0`
                Parametric continuity equivalent (:math:`C^0`)

        Parameters
        ----------
        other: NURBSSurface
            Another NURBS surface along which an edge will be used for stitching
        surface_edge: SurfaceEdge
            The edge of the current surface to modify
        other_surface_edge: SurfaceEdge
            Tool edge of surface ``other`` which determines the positions of control points along ``surface_edge``
            of the current surface
        """
        # P^b[:, 0] = P^a[:, -1]
        self_parallel_knots = self.get_parallel_knots(surface_edge)
        other_parallel_knots = other.get_parallel_knots(other_surface_edge)
        self_parallel_degree = self.get_parallel_degree(surface_edge)
        other_parallel_degree = other.get_parallel_degree(other_surface_edge)

        # Check clamped for self:
        if not self.is_clamped(surface_edge):
            raise ValueError(" The self surface is not clamped on the edge that has to be stitched")

        #Check clamped for other:
        if not other.is_clamped(other_surface_edge):
            raise ValueError(" The self surface is not clamped on the edge that has to be stitched")

        if len(self_parallel_knots) != len(other_parallel_knots):
            raise ValueError(f"Length of the knot vector parallel to the edge of the input surface "
                             f"({len(self_parallel_knots)}) is not equal to the length of the knot vector "
                             f"parallel to the edge of the other surface ({len(other_parallel_knots)})")
        if any([k1 != k2 for k1, k2 in zip(self_parallel_knots, other_parallel_knots)]):
            raise ValueError(f"Knots parallel to the edge of the input surface ({self_parallel_knots}) "
                             f"are not equal to the knots in the direction parallel to the edge of the other "
                             f"surface ({other_parallel_knots})")
        if self_parallel_degree != other_parallel_degree:
            raise ValueError(f"Degree parallel to the edge of the input surface ({self_parallel_degree}) does "
                             f"not match the degree parallel to the edge of the other surface "
                             f"({other_parallel_degree})")
        for row_index in range(self.get_parallel_control_point_length(surface_edge)):
            self.set_point(other.get_point(row_index, 0, other_surface_edge), row_index, 0, surface_edge)
            self.set_weight(other.get_weight(row_index, 0, other_surface_edge), row_index, 0, surface_edge)

    def enforce_c0(self, other: "NURBSSurface", surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge):
        """
        For zeroth-degree continuity, there is no difference between geometric (:math:`G^0`) and parametric
        (:math:`C^0`) continuity. Because this method is simply a convenience method that calls
        :obj:`~aerocaps.geom.surfaces.NURBSSurface.enforce_g0`, see the documentation for that method for more
        detailed documentation.

        Parameters
        ----------
        other: NURBSSurface
            Another NURBS surface along which an edge will be used for stitching
        surface_edge: SurfaceEdge
            The edge of the current surface to modify
        other_surface_edge: SurfaceEdge
            Tool edge of surface ``other`` which determines the positions of control points along ``surface_edge``
            of the current surface
        """
        self.enforce_g0(other, surface_edge, other_surface_edge)

    def enforce_g0g1(self, other: "NURBSSurface", f: float,
                     surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge):
        r"""
        First enforces :math:`G^0` continuity, then tangent (:math:`G^1`) continuity is enforced according to
        the following equations:

        .. math::

            \mathcal{W}^{b,\mathcal{E}_b}_{k,1} = \mathcal{W}^{b,\mathcal{E}_b}_{k,0} + f \frac{p_{\perp}^{a,\mathcal{E}_a}}{p_{\perp}^{b,\mathcal{E}_b}} \left( \mathcal{W}^{a,\mathcal{E}_a}_{k,0} - \mathcal{W}^{a,\mathcal{E}_a}_{k,1} \right) \text{ for }k=0,1,\ldots,p_{\parallel}^{b,\mathcal{E}_b}

        .. math::

            \mathcal{P}^{b,\mathcal{E}_b}_{k,1} = \frac{\mathcal{W}^{b,\mathcal{E}_b}_{k,0}}{\mathcal{W}^{b,\mathcal{E}_b}_{k,1}} \mathcal{P}^{b,\mathcal{E}_b}_{k,0} + f \frac{p_{\perp}^{a,\mathcal{E}_a}}{p_{\perp}^{b,\mathcal{E}_b}} \frac{1}{\mathcal{W}^{b,\mathcal{E}_b}_{k,1}} \left[\mathcal{W}^{a,\mathcal{E}_a}_{k,0} \mathcal{P}^{a,\mathcal{E}_a}_{k,0} - \mathcal{P}^{a,\mathcal{E}_a}_{k,1} \mathcal{W}^{a,\mathcal{E}_a}_{k,1} \right] \text{ for }k=0,1,\ldots,p_{\parallel}^{b,\mathcal{E}_b}

        Here, :math:`b` corresponds to the current surface, and :math:`a` corresponds to the ``other`` surface.
        The control points of the surface from which this method is called are modified in-place, and the control
        points of ``other`` are left unchanged.

        .. seealso::

            :obj:`~aerocaps.geom.surfaces.NURBSSurface.enforce_g0`
                Geometric point continuity enforcement (:math:`G^0`)
            :obj:`~aerocaps.geom.surfaces.NURBSSurface.enforce_c0c1`
                Parametric continuity equivalent (:math:`C^1`)

        Parameters
        ----------
        other: NURBSSurface
            Another NURBS surface along which an edge will be used for stitching
        f: float
            Tangent proportionality factor
        surface_edge: SurfaceEdge
            The edge of the current surface to modify
        other_surface_edge: SurfaceEdge
            Tool edge of surface ``other`` which determines the positions of control points along ``surface_edge``
            of the current surface
        """
        self.enforce_g0(other, surface_edge, other_surface_edge)

        n_ratio = other.get_perpendicular_degree(other_surface_edge) / self.get_perpendicular_degree(surface_edge)
        for row_index in range(self.get_parallel_control_point_length(surface_edge)):

            w_i0_b = self.get_weight(row_index, 0, surface_edge)
            w_im_a = other.get_weight(row_index, 0, other_surface_edge)
            w_im1_a = other.get_weight(row_index, 1, other_surface_edge)

            w_i1_b = w_i0_b + f * n_ratio * (w_im_a - w_im1_a)

            if w_i1_b < 0:
                raise NegativeWeightError("G1 enforcement generated a negative weight")

            self.set_weight(w_i1_b, row_index, 1, surface_edge)

            P_i0_b = self.get_point(row_index, 0, surface_edge)
            P_im_a = other.get_point(row_index, 0, other_surface_edge)
            P_im1_a = other.get_point(row_index, 1, other_surface_edge)

            P_i1_b = w_i0_b / w_i1_b * P_i0_b + f * n_ratio / w_i1_b * (w_im_a * P_im_a - w_im1_a * P_im1_a)
            self.set_point(P_i1_b, row_index, 1, surface_edge)

    def enforce_c0c1(self, other: "NURBSSurface",
                     surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge):
        r"""
        Equivalent to calling :obj:`~aerocaps.geom.surfaces.NURBSSurface.enforce_g0g1` with ``f=1.0``. See that
        method for more detailed documentation.

        Parameters
        ----------
        other: NURBSSurface
            Another NURBS surface along which an edge will be used for stitching
        surface_edge: SurfaceEdge
            The edge of the current surface to modify
        other_surface_edge: SurfaceEdge
            Tool edge of surface ``other`` which determines the positions of control points along ``surface_edge``
            of the current surface
        """
        self.enforce_g0g1(other, 1.0, surface_edge, other_surface_edge)

    def enforce_g0g1g2(self, other: "NURBSSurface", f: float,
                       surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge):
        r"""
        First enforces :math:`G^0` and :math:`G^1` continuity, then curvature (:math:`G^2`) continuity is enforced
        according to the following equations:

        .. math::

            \mathcal{W}^{b,\mathcal{E}_b}_{k,2} = 2 \mathcal{W}^{b,\mathcal{E}_b}_{k,1} - \mathcal{W}^{b,\mathcal{E}_b}_{k,0} + f^2 \frac{p_{\perp}^{a,\mathcal{E}_a}(p_{\perp}^{a,\mathcal{E}_a}-1)}{p_{\perp}^{b,\mathcal{E}_b}(p_{\perp}^{b,\mathcal{E}_b}-1)} \left[ \mathcal{W}^{a,\mathcal{E}_a}_{k,0} - 2 \mathcal{W}^{a,\mathcal{E}_a}_{k,1} + \mathcal{W}^{a,\mathcal{E}_a}_{k,2} \right]  \text{ for }k=0,1,\ldots,p_{\parallel}^{b,\mathcal{E}_b}

        .. math::

            \mathcal{P}^{b,\mathcal{E}_b}_{k,2} = 2 \frac{\mathcal{W}^{b,\mathcal{E}_b}_{k,1}}{\mathcal{W}^{b,\mathcal{E}_b}_{k,2}} \mathcal{P}^{b,\mathcal{E}_b}_{k,1} - \frac{\mathcal{W}^{b,\mathcal{E}_b}_{k,0}}{\mathcal{W}^{b,\mathcal{E}_b}_{k,2}} \mathcal{P}^{b,\mathcal{E}_b}_{k,0} + f^2 \frac{p_{\perp}^{a,\mathcal{E}_a}(p_{\perp}^{a,\mathcal{E}_a}-1)}{p_{\perp}^{b,\mathcal{E}_b}(p_{\perp}^{b,\mathcal{E}_b}-1)} \frac{1}{\mathcal{W}^{b,\mathcal{E}_b}_{k,2}} \left[ \mathcal{W}^{a,\mathcal{E}_a}_{k,1} \mathcal{P}^{a,\mathcal{E}_a}_{k,0} - 2 \mathcal{W}^{a,\mathcal{E}_a}_{k,1} \mathcal{P}^{a,\mathcal{E}_a}_{k,1} + \mathcal{W}^{a,\mathcal{E}_a}_{k,2} \mathcal{P}^{a,\mathcal{E}_a}_{k,2} \right]  \text{ for }k=0,1,\ldots,p_{\parallel}^{b,\mathcal{E}_b}

        Here, :math:`b` corresponds to the current surface, and :math:`a` corresponds to the ``other`` surface.
        The control points of the surface from which this method is called are modified in-place, and the control
        points of ``other`` are left unchanged.

        .. seealso::

         :obj:`~aerocaps.geom.surfaces.NURBSSurface.enforce_g0`
             Geometric point continuity enforcement (:math:`G^0`)
         :obj:`~aerocaps.geom.surfaces.NURBSSurface.enforce_g0g1`
             Geometric tangent continuity enforcement (:math:`G^1`)
         :obj:`~aerocaps.geom.surfaces.NURBSSurface.enforce_c0c1c2`
             Parametric continuity equivalent (:math:`C^2`)

        Parameters
        ----------
        other: NURBSSurface
            Another NURBS surface along which an edge will be used for stitching
        f: float
            Tangent proportionality factor
        surface_edge: SurfaceEdge
            The edge of the current surface to modify
        other_surface_edge: SurfaceEdge
            Tool edge of surface ``other`` which determines the positions of control points along ``surface_edge``
            of the current surface
        """

        self.enforce_g0g1(other, f, surface_edge, other_surface_edge)
        n_ratio = (other.get_perpendicular_degree(other_surface_edge) ** 2 -
                   other.get_perpendicular_degree(other_surface_edge)) / (
                          self.get_perpendicular_degree(surface_edge) ** 2 - self.get_perpendicular_degree(
                      surface_edge))
        for row_index in range(self.get_parallel_control_point_length(surface_edge)):

            w_i0_b = self.get_weight(row_index, 0, surface_edge)
            w_i1_b = self.get_weight(row_index, 1, surface_edge)
            w_im_a = other.get_weight(row_index, 0, other_surface_edge)
            w_im1_a = other.get_weight(row_index, 1, other_surface_edge)
            w_im2_a = other.get_weight(row_index, 2, other_surface_edge)

            w_i2_b = 2.0 * w_i1_b - w_i0_b + f ** 2 * n_ratio * (w_im_a - 2.0 * w_im1_a + w_im2_a)

            if w_i2_b < 0:
                raise NegativeWeightError("G2 enforcement generated a negative weight")

            self.set_weight(w_i2_b, row_index, 2, surface_edge)

            P_i0_b = self.get_point(row_index, 0, surface_edge)
            P_i1_b = self.get_point(row_index, 1, surface_edge)
            P_im_a = other.get_point(row_index, 0, other_surface_edge)
            P_im1_a = other.get_point(row_index, 1, other_surface_edge)
            P_im2_a = other.get_point(row_index, 2, other_surface_edge)

            P_i2_b = (2.0 * w_i1_b / w_i2_b * P_i1_b - w_i0_b / w_i2_b * P_i0_b) + f ** 2 * n_ratio * (
                    1 / w_i2_b) * (
                             w_im_a * P_im_a - 2.0 * w_im1_a * P_im1_a + w_im2_a * P_im2_a)
            self.set_point(P_i2_b, row_index, 2, surface_edge)

    def enforce_c0c1c2(self, other: "NURBSSurface",
                       surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge):
        r"""
        Equivalent to calling :obj:`~aerocaps.geom.surfaces.NURBSSurface.enforce_g0g1g2` with ``f=1.0``.
        See that method for more detailed documentation.

        Parameters
        ----------
        other: NURBSSurface
            Another NURBS surface along which an edge will be used for stitching
        surface_edge: SurfaceEdge
            The edge of the current surface to modify
        other_surface_edge: SurfaceEdge
            Tool edge of surface ``other`` which determines the positions of control points along ``surface_edge``
            of the current surface
        """
        self.enforce_g0g1g2(other, 1.0, surface_edge, other_surface_edge)

    def dSdu(self, u: float, v: float) -> np.ndarray:
        r"""
        Evaluates the first derivative with respect to :math:`u` at a single :math:`(u,v)` pair

        Parameters
        ----------
        u: float
            Position along :math:`u` in parametric space. Normally in the range :math:`[0,1]`
        v: float
            Position along :math:`v` in parametric space. Normally in the range :math:`[0,1]`

        Returns
        -------
        np.ndarray
            1-D array containing the :math:`x`-, :math:`y`-, and :math:`z`-components of the second derivative
        """
        P = self.get_control_point_array()
        return np.array(nurbs_surf_dsdu(P, self.weights, self.knots_u, self.knots_v, u, v))

    def dSdu_grid(self, Nu: int, Nv: int) -> np.ndarray:
        r"""
        Evaluates the first derivative with respect to :math:`u` on a linearly-spaced grid of :math:`u`- and
        :math:`v`-values.

        Parameters
        ----------
        Nu: int
            Number of evenly spaced :math:`u` values
        Nv: int
            Number of evenly spaced :math:`v` values

        Returns
        -------
        np.ndarray
            Array of size :math:`N_u \times N_v \times 3`
        """
        P = self.get_control_point_array()
        return np.array(nurbs_surf_dsdu_grid(P, self.weights, self.knots_u, self.knots_v, Nu, Nv))

    def dSdu_uvvecs(self, u: np.ndarray, v: np.ndarray):
        r"""
        Evaluates the first derivative of the surface with respect to :math:`u` at arbitrary vectors of
        :math:`u` and :math:`v`-values.

        Parameters
        ----------
        u: np.ndarray
            1-D array of :math:`u`-parameter values
        v: np.ndarray
            1-D array of :math:`v`-parameter values

        Returns
        -------
        np.ndarray
            Array of size :math:`\text{len}(u) \times \text{len}(v) \times 3`
        """
        P = self.get_control_point_array()
        return np.array(nurbs_surf_dsdu_uvvecs(P, self.weights, self.knots_u, self.knots_v, u, v))

    def dSdv(self, u: float or np.ndarray, v: float or np.ndarray):
        r"""
        Evaluates the first derivative with respect to :math:`v` at a single :math:`(u,v)` pair

        Parameters
        ----------
        u: float
            Position along :math:`u` in parametric space. Normally in the range :math:`[0,1]`
        v: float
            Position along :math:`v` in parametric space. Normally in the range :math:`[0,1]`

        Returns
        -------
        np.ndarray
            1-D array containing the :math:`x`-, :math:`y`-, and :math:`z`-components of the second derivative
        """
        P = self.get_control_point_array()
        return np.array(nurbs_surf_dsdv(P, self.weights, self.knots_u, self.knots_v, u, v))

    def dSdv_grid(self, Nu: int, Nv: int) -> np.ndarray:
        r"""
        Evaluates the first derivative with respect to :math:`v` on a linearly-spaced grid of :math:`u`- and
        :math:`v`-values.

        Parameters
        ----------
        Nu: int
            Number of evenly spaced :math:`u` values
        Nv: int
            Number of evenly spaced :math:`v` values

        Returns
        -------
        np.ndarray
            Array of size :math:`N_u \times N_v \times 3`
        """
        P = self.get_control_point_array()
        return np.array(nurbs_surf_dsdv_grid(P, self.weights, self.knots_u, self.knots_v, Nu, Nv))

    def dSdv_uvvecs(self, u: np.ndarray, v: np.ndarray):
        r"""
        Evaluates the first derivative of the surface with respect to :math:`v` at arbitrary vectors of
        :math:`u` and :math:`v`-values.

        Parameters
        ----------
        u: np.ndarray
            1-D array of :math:`u`-parameter values
        v: np.ndarray
            1-D array of :math:`v`-parameter values

        Returns
        -------
        np.ndarray
            Array of size :math:`\text{len}(u) \times \text{len}(v) \times 3`
        """
        P = self.get_control_point_array()
        return np.array(nurbs_surf_dsdv_uvvecs(P, self.weights, self.knots_u, self.knots_v, u, v))

    def d2Sdu2(self, u: float, v: float) -> np.ndarray:
        r"""
        Evaluates the second derivative with respect to :math:`u` at a single :math:`(u,v)` pair

        Parameters
        ----------
        u: float
            Position along :math:`u` in parametric space. Normally in the range :math:`[0,1]`
        v: float
            Position along :math:`v` in parametric space. Normally in the range :math:`[0,1]`

        Returns
        -------
        np.ndarray
            1-D array containing the :math:`x`-, :math:`y`-, and :math:`z`-components of the second derivative
        """
        P = self.get_control_point_array()
        return np.array(nurbs_surf_d2sdu2(P, self.weights, self.knots_u, self.knots_v, u, v))

    def d2Sdu2_grid(self, Nu: int, Nv: int) -> np.ndarray:
        r"""
        Evaluates the second derivative with respect to :math:`u` on a linearly-spaced grid of :math:`u`- and
        :math:`v`-values.

        Parameters
        ----------
        Nu: int
            Number of evenly spaced :math:`u` values
        Nv: int
            Number of evenly spaced :math:`v` values

        Returns
        -------
        np.ndarray
            Array of size :math:`N_u \times N_v \times 3`
        """
        P = self.get_control_point_array()
        return np.array(nurbs_surf_d2sdu2_grid(P, self.weights, self.knots_u, self.knots_v, Nu, Nv))

    def d2Sdu2_uvvecs(self, u: np.ndarray, v: np.ndarray):
        r"""
        Evaluates the second derivative of the surface with respect to :math:`u` at arbitrary vectors of
        :math:`u` and :math:`v`-values.

        Parameters
        ----------
        u: np.ndarray
            1-D array of :math:`u`-parameter values
        v: np.ndarray
            1-D array of :math:`v`-parameter values

        Returns
        -------
        np.ndarray
            Array of size :math:`\text{len}(u) \times \text{len}(v) \times 3`
        """
        P = self.get_control_point_array()
        return np.array(nurbs_surf_d2sdu2_uvvecs(P, self.weights, self.knots_u, self.knots_v, u, v))

    def d2Sdv2(self, u: float or np.ndarray, v: float or np.ndarray):
        r"""
        Evaluates the second derivative with respect to :math:`v` at a single :math:`(u,v)` pair

        Parameters
        ----------
        u: float
            Position along :math:`u` in parametric space. Normally in the range :math:`[0,1]`
        v: float
            Position along :math:`v` in parametric space. Normally in the range :math:`[0,1]`

        Returns
        -------
        np.ndarray
            1-D array containing the :math:`x`-, :math:`y`-, and :math:`z`-components of the second derivative
        """
        P = self.get_control_point_array()
        return np.array(nurbs_surf_d2sdv2(P, self.weights, self.knots_u, self.knots_v, u, v))

    def d2Sdv2_grid(self, Nu: int, Nv: int) -> np.ndarray:
        r"""
        Evaluates the second derivative with respect to :math:`v` on a linearly-spaced grid of :math:`u`- and
        :math:`v`-values.

        Parameters
        ----------
        Nu: int
            Number of evenly spaced :math:`u` values
        Nv: int
            Number of evenly spaced :math:`v` values

        Returns
        -------
        np.ndarray
            Array of size :math:`N_u \times N_v \times 3`
        """
        P = self.get_control_point_array()
        return np.array(nurbs_surf_d2sdv2_grid(P, self.weights, self.knots_u, self.knots_v, Nu, Nv))

    def d2Sdv2_uvvecs(self, u: np.ndarray, v: np.ndarray):
        r"""
        Evaluates the second derivative of the surface with respect to :math:`v` at arbitrary vectors of
        :math:`u` and :math:`v`-values.

        Parameters
        ----------
        u: np.ndarray
            1-D array of :math:`u`-parameter values
        v: np.ndarray
            1-D array of :math:`v`-parameter values

        Returns
        -------
        np.ndarray
            Array of size :math:`\text{len}(u) \times \text{len}(v) \times 3`
        """
        P = self.get_control_point_array()
        return np.array(nurbs_surf_d2sdv2_uvvecs(P, self.weights, self.knots_u, self.knots_v, u, v))

    def get_edge(self, edge: SurfaceEdge, n_points: int = 10) -> np.ndarray:
        r"""
        Evaluates the surface at ``n_points`` parameter locations along a given edge.

        Parameters
        ----------
        edge: SurfaceEdge
            Edge along which to evaluate
        n_points: int
            Number of evenly-spaced parameter locations at which to evaluate the edge curve. Default: 10

        Returns
        -------
        numpy.ndarray
            2-D array of size :math:`n_\text{points} \times 3`
        """
        P = self.get_control_point_array()
        if edge == SurfaceEdge.v1:
            return np.array(nurbs_surf_eval_iso_v(P, self.weights, self.knots_u, self.knots_v, n_points, 1.0))
        elif edge == SurfaceEdge.v0:
            return np.array(nurbs_surf_eval_iso_v(P, self.weights, self.knots_u, self.knots_v, n_points, 0.0))
        elif edge == SurfaceEdge.u1:
            return np.array(nurbs_surf_eval_iso_u(P, self.weights, self.knots_u, self.knots_v, 1.0, n_points))
        elif edge == SurfaceEdge.u0:
            return np.array(nurbs_surf_eval_iso_u(P, self.weights, self.knots_u, self.knots_v, 0.0, n_points))
        else:
            raise ValueError(f"No edge called {edge}")

    def get_first_derivs_along_edge(self, edge: SurfaceEdge, n_points: int = 10, perp: bool = True) -> np.ndarray:
        r"""
        Evaluates the parallel or perpendicular derivative along a surface edge at ``n_points`` parameter locations.
        The derivative represents either :math:`\frac{\partial \mathbf{S}(u,v)}{\partial u}` or
        :math:`\frac{\partial \mathbf{S}(u,v)}{\partial v}` depending on which edge is selected and which value is
        assigned to ``perp``.

        Parameters
        ----------
        edge: SurfaceEdge
            Edge along which to evaluate
        n_points: int
            Number of evenly-spaced parameter locations at which to evaluate the derivative. Default: 10
        perp: bool
            Whether to evaluate the cross-derivative. If ``False``, the derivative along the parameter direction
            parallel to the edge will be evaluated instead. Default: ``True``

        Returns
        -------
        numpy.ndarray
            2-D array of size :math:`n_\text{points} \times 3`
        """
        P = self.get_control_point_array()
        if edge == SurfaceEdge.v1:
            return np.array(nurbs_surf_dsdv_iso_v(
                P, self.weights, self.knots_u, self.knots_v, n_points, 1.0)) if perp else np.array(
                nurbs_surf_dsdu_iso_v(P, self.weights, self.knots_u, self.knots_v, n_points, 1.0))
        elif edge == SurfaceEdge.v0:
            return np.array(nurbs_surf_dsdv_iso_v(
                P, self.weights, self.knots_u, self.knots_v, n_points, 0.0)) if perp else np.array(
                nurbs_surf_dsdu_iso_v(P, self.weights, self.knots_u, self.knots_v, n_points, 0.0))
        elif edge == SurfaceEdge.u1:
            return np.array(nurbs_surf_dsdu_iso_u(
                P, self.weights, self.knots_u, self.knots_v, 1.0, n_points)) if perp else np.array(
                nurbs_surf_dsdv_iso_u(P, self.weights, self.knots_u, self.knots_v, 1.0, n_points))
        elif edge == SurfaceEdge.u0:
            return np.array(nurbs_surf_dsdu_iso_u(
                P, self.weights, self.knots_u, self.knots_v, 0.0, n_points)) if perp else np.array(
                nurbs_surf_dsdv_iso_u(P, self.weights, self.knots_u, self.knots_v, 0.0, n_points))
        else:
            raise ValueError(f"No edge called {edge}")

    def get_second_derivs_along_edge(self, edge: SurfaceEdge, n_points: int = 10, perp: bool = True) -> np.ndarray:
        r"""
        Evaluates the parallel or perpendicular second derivative along a surface edge at ``n_points`` parameter
        locations. The derivative represents either :math:`\frac{\partial^2 \mathbf{S}(u,v)}{\partial u^2}` or
        :math:`\frac{\partial^2 \mathbf{S}(u,v)}{\partial v^2}` depending on which edge is selected and which value is
        assigned to ``perp``.

        Parameters
        ----------
        edge: SurfaceEdge
            Edge along which to evaluate
        n_points: int
            Number of evenly-spaced parameter locations at which to evaluate the second derivative. Default: 10
        perp: bool
            Whether to evaluate the cross-derivative. If ``False``, the second derivative along the parameter direction
            parallel to the edge will be evaluated instead. Default: ``True``

        Returns
        -------
        numpy.ndarray
            2-D array of size :math:`n_\text{points} \times 3`
        """
        P = self.get_control_point_array()
        if edge == SurfaceEdge.v1:
            return np.array(nurbs_surf_d2sdv2_iso_v(
                P, self.weights, self.knots_u, self.knots_v, n_points, 1.0)) if perp else np.array(
                nurbs_surf_d2sdu2_iso_v(P, self.weights, self.knots_u, self.knots_v, n_points, 1.0))
        elif edge == SurfaceEdge.v0:
            return np.array(nurbs_surf_d2sdv2_iso_v(
                P, self.weights, self.knots_u, self.knots_v, n_points, 0.0)) if perp else np.array(
                nurbs_surf_d2sdu2_iso_v(P, self.weights, self.knots_u, self.knots_v, n_points, 0.0))
        elif edge == SurfaceEdge.u1:
            return np.array(nurbs_surf_d2sdu2_iso_u(
                P, self.weights, self.knots_u, self.knots_v, 1.0, n_points)) if perp else np.array(
                nurbs_surf_d2sdv2_iso_u(P, self.weights, self.knots_u, self.knots_v, 1.0, n_points))
        elif edge == SurfaceEdge.u0:
            return np.array(nurbs_surf_d2sdu2_iso_u(
                P, self.weights, self.knots_u, self.knots_v, 0.0, n_points)) if perp else np.array(
                nurbs_surf_d2sdv2_iso_u(P, self.weights, self.knots_u, self.knots_v, 0.0, n_points))
        else:
            raise ValueError(f"No edge called {edge}")

    def verify_g0(self, other: 'NURBSSurface', surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge,
                  n_points: int = 10):
        """ Verifies that two NURBS Surfaces are G0 continuous along their shared edge"""
        self_edge = self.get_edge(surface_edge, n_points=n_points)
        other_edge = other.get_edge(other_surface_edge, n_points=n_points)
        assert np.array_equal(self_edge, other_edge)

    def verify_g1(self, other: "NURBSSurface", surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge,
                  n_points: int = 10):
        """
        Verifies that two NURBSSurfaces are G1 continuous along their shared edge
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

    def verify_g2(self, other: "NURBSSurface", surface_edge: SurfaceEdge, other_surface_edge: SurfaceEdge,
                  n_points: int = 10):
        """
        Verifies that two NURBSSurfaces are G2 continuous along their shared edge
        """
        # Get the first derivatives at the boundary and perpendicular to the boundary for each surface,
        # evaluated at "n_points" locations along the boundary
        self_perp_edge_derivs = self.get_second_derivs_along_edge(surface_edge, n_points=n_points, perp=True)
        other_perp_edge_derivs = other.get_second_derivs_along_edge(other_surface_edge, n_points=n_points, perp=True)
        print(f"{self_perp_edge_derivs=},{other_perp_edge_derivs=}")
        self_perp_edge_derivs[np.absolute(self_perp_edge_derivs) < 1e-6] = 0.0
        other_perp_edge_derivs[np.absolute(other_perp_edge_derivs) < 1e-6] = 0.0

        ratios_other_self = other_perp_edge_derivs / self_perp_edge_derivs
        #print(f"{ratios_other_self=}")
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
                               x: Length = None, y: Length = None, z: Length = None):
        """
        Computes one parametric value given the other and a specified :math:`x`-, :math:`y`-, or :math:`z`-location.
        As an example, given a :obj:`~aerocaps.geom.surfaces.NURBSSurface` object
        assigned to the variable ``surf``,
        the :math:`u`-parameter corresponding to :math:`y=1.4` along the :math:`v=0.8` isoparametric curve can be
        computed using

        .. code-block:: python

            u = surf.get_u_or_v_given_uvxyz(v=0.8, y=1.4)

        Note that the inputs are keyword arguments to avoid having to specify ``None`` for each of the arguments
        not used.

        Parameters
        ----------
        u: float or None
            Value of :math:`u` to solve for or specify. If left as ``None``, this parameter will be solved for.
            If ``None``, :math:`v` must be specified. Default: ``None``
        v: float or None
            Value of :math:`v` to solve for or specify. If left as ``None``, this parameter will be solved for.
            If ``None``, :math:`u` must be specified. Default: ``None``
        uv_guess: float
            Starting guess for the unsolved :math:`u` or :math:`v` parameter. Default: ``0.5``
        x: Length or None
            :math:`x`-location corresponding to the :math:`u` or :math:`v` parameter to be solved. If this value is
            outside the surface geometry, the root-finder will fail and an error will be raised. If unspecified,
            either :math:`y` or :math:`z` must be specified. Default: ``None``
        y: Length or None
            :math:`y`-location corresponding to the :math:`u` or :math:`v` parameter to be solved. If this value is
            outside the surface geometry, the root-finder will fail and an error will be raised. If unspecified,
            either :math:`x` or :math:`z` must be specified. Default: ``None``
        z: Length or None
            :math:`z`-location corresponding to the :math:`u` or :math:`v` parameter to be solved. If this value is
            outside the surface geometry, the root-finder will fail and an error will be raised. If unspecified,
            either :math:`x` or :math:`y` must be specified. Default: ``None``

        Returns
        -------
        float
            The value of :math:`u` if :math:`v` is specified or :math:`v` if :math:`u` is specified
        """
        # Validate inputs
        if u is None and v is None or (u is not None and v is not None):
            raise ValueError("Must specify exactly one of either u or v")
        xyz_spec = (x is not None, y is not None, z is not None)
        if len([xyz for xyz in xyz_spec if xyz]) != 1:
            raise ValueError("Must specify exactly one of x, y, or z")

        if x is not None:
            xyz, xyz_val = "x", x.m
        elif y is not None:
            xyz, xyz_val = "y", y.m
        elif z is not None:
            xyz, xyz_val = "z", z.m
        else:
            raise ValueError("Did not detect an x, y, or z input")

        def root_find_func_u(u_current):
            point = self.evaluate_point3d(u_current, v)
            return np.array([getattr(point, xyz).m - xyz_val])

        def root_find_func_v(v_current):
            point = self.evaluate_point3d(u, v_current)
            return np.array([getattr(point, xyz).m - xyz_val])

        if v is not None:
            return fsolve(root_find_func_u, x0=np.array([uv_guess]))[0]
        if u is not None:
            return fsolve(root_find_func_v, x0=np.array([uv_guess]))[0]
        raise ValueError("Did not detect a u or v input")

    def split_at_u(self, u0: float) -> ("NURBSSurface", "NURBSSurface"):
        """
        Splits the NURBS surface at :math:`u=u_0` along the :math:`v`-parametric direction.
        """
        if self.has_internal_knots("u"):
            raise NotImplementedError(
                "Curve splitting perpendicular to an edge with internal knots is not yet implemented"
            )
        Pw = self.get_homogeneous_control_points()

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
            [de_casteljau(i=0, j=i, k=k) for i in range(self.n_points_u)] for k in range(self.n_points_v)
        ])
        bez_surf_split_2_Pw = np.array([
            [de_casteljau(i=i, j=self.degree_u - i, k=k) for i in range(self.n_points_u)] for k in
            range(self.n_points_v)
        ])

        transposed_Pw_1 = np.transpose(bez_surf_split_1_Pw, (1, 0, 2))
        transposed_Pw_2 = np.transpose(bez_surf_split_2_Pw, (1, 0, 2))

        P1, w1 = self.project_homogeneous_control_points(transposed_Pw_1)
        P2, w2 = self.project_homogeneous_control_points(transposed_Pw_2)

        return (
            NURBSSurface(P1, self.knots_u, self.knots_v, w1),
            NURBSSurface(P2, self.knots_u, self.knots_v, w2)
        )

    def split_at_v(self, v0: float) -> ("NURBSSurface", "NURBSSurface"):
        """
        Splits the NURBS surface at :math:`v=v_0` along the :math:`u`-parametric direction.
        """
        if self.has_internal_knots("v"):
            raise NotImplementedError(
                "Curve splitting perpendicular to an edge with internal knots is not yet implemented"
            )
        Pw = self.get_homogeneous_control_points()

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
            [de_casteljau(i=0, j=i, k=k) for i in range(self.n_points_v)] for k in range(self.n_points_u)
        ])
        bez_surf_split_2_Pw = np.array([
            [de_casteljau(i=i, j=self.degree_v - i, k=k) for i in range(self.n_points_v)] for k in
            range(self.n_points_u)
        ])

        P1, w1 = self.project_homogeneous_control_points(bez_surf_split_1_Pw)
        P2, w2 = self.project_homogeneous_control_points(bez_surf_split_2_Pw)

        return (
            NURBSSurface(P1, self.knots_u, self.knots_v, w1),
            NURBSSurface(P2, self.knots_u, self.knots_v, w2)
        )

    def transform(self, **transformation_kwargs) -> "NURBSSurface":
        """
        Creates a transformed copy of the surface by transforming each of the control points

        Parameters
        ----------
        transformation_kwargs
            Keyword arguments passed to :obj:`~aerocaps.geom.transformation.Transformation3D`

        Returns
        -------
        BSplineSurface
            Transformed surface
        """
        transformation = Transformation3D(**transformation_kwargs)
        initial_control_points = self.get_control_point_array()
        return NURBSSurface(
            np.array([transformation.transform(p_arr) for p_arr in initial_control_points]),
            weights=deepcopy(self.weights),
            knots_u=deepcopy(self.knots_u),
            knots_v=deepcopy(self.knots_v),
            name=self.name, construction=self.construction
        )

    def generate_control_point_net(self) -> (typing.List[Point3D], typing.List[Line3D]):
        """
        Generates a list of :obj:`~aerocaps.geom.point.Point3D` and :obj:`~aerocaps.geom.curves.Line3D` objects
        representing the NURBS surface's control points and connections between them

        Returns
        -------
        typing.List[Point3D], typing.List[Line3D]
            Control points and lines between adjacent control points in flattened lists
        """
        control_points = self.get_control_point_array()
        points = []
        lines = []

        for i in range(self.n_points_u):
            for j in range(self.n_points_v):
                points.append(Point3D.from_array(control_points[i, j, :]))

        for i in range(self.n_points_u - 1):
            for j in range(self.n_points_v - 1):
                point_obj_1 = Point3D.from_array(control_points[i, j, :])
                point_obj_2 = Point3D.from_array(control_points[i + 1, j, :])
                point_obj_3 = Point3D.from_array(control_points[i, j + 1, :])

                line_1 = Line3D(p0=point_obj_1, p1=point_obj_2)
                line_2 = Line3D(p0=point_obj_1, p1=point_obj_3)
                lines.extend([line_1, line_2])

                if i < self.n_points_u - 2 and j < self.n_points_v - 2:
                    continue

                point_obj_4 = Point3D.from_array(control_points[i + 1, j + 1, :])
                line_3 = Line3D(p0=point_obj_3, p1=point_obj_4)
                line_4 = Line3D(p0=point_obj_2, p1=point_obj_4)
                lines.extend([line_3, line_4])

        return points, lines

    def plot_surface(self, plot: pv.Plotter, Nu: int = 50, Nv: int = 50, **mesh_kwargs):
        """
        Plots the NURBS surface using the `pyvista <https://pyvista.org/>`_ library

        Parameters
        ----------
        plot:
            :obj:`pyvista.Plotter` instance
        Nu: int
            Number of points to evaluate in the :math:`u`-parametric direction. Default: ``50``
        Nv: int
            Number of points to evaluate in the :math:`v`-parametric direction. Default: ``50``
        mesh_kwargs:
            Keyword arguments to pass to :obj:`pyvista.Plotter.add_mesh`

        Returns
        -------
        pyvista.core.pointset.StructuredGrid
            The evaluated NURBS surface
        """
        XYZ = self.evaluate_grid(Nu, Nv)
        grid = pv.StructuredGrid(XYZ[:, :, 0], XYZ[:, :, 1], XYZ[:, :, 2])
        plot.add_mesh(grid, **mesh_kwargs)
        return grid

    def plot_control_point_mesh_lines(self, plot: pv.Plotter, **line_kwargs) -> pv.Actor:
        """
        Plots the network of lines connecting the NURBS surface control points using the
        `pyvista <https://pyvista.org/>`_ library

        Parameters
        ----------
        plot:
            :obj:`pyvista.Plotter` instance
        line_kwargs:
            Keyword arguments to pass to the :obj:`pyvista.Plotter.add_lines`

        Returns
        -------
        pv.Actor
            The lines actor
        """
        _, line_objs = self.generate_control_point_net()
        line_arr = np.array([[line_obj.p0.as_array(), line_obj.p1.as_array()] for line_obj in line_objs])
        line_arr = line_arr.reshape((len(line_objs) * 2, 3))
        line_actor = plot.add_lines(line_arr, **line_kwargs)
        return line_actor

    def plot_control_points(self, plot: pv.Plotter, **point_kwargs) -> pv.Actor:
        """
        Plots the NURBS surface control points using the `pyvista <https://pyvista.org/>`_ library

        Parameters
        ----------
        plot:
            :obj:`pyvista.Plotter` instance
        point_kwargs:
            Keyword arguments to pass to the :obj:`pyvista.Plotter.add_points`

        Returns
        -------
        pv.Actor
            The points actor
        """
        point_objs, _ = self.generate_control_point_net()
        point_arr = np.array([point_obj.as_array() for point_obj in point_objs])
        point_actor = plot.add_points(point_arr, **point_kwargs)
        return point_actor

    def __repr__(self):
        return (f"{self.name}: {self.degree_u} x {self.degree_v} {self.__class__.__name__} "
                f"({self.n_points_u} x {self.n_points_v} control points)")


class TrimmedSurface(Surface):
    def __init__(self,
                 untrimmed_surface: Surface,
                 outer_boundary: CompositeCurve3D,
                 outer_boundary_para: CompositeCurve3D,
                 outer_curve_on_parametric_surf_para: CurveOnParametricSurface,
                 inner_boundaries: typing.List[CompositeCurve3D] = None,
                 inner_boundaries_para: typing.List[CurveOnParametricSurface] = None,
                 name: str = "TrimmedSurface",
                 construction: bool = False):
        """

        Parameters
        ----------
        untrimmed_surface: Surface
        outer_boundary: CompositeCurve3D
        outer_boundary_para: CompositeCurve3D
        outer_curve_on_parametric_surf_para: CurveOnParametricSurface
        inner_boundaries: typing.List[CompositeCurve3D] or None
        inner_boundaries_para: typing.List[CurveOnParametricSurface] or None
        name: str
            Name of the geometric object. May be re-assigned a unique name when added to a
            :obj:`~aerocaps.geom.geometry_container.GeometryContainer`
        construction: bool
            Whether this is a geometry used only for construction of other geometries. If ``True``, this
            geometry will not be exported or plotted. Default: ``False``
        """
        self.untrimmed_surface = untrimmed_surface
        self.outer_boundary = outer_boundary
        self.outer_boundary_para = outer_boundary_para
        self.outer_curve_on_parametric_surf_para = outer_curve_on_parametric_surf_para
        self.inner_boundaries = inner_boundaries
        self.inner_boundaries_para = inner_boundaries_para
        super().__init__(name=name, construction=construction)

    @classmethod
    def from_planar_boundary_curves(cls, outer_boundary: CompositeCurve3D) -> "TrimmedSurface":
        composite_para, planar_surf = cls._get_envelope(outer_boundary)

        curve_on_parametric_surface = CurveOnParametricSurface(
            planar_surf,
            composite_para,
            outer_boundary
        )

        # Create the trimmed surface object
        return cls(planar_surf, outer_boundary, composite_para, curve_on_parametric_surface)

    @staticmethod
    def _get_envelope(outer_curves: CompositeCurve3D) -> (CompositeCurve3D, BezierSurface):
        control_point_loop = []
        for c_idx, c in enumerate(outer_curves.ordered_curves):
            if c_idx == 0:
                control_point_loop.extend(c.control_points)
            else:
                control_point_loop.extend(c.control_points[1:])
        loop_array = np.array([p.as_array() for p in control_point_loop])
        parametric_curves = []

        # Need to convert to 2-D to use shapely. Get the coordinate system of the plane containing the points
        # using cross products of vectors described by the points
        v1, v2 = None, None
        v3 = Vector3D(p0=Point3D.from_array(np.array([0.0, 0.0, 0.0])),
                      p1=Point3D.from_array(np.array([0.0, 0.0, 0.0])))
        t = 0.1
        while True:
            if not v3.is_zero_vector():
                break
            if t > 0.9:
                break
            v1 = Vector3D(
                outer_curves.ordered_curves[0].evaluate_point3d(0.0),
                outer_curves.ordered_curves[0].evaluate_point3d(t)
            )
            v2 = Vector3D(
                outer_curves.ordered_curves[0].evaluate_point3d(0.0),
                outer_curves.ordered_curves[1].evaluate_point3d(t) if len(
                    outer_curves.ordered_curves) > 1 else outer_curves.ordered_curves[0].evaluate_point3d(t + 0.1)
            )
            v3 = v1.cross(v2)
            t += 0.1
        else:
            raise ValueError("Could not compute curve loop plane from input curves")

        v4 = v1.cross(v3)

        # The coordinate system is now fully described by v1, v3, and v4. v1 and v4 are the in-plane components,
        # while v3 is the out-of-plane component. The origin of this coordinate system is at control_point_loop[0].
        loop_array_transformed = transform_points_into_coordinate_system(
            loop_array, [v1, v4, v3], [IHat3D(), JHat3D(), KHat3D()]
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
        x_min, x_max = envelope_2d[:, 0].min(), envelope_2d[:, 0].max()
        y_min, y_max = envelope_2d[:, 1].min(), envelope_2d[:, 1].max()
        dx, dy = (x_max - x_min), (y_max - y_min)
        ds = max(dx, dy)
        envelope_2d = np.array([
            [x_min, y_min],
            [x_min + ds, y_min],
            [x_min + ds, y_min + ds],
            [x_min, y_min + ds],
            [x_min, y_min]
        ])
        # envelope_2d[:,]

        # Get parametric curves in the plane defined by the envelope for each curve in the ordered curve list
        for curve in outer_curves.ordered_curves:
            cps_transformed = transform_points_into_coordinate_system(
                curve.get_control_point_array(), [v1, v4, v3], [IHat3D(), JHat3D(), KHat3D()]
            )
            cps_x = cps_transformed[:, 0]
            cps_y = cps_transformed[:, 1]
            u = [(cp_x - x_min) / ds for cp_x in cps_x]
            v = [(cp_y - y_min) / ds for cp_y in cps_y]
            # u = [cp_x - x_min for cp_x in cps_x]
            # v = [cp_y - y_min for cp_y in cps_y]
            uv = np.array([u, v]).T
            uv0 = np.column_stack((uv, np.zeros(uv.shape[0])))
            if isinstance(curve, Line3D):
                parametric_curve = curve.__class__(p0=Point3D.from_array(uv0[0, :]), p1=Point3D.from_array(uv0[1, :]))
            elif isinstance(curve, BezierCurve3D):
                parametric_curve = curve.__class__(uv0)
            elif isinstance(curve, BSplineCurve3D):
                parametric_curve = curve.__class__(uv0, curve.knot_vector, curve.degree)
            elif isinstance(curve, RationalBezierCurve3D):
                # w = np.array([np.min(curve.weights) + ds * (np.max(curve.weights) - np.min(curve.weights)) for weight in curve.weights])
                parametric_curve = curve.__class__(uv0, curve.weights * ds)  # * ds
            elif isinstance(curve, NURBSCurve3D):
                parametric_curve = curve.__class__(uv0, curve.weights, curve.knot_vector, curve.degree)
            else:
                raise ValueError(f"Invalid curve type {type(curve)}")
            parametric_curves.append(parametric_curve)

        envelope_3d = np.column_stack((envelope_2d, z_plane * np.ones(envelope_2d.shape[0])))

        # Transform the newly created envelope back into the original coordinate system
        reverse_transformed_envelope_3d = transform_points_into_coordinate_system(
            envelope_3d, [IHat3D(), JHat3D(), KHat3D()], [v1, v4, v3]
        )

        # Create a planar rectangular surface from the transformed points
        pa = Point3D.from_array(reverse_transformed_envelope_3d[0, :])
        pb = Point3D.from_array(reverse_transformed_envelope_3d[1, :])
        pc = Point3D.from_array(reverse_transformed_envelope_3d[2, :])
        pd = Point3D.from_array(reverse_transformed_envelope_3d[3, :])
        planar_surf = BezierSurface([[pa, pd], [pb, pc]])

        return CompositeCurve3D(parametric_curves), planar_surf

    def transform(self, **transformation_kwargs) -> "TrimmedSurface":
        """
        Creates a transformed copy of the surface by transforming both the outer boundary
        in geometric space and the untrimmed surface. The parameteric trimming curves
        are left unmodified.

        Parameters
        ----------
        transformation_kwargs
            Keyword arguments passed to :obj:`~aerocaps.geom.transformation.Transformation3D`

        Returns
        -------
        TrimmedSurface
            Transformed surface
        """
        return TrimmedSurface(
            untrimmed_surface=self.untrimmed_surface.transform(**transformation_kwargs),
            outer_boundary=self.outer_boundary.transform(**transformation_kwargs),
            outer_boundary_para=self.outer_boundary_para,
            outer_curve_on_parametric_surf_para=self.outer_curve_on_parametric_surf_para,
            inner_boundaries=[curve.transform(**transformation_kwargs) for curve in self.inner_boundaries] 
                if self.inner_boundaries is not None else None,
            inner_boundaries_para=self.inner_boundaries_para 
                if self.inner_boundaries_para is not None else None,
            name=self.name,
            construction=self.construction
        )

    def evaluate(self, Nt: int) -> np.ndarray:
        if self.inner_boundaries is not None:
            raise NotImplementedError("Evaluation not yet implemented for trimmed surfaces with inner loops")
        uv_locs = self.outer_boundary_para.evaluate(Nt)[:, :2]
        points, lines = concave_hull(uv_locs)
        surf_points = np.array([self.untrimmed_surface.evaluate(p[0], p[1]) for p in points])
        return surf_points, lines

    def plot_surface(self, plot: pv.Plotter, Nt: int = 100, **mesh_kwargs):
        """
        Plots the trimmed surface using the `pyvista <https://pyvista.org/>`_ library

        Parameters
        ----------
        plot:
            :obj:`pyvista.Plotter` instance
        Nt: int
            Number of points to evaluate on each boundary curve. Default: ``100``
        mesh_kwargs:
            Keyword arguments to pass to :obj:`pyvista.Plotter.add_mesh`

        Returns
        -------
        pyvista.core.pointset.StructuredGrid
            The evaluated rational Bézier surface
        """
        surf_points, lines = self.evaluate(Nt)
        faces = np.hstack(np.insert(lines, 0, 3, axis=1))
        mesh = pv.PolyData(surf_points, faces=faces)
        plot.add_mesh(mesh, **mesh_kwargs)
        return mesh

    def to_iges(self, *args, **kwargs) -> typing.List[aerocaps.iges.entity.IGESEntity]:

        # Compile the list of entities
        K1 = len(self.outer_boundary.ordered_curves)
        K2 = K1 + len(self.outer_boundary_para.ordered_curves)
        entities = [curve.to_iges() for curve in self.outer_boundary.ordered_curves]
        entities.extend([curve.to_iges() for curve in self.outer_boundary_para.ordered_curves])
        entities.append(self.outer_boundary.to_iges(entities[0:K1]))
        entities.append(self.outer_boundary_para.to_iges(entities[K1:K2]))
        entities.append(self.untrimmed_surface.to_iges())
        entities.append(self.outer_curve_on_parametric_surf_para.to_iges(entities[K2 + 2], entities[K2 + 1], entities[K2]))
        entities.append(aerocaps.iges.surfaces.TrimmedSurfaceIGES(
            entities[K2 + 2],
            entities[K2 + 3],
            outer_boundary_is_boundary_of_surface=True
        ))
        return entities
