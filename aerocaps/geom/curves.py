"""
Parametric curve classes (one-dimensional geometric objects defined by parameter :math:`t` that reside in
two- or three-dimensional space)
"""
import typing
from abc import abstractmethod

import numpy as np
import pyvista as pv
from matplotlib import pyplot as plt
from scipy.optimize import fsolve

import aerocaps.iges
import aerocaps.iges.curves
import aerocaps.iges.entity
from aerocaps.geom import Geometry2D, Geometry3D, NegativeWeightError
from aerocaps.geom.point import Point2D, Point3D
from aerocaps.geom.transformation import Transformation2D, Transformation3D
from aerocaps.geom.vector import Vector3D, Vector2D
from aerocaps.units.angle import Angle
from aerocaps.units.length import Length
from aerocaps.utils.math import bernstein_poly

__all__ = [
    "PCurveData2D",
    "PCurveData3D",
    "PCurve2D",
    "PCurve3D",
    "Line2D",
    "Line3D",
    "CircularArc2D",
    "Bezier2D",
    "Bezier3D",
    "BSpline3D",
    "RationalBezierCurve3D",
    "NURBSCurve3D",
    "CompositeCurve3D",
    "CurveOnParametricSurface"
]

_projection_dict = {
    "X": 0,
    "Y": 1,
    "Z": 2,
}


class PCurveData2D:
    def __init__(self, t: np.ndarray, xy: np.ndarray, xpyp: np.ndarray, xppypp: np.ndarray, k: np.ndarray,
                 R: np.ndarray):
        self.t = t
        self.xy = xy
        self.xpyp = xpyp
        self.xppypp = xppypp
        self.k = k
        self.R = R
        self.R_abs_min = np.min(np.abs(self.R))

    def plot(self, ax: plt.Axes, **kwargs):
        ax.plot(self.xy[:, 0], self.xy[:, 1], **kwargs)

    def get_curvature_comb(self, max_k_normalized_scale_factor, interval: int = 1):
        first_deriv_mag = np.hypot(self.xpyp[:, 0], self.xpyp[:, 1])
        comb_heads_x = self.xy[:, 0] - self.xpyp[:, 1] / first_deriv_mag * self.k * max_k_normalized_scale_factor
        comb_heads_y = self.xy[:, 1] + self.xpyp[:, 0] / first_deriv_mag * self.k * max_k_normalized_scale_factor
        # Stack the x and y columns (except for the last x and y values) horizontally and keep only the rows by the
        # specified interval:
        comb_tails = np.column_stack((self.xy[:, 0], self.xy[:, 1]))[:-1:interval, :]
        comb_heads = np.column_stack((comb_heads_x, comb_heads_y))[:-1:interval, :]
        # Add the last x and y values onto the end (to make sure they do not get skipped with input interval)
        comb_tails = np.vstack((comb_tails, np.array([self.xy[-1, 0], self.xy[-1, 1]])))
        comb_heads = np.vstack((comb_heads, np.array([comb_heads_x[-1], comb_heads_y[-1]])))
        return comb_tails, comb_heads

    def approximate_arc_length(self):
        return np.sum(np.hypot(self.xy[1:, 0] - self.xy[:-1, 0], self.xy[1:, 1] - self.xy[:-1, 1]))


class PCurve2D(Geometry2D):
    @abstractmethod
    def evaluate_point2d(self, t: float) -> Point2D:
        pass

    @abstractmethod
    def evaluate_single_t(self, t: float) -> np.ndarray:
        pass

    @abstractmethod
    def evaluate(self, t: np.ndarray = None) -> PCurveData2D:
        pass


class PCurveData3D:
    def __init__(self, t: np.ndarray, xyz: np.ndarray, xpypzp: np.ndarray, xppyppzpp: np.ndarray, k: np.ndarray,
                 R: np.ndarray):
        self.t = t
        self.xyz = xyz
        self.xpypzp = xpypzp
        self.xppyppzpp = xppyppzpp
        self.k = k
        self.R = R
        self.R_abs_min = np.min(np.abs(self.R))

    def plot(self, ax: plt.Axes, **kwargs):
        ax.plot3D(self.xyz[:, 0], self.xyz[:, 1], self.xyz[:, 2], **kwargs)

    def get_curvature_comb(self, max_k_normalized_scale_factor, interval: int = 1):
        first_deriv_mag = np.sqrt(self.xpypzp[:, 0] ** 2, self.xpypzp[:, 1] ** 2, self.xpypzp[:, 2] ** 2)
        comb_heads_x = self.xyz[:, 0] - self.xpypzp[:, 1] / first_deriv_mag * self.k * max_k_normalized_scale_factor
        comb_heads_y = self.xyz[:, 1] + self.xpypzp[:, 0] / first_deriv_mag * self.k * max_k_normalized_scale_factor
        comb_heads_z = self.xyz[:, 2] + self.xpypzp[:, 0] / first_deriv_mag * self.k * max_k_normalized_scale_factor
        # Stack the x and y columns (except for the last x and y values) horizontally and keep only the rows by the
        # specified interval:
        comb_tails = np.column_stack((self.xyz[:, 0], self.xyz[:, 1], self.xyz[:, 2]))[:-1:interval, :]
        comb_heads = np.column_stack((comb_heads_x, comb_heads_y))[:-1:interval, :]
        # Add the last x and y values onto the end (to make sure they do not get skipped with input interval)
        comb_tails = np.vstack((comb_tails, np.array([self.xyz[-1, 0], self.xyz[-1, 1], self.xyz[-1, 2]])))
        comb_heads = np.vstack((comb_heads, np.array([comb_heads_x[-1], comb_heads_y[-1], comb_heads_z[-1]])))
        return comb_tails, comb_heads

    def approximate_arc_length(self):
        return np.sum(np.sqrt(
            (self.xyz[1:, 0] - self.xyz[:-1, 0]) ** 2,
            (self.xyz[1:, 1] - self.xyz[:-1, 1]) ** 2,
            (self.xyz[1:, 2] - self.xyz[:-1, 2]) ** 2
        ))


class PCurve3D(Geometry3D):
    @abstractmethod
    def evaluate_point3d(self, t: float) -> Point3D:
        pass

    @abstractmethod
    def evaluate_single_t(self, t: float) -> np.ndarray:
        pass

    @abstractmethod
    def evaluate(self, t: np.ndarray = None) -> PCurveData3D:
        pass


class Line2D(PCurve2D):
    """
    Two-dimensional line class
    """
    def __init__(self,
                 p0: Point2D,
                 p1: Point2D = None,
                 theta: Angle = None,
                 d: Length = Length(m=1.0)
                 ):
        r"""
        Two-dimensional line defined by either two points or a point and an angle. If a second point (``p1``)
        is defined, the curve will be evaluated as

        .. math::

            \begin{align}
            x(t) &= x_0 + t (x_1 - x_0) \\
            y(t) &= y_0 + t (y_1 - y_0)
            \end{align}

        If an angle (``theta``) is specified instead, the curve will be evaluated as

        .. math::

            \begin{align}
            x(t) &= x_0 + d \cdot t \cdot \cos{\theta} \\
            y(t) &= y_0 + d \cdot t \cdot \sin{\theta}
            \end{align}

        Parameters
        ----------
        p0: Point2D
            Origin of the line
        p1: Point2D or None
            Endpoint of the line. If ``None``, ``theta`` must be specified. Default: ``None``
        theta: Angle or None
            Angle of the line (counter-clockwise positive, :math:`0^{\circ}` defined along the :math:`x`-axis).
            If ``None``, ``p1`` must be specified. Default: ``None``
        d: Length
            Used in conjunction with ``theta`` to determine the point corresponding to :math:`t=1`. If ``p1`` is
            specified, this value is not used. Default: ``Length(m=1.0)``
        """
        if theta and p1:
            raise ValueError("Angle theta should not be specified if p1 is specified")
        if not theta and not p1:
            raise ValueError("Must specify either angle theta or p1")
        self.p0 = p0
        self.theta = theta
        from aerocaps.geom.tools import measure_distance_between_points  # Avoid circular import
        self.d = d if not p1 else Length(m=measure_distance_between_points(p0, p1))
        self.p1 = self.evaluate_point2d(1.0) if not p1 else p1
        self.control_points = [self.p0, self.p1]

    def evaluate_point2d(self, t: float) -> Point2D:
        if self.theta:
            return Point2D(
                x=self.p0.x + self.d * np.cos(self.theta.rad) * t,
                y=self.p0.y + self.d * np.sin(self.theta.rad) * t
            )
        else:
            return self.p0 + t * (self.p1 - self.p0)

    def evaluate_single_t(self, t: float) -> np.ndarray:
        return self.evaluate_point2d(t).as_array()

    def evaluate(self, t: np.ndarray = None) -> PCurveData2D:
        t = np.linspace(0.0, 1.0, 10) if t is None else t

        if self.theta:
            x = self.p0.x.m + self.d.m * np.cos(self.theta.rad) * t
            y = self.p0.y.m + self.d.m * np.sin(self.theta.rad) * t
            xp = self.d.m * np.cos(self.theta.rad) * np.ones(t.shape)
            yp = self.d.m * np.sin(self.theta.rad) * np.ones(t.shape)
        else:
            x = self.p0.x.m + t * (self.p1.x.m - self.p0.x.m)
            y = self.p0.y.m + t * (self.p1.y.m - self.p0.y.m)
            xp = (self.p1.x.m - self.p0.x.m) * np.ones(t.shape)
            yp = (self.p1.y.m - self.p0.y.m) * np.ones(t.shape)

        xy = np.column_stack((x, y))
        xpyp = np.column_stack((xp, yp))
        xppypp = np.zeros((len(t), 2))
        R = np.inf * np.ones(t.shape)
        k = np.zeros(t.shape)
        return PCurveData2D(t=t, xy=xy, xpyp=xpyp, xppypp=xppypp, k=k, R=R)

    def get_vector(self) -> Vector2D:
        return Vector2D(p0=self.p0, p1=self.p1)

    def plot(self, plot: pv.Plotter, **line_kwargs):
        line_arr = np.array([self.p0.as_array(), self.p1.as_array()])
        plot.add_lines(line_arr, **line_kwargs)


class Line3D(PCurve3D):
    """
    Three-dimensional line class
    """
    def __init__(self,
                 p0: Point3D,
                 p1: Point3D = None,
                 theta: Angle = None,
                 phi: Angle = None,
                 d: Length = Length(m=1.0)
                 ):
        if (theta and p1) or (phi and p1):
            raise ValueError("Angles should not be specified if p1 is specified")
        if (not theta and not p1) or (not phi and not p1):
            raise ValueError("Must specify either both angles, theta and phi, or p1")
        self.p0 = p0
        self.theta = theta
        self.phi = phi
        from aerocaps.geom.tools import measure_distance_between_points  # Avoid circular import
        self.d = d if not p1 else Length(m=measure_distance_between_points(p0, p1))
        self.p1 = self.evaluate_point3d(1.0) if not p1 else p1
        self.control_points = [self.p0, self.p1]

    def to_iges(self, *args, **kwargs) -> aerocaps.iges.entity.IGESEntity:
        return aerocaps.iges.curves.LineIGES(self.p0.as_array(), self.p1.as_array())

    def from_iges(self):
        pass

    def get_control_point_array(self, unit: str = "m") -> np.ndarray:
        return np.array([p.as_array(unit=unit) for p in self.control_points])

    def reverse(self) -> "Line3D":
        return self.__class__(p0=self.p1, p1=self.p0)

    def projection_on_principal_plane(self, plane: str = "XY") -> Line2D:
        return Line2D(p0=self.p0.projection_on_principal_plane(plane), p1=self.p1.projection_on_principal_plane(plane))

    def evaluate_point3d(self, t: float) -> Point3D:
        if self.phi and self.theta:
            return Point3D(
                x=self.p0.x + self.d * np.cos(self.phi.rad) * np.cos(self.theta.rad) * t,
                y=self.p0.y + self.d * np.cos(self.phi.rad) * np.sin(self.theta.rad) * t,
                z=self.p0.z + self.d * np.sin(self.phi.rad) * t
            )
        else:
            return self.p0 + t * (self.p1 - self.p0)

    def evaluate_single_t(self, t: float) -> np.ndarray:
        return self.evaluate_point3d(t).as_array()

    def evaluate(self, t: np.ndarray = None) -> PCurveData3D:
        t = np.linspace(0.0, 1.0, 10) if t is None else t

        if self.theta:
            x = self.p0.x + self.d * np.cos(self.phi.rad) * np.cos(self.theta.rad) * t,
            y = self.p0.y + self.d * np.cos(self.phi.rad) * np.sin(self.theta.rad) * t,
            z = self.p0.z + self.d * np.sin(self.phi.rad) * t
            xp = self.d.m * np.cos(self.phi.rad) * np.cos(self.theta.rad) * np.ones(t.shape)
            yp = self.d.m * np.cos(self.phi.rad) * np.sin(self.theta.rad) * np.ones(t.shape)
            zp = self.d.m * np.sin(self.phi.rad) * np.ones(t.shape)
        else:
            x = self.p0.x.m + t * (self.p1.x.m - self.p0.x.m)
            y = self.p0.y.m + t * (self.p1.y.m - self.p0.y.m)
            z = self.p0.z.m + t * (self.p1.z.m - self.p0.z.m)
            xp = (self.p1.x.m - self.p0.x.m) * np.ones(t.shape)
            yp = (self.p1.y.m - self.p0.y.m) * np.ones(t.shape)
            zp = (self.p1.z.m - self.p0.z.m) * np.ones(t.shape)

        xyz = np.column_stack((x, y, z))
        xpypzp = np.column_stack((xp, yp, zp))
        xppyppzpp = np.zeros((len(t), 3))
        R = np.inf * np.ones(t.shape)
        k = np.zeros(t.shape)
        return PCurveData3D(t=t, xyz=xyz, xpypzp=xpypzp, xppyppzpp=xppyppzpp, k=k, R=R)

    def get_vector(self) -> Vector3D:
        return Vector3D(p0=self.p0, p1=self.p1)

    def plot(self, plot: pv.Plotter, **line_kwargs):
        line_arr = np.array([self.p0.as_array(), self.p1.as_array()])
        plot.add_lines(line_arr, **line_kwargs)


class CircularArc2D(PCurve2D):
    def __init__(self, center: Point2D, radius: Length, start_point: Point2D = None, end_point: Point2D = None,
                 start_angle: Angle = None, end_angle: Angle = None, complement: bool = False):
        self.center = center
        self.radius = radius
        self.start_point = start_point
        self.end_point = end_point
        if start_angle is None:
            self.start_angle = Angle(rad=np.arctan2(start_point.y.m - center.y.m, start_point.x.m - center.x.m))
        else:
            self.start_angle = start_angle
        if end_angle is None:
            self.end_angle = Angle(rad=np.arctan2(end_point.y.m - center.y.m, end_point.x.m - center.x.m))
        else:
            self.end_angle = end_angle
        self.complement = complement

    def _map_t_to_angle(self, t):
        if self.complement:
            return self.start_angle.rad - (2 * np.pi - (self.end_angle.rad - self.start_angle.rad)) * t
        else:
            return (self.end_angle.rad - self.start_angle.rad) * t + self.start_angle.rad

    def evaluate_point2d(self, t: float) -> Point2D:
        return Point2D(
            x=self.center.x + self.radius * np.cos(self._map_t_to_angle(t)),
            y=self.center.y + self.radius * np.sin(self._map_t_to_angle(t)),
        )

    def evaluate_single_t(self, t: float) -> np.ndarray:
        return self.evaluate_point2d(t).as_array()

    def evaluate(self, t: np.ndarray = None) -> PCurveData2D:
        t = np.linspace(0.0, 1.0, 100) if t is None else t
        angle_range = self.end_angle.rad - self.start_angle.rad
        x = self.center.x.m + self.radius.m * np.cos(self._map_t_to_angle(t))
        y = self.center.y.m + self.radius.m * np.sin(self._map_t_to_angle(t))
        xp = -self.radius.m * angle_range * np.sin(self._map_t_to_angle(t))
        yp = self.radius.m * angle_range * np.cos(self._map_t_to_angle(t))
        xpp = -self.radius.m * angle_range ** 2 * np.cos(self._map_t_to_angle(t))
        ypp = -self.radius.m * angle_range ** 2 * np.sin(self._map_t_to_angle(t))
        xy = np.column_stack((x, y))
        xpyp = np.column_stack((xp, yp))
        xppypp = np.column_stack((xpp, ypp))
        R = self.radius.m * np.ones(t.shape)
        k = 1 / self.radius.m * np.ones(t.shape)
        return PCurveData2D(t=t, xy=xy, xpyp=xpyp, xppypp=xppypp, k=k, R=R)


class Bezier2D(PCurve2D):

    def __init__(self, control_points: typing.List[Point2D]):
        self.control_points = control_points
        self.curve_connections = []

    @property
    def degree(self):
        return len(self.control_points) - 1

    @degree.setter
    def degree(self, value):
        raise AttributeError("The 'degree' property is read-only. Use the Bezier2D.elevate_degree method to increase"
                             "the degree of the curve while retaining the shape, or manually add or remove control "
                             "points to change the degree directly.")

    @staticmethod
    def finite_diff_P(P: np.ndarray, k: int, i: int):
        """Calculates the finite difference of the control points as shown in
        https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/Bezier/bezier-der.html

        Arguments
        =========
        P: np.ndarray
            Array of control points for the Bézier curve
        k: int
            Finite difference level (e.g., k = 1 is the first derivative finite difference)
        i: int
            An index referencing a location in the control point array
        """

        def finite_diff_recursive(_k, _i):
            if _k > 1:
                return finite_diff_recursive(_k - 1, _i + 1) - finite_diff_recursive(_k - 1, _i)
            else:
                return P[_i + 1, :] - P[_i, :]

        return finite_diff_recursive(k, i)

    def derivative(self, P: np.ndarray, t: np.ndarray, degree: int, order: int):
        r"""
        Calculates an arbitrary-order derivative of the Bézier curve

        Parameters
        ==========
        P: np.ndarray
            The control point array

        t: np.ndarray
            The parameter vector

        degree: int
            The degree of the Bézier curve

        order: int
            The derivative order. For example, ``order=2`` returns the second derivative.

        Returns
        =======
        np.ndarray
            An array of ``shape=(N,2)`` where ``N`` is the number of evaluated points specified by the :math:`t` vector.
            The columns represent :math:`C^{(m)}_x(t)` and :math:`C^{(m)}_y(t)`, where :math:`m` is the
            derivative order.
        """
        return np.sum(np.array([np.prod(np.array([degree - idx for idx in range(order)])) *
                                np.array([self.finite_diff_P(P, order, i)]).T *
                                np.array([bernstein_poly(degree - order, i, t)])
                                for i in range(degree + 1 - order)]), axis=0).T

    def get_control_point_array(self, unit: str = "m") -> np.ndarray:
        return np.array([p.as_array(unit=unit) for p in self.control_points])

    @classmethod
    def generate_from_array(cls, P: np.ndarray, unit: str = "m"):
        return cls([Point2D(x=Length(**{unit: xy[0]}), y=Length(**{unit: xy[1]})) for xy in P])

    def evaluate_point2d(self, t: float) -> Point2D:
        n_ctrl_points = len(self.control_points)
        degree = n_ctrl_points - 1
        P = self.get_control_point_array()

        # Evaluate the curve
        x, y = 0.0, 0.0
        for i in range(n_ctrl_points):
            # Calculate the x- and y-coordinates of the Bézier curve given the input vector t
            x += P[i, 0] * bernstein_poly(degree, i, t)
            y += P[i, 1] * bernstein_poly(degree, i, t)

        return Point2D(x=Length(m=x), y=Length(m=y))

    def evaluate_single_t(self, t: float) -> np.ndarray:
        return self.evaluate_point2d(t).as_array()

    def evaluate(self, t: np.array or None = None) -> PCurveData2D:
        r"""
        Evaluates the curve using an optionally specified parameter vector.

        Parameters
        ==========
        t: np.ndarray or ``None``
            Optional direct specification of the parameter vector for the curve. Not specifying this value
            gives a linearly spaced parameter vector from ``t_start`` or ``t_end`` with the default size.
            Default: ``None``

        Returns
        =======
        PCurveData
            Data class specifying the following information about the Bézier curve:

            .. math::

                    C_x(t), C_y(t), C'_x(t), C'_y(t), C''_x(t), C''_y(t), \kappa(t)

            where the :math:`x` and :math:`y` subscripts represent the :math:`x` and :math:`y` components of the
            vector-valued functions :math:`\vec{C}(t)`, :math:`\vec{C}'(t)`, and :math:`\vec{C}''(t)`.
        """
        # Pass the starting and ending parameter vector values to the parameter vector generator if they were
        # specified directly

        # Generate the parameter vector
        t = np.linspace(0.0, 1.0, 100) if t is None else t

        # Number of control points, curve degree, control point array
        n_ctrl_points = len(self.control_points)
        degree = n_ctrl_points - 1
        P = self.get_control_point_array()

        # Evaluate the curve
        x, y = np.zeros(t.shape), np.zeros(t.shape)
        for i in range(n_ctrl_points):
            # Calculate the x- and y-coordinates of the Bézier curve given the input vector t
            x += P[i, 0] * bernstein_poly(degree, i, t)
            y += P[i, 1] * bernstein_poly(degree, i, t)
        xy = np.column_stack((x, y))

        # Calculate the first derivative
        first_deriv = self.derivative(P=P, t=t, degree=degree, order=1)
        xp = first_deriv[:, 0]
        yp = first_deriv[:, 1]

        # Calculate the second derivative
        second_deriv = self.derivative(P=P, t=t, degree=degree, order=2)
        xpp = second_deriv[:, 0]
        ypp = second_deriv[:, 1]

        # Combine the derivative x and y data
        xpyp = np.column_stack((xp, yp))
        xppypp = np.column_stack((xpp, ypp))

        # Calculate the curvature
        with np.errstate(divide='ignore', invalid='ignore'):
            # Calculate the curvature of the Bézier curve (k = kappa = 1 / R, where R is the radius of curvature)
            k = np.true_divide((xp * ypp - yp * xpp), (xp ** 2 + yp ** 2) ** (3 / 2))

        # Calculate the radius of curvature: R = 1 / kappa
        with np.errstate(divide='ignore', invalid='ignore'):
            R = np.true_divide(1, k)

        return PCurveData2D(t=t, xy=xy, xpyp=xpyp, xppypp=xppypp, k=k, R=R)

    def compute_t_corresponding_to_x(self, x_seek: float, t0: float = 0.5):
        def bez_root_find_func(t):
            point = self.evaluate_point2d(t[0])
            return np.array([point.x.m - x_seek])

        return fsolve(bez_root_find_func, x0=np.array([t0]))[0]

    def compute_t_corresponding_to_y(self, y_seek: float, t0: float = 0.5):
        def bez_root_find_func(t):
            point = self.evaluate_point2d(t[0])
            return np.array([point.y.m - y_seek])

        return fsolve(bez_root_find_func, x0=np.array([t0]))[0]

    def convert_to_3d(self, plane: str = "XY"):
        valid_planes = ["XY", "YZ", "XZ"]
        plane_axis_mapping = {k: v for k, v in zip(valid_planes, [2, 0, 1])}
        if plane not in valid_planes:
            raise ValueError(f"Plane must be one of {valid_planes}. Given plane was {plane}")
        P = self.get_control_point_array()
        new_P = np.insert(P, plane_axis_mapping[plane], 0.0, axis=1)
        return Bezier3D.generate_from_array(new_P)

    def transform(self, **transformation_kwargs):
        transformation = Transformation2D(**transformation_kwargs)
        return Bezier2D.generate_from_array(transformation.transform(self.get_control_point_array()))

    def elevate_degree(self) -> "Bezier2D":
        """
        Elevates the degree of the Bézier curve. See algorithm source
        `here <https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/Bezier/bezier-elev.html>`_.

        Returns
        -------
        Bezier2D
            A new Bézier curve with identical shape to the current one but with one additional control point.
        """
        n = self.degree
        P = self.get_control_point_array()

        # New array has one additional control point (current array only has n+1 control points)
        new_control_points = np.zeros((P.shape[0] + 1, P.shape[1]))

        # Set starting and ending control points to what they already were
        new_control_points[0, :] = P[0, :]
        new_control_points[-1, :] = P[-1, :]

        # Update all the other control points
        for i in range(1, n + 1):  # 1 <= i <= n
            new_control_points[i, :] = i / (n + 1) * P[i - 1, :] + (1 - i / (n + 1)) * P[i, :]

        return Bezier2D.generate_from_array(new_control_points)

    def split(self, t_split: float):

        # Number of control points, curve degree, control point array
        n_ctrl_points = len(self.control_points)
        degree = n_ctrl_points - 1
        P = np.array([p.as_array() for p in self.control_points])

        def de_casteljau(i: int, j: int) -> np.ndarray:
            """
            Based on https://en.wikipedia.org/wiki/De_Casteljau%27s_algorithm. Recursive algorithm where the
            base case is just the value of the ith original control point.

            Parameters
            ----------
            i: int
                Lower index
            j: int
                Upper index

            Returns
            -------
            np.ndarray
                A one-dimensional array containing the :math:`x` and :math:`y` values of a control point evaluated
                at :math:`(i,j)` for a Bézier curve split at the parameter value ``t_split``
            """
            if j == 0:
                return P[i, :]
            return de_casteljau(i, j - 1) * (1 - t_split) + de_casteljau(i + 1, j - 1) * t_split

        bez_split_1_P = np.array([de_casteljau(i=0, j=i) for i in range(n_ctrl_points)])
        bez_split_2_P = np.array([de_casteljau(i=i, j=degree - i) for i in range(n_ctrl_points)])

        bez_1_points = [self.control_points[0]] + [Point2D(Length(m=xy[0]), Length(m=xy[1])) for xy in bez_split_1_P[1:, :]]
        bez_2_points = [bez_1_points[-1]] + [Point2D(Length(m=xy[0]), Length(m=xy[1])) for xy in bez_split_2_P[1:-1, :]] + [
            self.control_points[-1]]

        return (
            Bezier2D(bez_1_points),
            Bezier2D(bez_2_points)
        )


class Bezier3D(PCurve3D):

    def __init__(self, control_points: typing.List[Point3D]):
        self.control_points = control_points
        self.curve_connections = []

    @property
    def degree(self):
        return len(self.control_points) - 1

    @degree.setter
    def degree(self, value):
        raise AttributeError("The 'degree' property is read-only. Use the Bezier3D.elevate_degree method to increase"
                             "the degree of the curve while retaining the shape, or manually add or remove control "
                             "points to change the degree directly.")

    def to_iges(self, *args, **kwargs) -> aerocaps.iges.entity.IGESEntity:
        return aerocaps.iges.curves.BezierIGES(
            control_points_XYZ=self.get_control_point_array(),
        )

    def reverse(self) -> "Bezier3D":
        return self.__class__(self.control_points[::-1])

    def to_rational_bezier_curve(self) -> "RationalBezierCurve3D":
        return RationalBezierCurve3D(self.control_points, np.ones(self.degree + 1))

    def projection_on_principal_plane(self, plane: str = "XY") -> Bezier2D:
        return Bezier2D(control_points=[pt.projection_on_principal_plane(plane) for pt in self.control_points])

    @staticmethod
    def finite_diff_P(P: np.ndarray, k: int, i: int):
        """Calculates the finite difference of the control points as shown in
        https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/Bezier/bezier-der.html

        Arguments
        =========
        P: np.ndarray
            Array of control points for the Bézier curve
        k: int
            Finite difference level (e.g., k = 1 is the first derivative finite difference)
        i: int
            An index referencing a location in the control point array
        """

        def finite_diff_recursive(_k, _i):
            if _k > 1:
                return finite_diff_recursive(_k - 1, _i + 1) - finite_diff_recursive(_k - 1, _i)
            else:
                return P[_i + 1, :] - P[_i, :]

        return finite_diff_recursive(k, i)

    def derivative(self, P: np.ndarray, t: np.ndarray, degree: int, order: int):
        r"""
        Calculates an arbitrary-order derivative of the Bézier curve

        Parameters
        ==========
        P: np.ndarray
            The control point array

        t: np.ndarray
            The parameter vector

        degree: int
            The degree of the Bézier curve

        order: int
            The derivative order. For example, ``order=2`` returns the second derivative.

        Returns
        =======
        np.ndarray
            An array of ``shape=(N,2)`` where ``N`` is the number of evaluated points specified by the :math:`t` vector.
            The columns represent :math:`C^{(m)}_x(t)` and :math:`C^{(m)}_y(t)`, where :math:`m` is the
            derivative order.
        """
        if order > degree:
            return np.zeros((len(t), 3))
        return np.sum(np.array([np.prod(np.array([degree - idx for idx in range(order)])) *
                                np.array([self.finite_diff_P(P, order, i)]).T *
                                np.array([bernstein_poly(degree - order, i, t)])
                                for i in range(degree + 1 - order)]), axis=0).T

    def get_control_point_array(self, unit: str = "m") -> np.ndarray:
        return np.array([p.as_array(unit=unit) for p in self.control_points])

    @classmethod
    def generate_from_array(cls, P: np.ndarray, unit: str = "m"):
        return cls([Point3D(x=Length(**{unit: xyz[0]}),
                            y=Length(**{unit: xyz[1]}),
                            z=Length(**{unit: xyz[2]})) for xyz in P])

    def evaluate_point3d(self, t: float) -> Point3D:
        n_ctrl_points = len(self.control_points)
        degree = n_ctrl_points - 1
        P = self.get_control_point_array()

        # Evaluate the curve
        x, y, z = 0.0, 0.0, 0.0
        for i in range(n_ctrl_points):
            # Calculate the x- and y-coordinates of the Bézier curve given the input vector t
            x += P[i, 0] * bernstein_poly(degree, i, t)
            y += P[i, 1] * bernstein_poly(degree, i, t)
            z += P[i, 2] * bernstein_poly(degree, i, t)

        return Point3D(x=Length(m=x), y=Length(m=y), z=Length(m=z))

    def evaluate_single_t(self, t: float) -> np.ndarray:
        return self.evaluate_point3d(t).as_array()

    def evaluate(self, t: np.array = None) -> PCurveData3D:
        r"""
        Evaluates the curve using an optionally specified parameter vector.

        Parameters
        ==========
        t: np.ndarray or ``None``
            Optional direct specification of the parameter vector for the curve. Not specifying this value
            gives a linearly spaced parameter vector from ``t_start`` or ``t_end`` with the default size.
            Default: ``None``

        Returns
        =======
        PCurveData
            Data class specifying the following information about the Bézier curve:

            .. math::

                    C_x(t), C_y(t), C'_x(t), C'_y(t), C''_x(t), C''_y(t), \kappa(t)

            where the :math:`x` and :math:`y` subscripts represent the :math:`x` and :math:`y` components of the
            vector-valued functions :math:`\vec{C}(t)`, :math:`\vec{C}'(t)`, and :math:`\vec{C}''(t)`.
        """
        # Pass the starting and ending parameter vector values to the parameter vector generator if they were
        # specified directly

        # Generate the parameter vector
        t = np.linspace(0.0, 1.0, 100) if t is None else t

        # Number of control points, curve degree, control point array
        n_ctrl_points = len(self.control_points)
        degree = n_ctrl_points - 1
        P = self.get_control_point_array()

        # Evaluate the curve
        x, y, z = np.zeros(t.shape), np.zeros(t.shape), np.zeros(t.shape)
        for i in range(n_ctrl_points):
            # Calculate the x- and y-coordinates of the Bézier curve given the input vector t
            x += P[i, 0] * bernstein_poly(degree, i, t)
            y += P[i, 1] * bernstein_poly(degree, i, t)
            z += P[i, 2] * bernstein_poly(degree, i, t)
        xyz = np.column_stack((x, y, z))

        # Calculate the first derivative
        first_deriv = self.derivative(P=P, t=t, degree=degree, order=1)
        xp = first_deriv[:, 0]
        yp = first_deriv[:, 1]
        zp = first_deriv[:, 2]

        # Calculate the second derivative
        second_deriv = self.derivative(P=P, t=t, degree=degree, order=2)
        xpp = second_deriv[:, 0]
        ypp = second_deriv[:, 1]
        zpp = second_deriv[:, 2]

        # Combine the derivative x and y data
        xpypzp = np.column_stack((xp, yp, zp))
        xppyppzpp = np.column_stack((xpp, ypp, zpp))

        # Calculate the curvature
        with np.errstate(divide='ignore', invalid='ignore'):
            # Calculate the curvature of the Bézier curve (k = kappa = 1 / R, where R is the radius of curvature)
            k = np.true_divide(
                np.linalg.norm(np.cross(xpypzp, xppyppzpp), axis=1),
                np.linalg.norm(xpypzp, axis=1) ** 3
            )

        # Calculate the radius of curvature: R = 1 / kappa
        with np.errstate(divide='ignore', invalid='ignore'):
            R = np.true_divide(1, k)

        return PCurveData3D(t=t, xyz=xyz, xpypzp=xpypzp, xppyppzpp=xppyppzpp, k=k, R=R)

    def compute_t_corresponding_to_x(self, x_seek: float, t0: float = 0.5):
        def bez_root_find_func(t):
            point = self.evaluate_point3d(t[0])
            return np.array([point.x.m - x_seek])

        return fsolve(bez_root_find_func, x0=np.array([t0]))[0]

    def compute_t_corresponding_to_y(self, y_seek: float, t0: float = 0.5):
        def bez_root_find_func(t):
            point = self.evaluate_point3d(t[0])
            return np.array([point.y.m - y_seek])

        return fsolve(bez_root_find_func, x0=np.array([t0]))[0]

    def compute_t_corresponding_to_z(self, z_seek: float, t0: float = 0.5):
        def bez_root_find_func(t):
            point = self.evaluate_point3d(t[0])
            return np.array([point.z.m - z_seek])

        return fsolve(bez_root_find_func, x0=np.array([t0]))[0]

    def transform(self, **transformation_kwargs) -> "Bezier3D":
        transformation = Transformation3D(**transformation_kwargs)
        return Bezier3D.generate_from_array(transformation.transform(self.get_control_point_array()))

    def elevate_degree(self) -> "Bezier3D":
        """
        Elevates the degree of the Bézier curve. See algorithm source
        `here <https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/Bezier/bezier-elev.html>`_.

        Returns
        -------
        Bezier3D
            A new Bézier curve with identical shape to the current one but with one additional control point.
        """
        n = self.degree
        P = self.get_control_point_array()

        # New array has one additional control point (current array only has n+1 control points)
        new_control_points = np.zeros((P.shape[0] + 1, P.shape[1]))

        # Set starting and ending control points to what they already were
        new_control_points[0, :] = P[0, :]
        new_control_points[-1, :] = P[-1, :]

        # Update all the other control points
        for i in range(1, n + 1):  # 1 <= i <= n
            new_control_points[i, :] = i / (n + 1) * P[i - 1, :] + (1 - i / (n + 1)) * P[i, :]

        return Bezier3D.generate_from_array(new_control_points)

    def split(self, t_split: float) -> ("Bezier3D", "Bezier3D"):

        # Number of control points, curve degree, control point array
        n_ctrl_points = len(self.control_points)
        degree = n_ctrl_points - 1
        P = np.array([p.as_array() for p in self.control_points])

        def de_casteljau(i: int, j: int) -> np.ndarray:
            """
            Based on https://en.wikipedia.org/wiki/De_Casteljau%27s_algorithm. Recursive algorithm where the
            base case is just the value of the ith original control point.

            Parameters
            ----------
            i: int
                Lower index
            j: int
                Upper index

            Returns
            -------
            np.ndarray
                A one-dimensional array containing the :math:`x` and :math:`y` values of a control point evaluated
                at :math:`(i,j)` for a Bézier curve split at the parameter value ``t_split``
            """
            if j == 0:
                return P[i, :]
            return de_casteljau(i, j - 1) * (1 - t_split) + de_casteljau(i + 1, j - 1) * t_split

        bez_split_1_P = np.array([de_casteljau(i=0, j=i) for i in range(n_ctrl_points)])
        bez_split_2_P = np.array([de_casteljau(i=i, j=degree - i) for i in range(n_ctrl_points)])

        bez_1_points = [self.control_points[0]] + [Point3D(
            Length(m=xyz[0]), Length(m=xyz[1]), Length(m=xyz[2])) for xyz in bez_split_1_P[1:, :]]
        bez_2_points = [bez_1_points[-1]] + [Point3D(
            Length(m=xyz[0]), Length(m=xyz[1]), Length(m=xyz[2])) for xyz in bez_split_2_P[1:-1, :]] + [self.control_points[-1]]

        return (
            Bezier3D(bez_1_points),
            Bezier3D(bez_2_points)
        )

    def plot(self, ax: plt.Axes or pv.Plotter, projection: str = None, t_vec: np.ndarray = None, **plt_kwargs):
        projection = "XYZ" if projection is None else projection
        t_vec = np.linspace(0.0, 1.0, 201) if t_vec is None else None
        data = self.evaluate(t_vec).xyz
        args = tuple([data[:, _projection_dict[axis]] for axis in projection])

        if isinstance(ax, plt.Axes):
            ax.plot(*args, **plt_kwargs)
        elif isinstance(ax, pv.Plotter):
            arr = [data[0]]
            for row in data[1:-1]:
                arr.append(row)
                arr.append(row)
            arr.append(data[-1])
            ax.add_lines(np.array(arr), **plt_kwargs)


class NURBSCurve3D(Geometry3D):
    def __init__(self,
                 control_points: typing.List[Point3D] or np.ndarray,
                 weights: np.ndarray,
                 knot_vector: np.ndarray,
                 degree: int):
        """
        Non-uniform rational B-spline (NURBS) curve evaluation class
        """
        control_points = [Point3D.from_array(p) for p in control_points] if isinstance(
            control_points, np.ndarray) else control_points
        assert weights.ndim == 1
        assert knot_vector.ndim == 1
        assert len(knot_vector) == len(control_points) + degree + 1
        assert len(control_points) == len(weights)

        # Negative weight check
        for weight in weights:
            if weight < 0:
                raise NegativeWeightError("All weights must be non-negative")

        self.control_points = control_points
        self.weights = np.array(weights)
        self.knot_vector = np.array(knot_vector)
        self.degree = degree
        self.possible_spans, self.possible_span_indices = self._get_possible_spans()

    def to_iges(self, *args, **kwargs) -> aerocaps.iges.entity.IGESEntity:
        return aerocaps.iges.curves.RationalBSplineCurveIGES(
            knots=self.knot_vector,
            weights=self.weights,
            control_points_XYZ=self.get_control_point_array(),
            degree=self.degree
        )

    def reverse(self) -> "NURBSCurve3D":
        return self.__class__(np.flipud(self.get_control_point_array()),
                              self.weights[::-1],
                              (1.0 - self.knot_vector)[::-1],
                              self.degree)

    def get_control_point_array(self) -> np.ndarray:
        return np.array([p.as_array() for p in self.control_points])

    def get_homogeneous_control_points(self) -> np.ndarray:
        r"""
        Gets the array of control points in homogeneous coordinates, :math:`\mathbf{P}_i \cdot w_i`

        Returns
        -------
        numpy.ndarray
            Array of size :math:`(n + 1) \times 4`, where :math:`n` is the curve degree. The four columns, in order,
            represent the :math:`x`-coordinate, :math:`y`-coordinate, :math:`z`-coordinate, and weight of each
            control point.
        """
        return np.column_stack((
            self.get_control_point_array() * np.repeat(self.weights[:, np.newaxis], 3, axis=1),
            self.weights
        ))

    def evaluate_ndarray(self, t: float) -> np.ndarray:
        """
        Evaluate the NURBS curve at parameter t
        """
        B = self._basis_functions(t, self.degree)
        P = self.get_control_point_array()
        point = np.dot(B * self.weights, P) / np.sum(B * self.weights)
        return point

    def evaluate_simple(self, t: float) -> Point3D:
        """
        Evaluates the NURBS curve at parameter t as a ``Point3D``

        Parameters
        ----------
        t: float
            Parameter value

        Returns
        -------
        Point3D
            Value of the NURBS curve at the specified parameter value
        """
        return Point3D.from_array(self.evaluate_ndarray(t))

    def evaluate(self, t_vec: np.ndarray) -> np.ndarray:
        """
        Evaluates the NURBS curve at a vector of parameter values
        """
        points = np.array([self.evaluate_ndarray(t) for t in t_vec])
        return points

    def _get_possible_spans(self) -> (np.ndarray, np.ndarray):
        possible_span_indices = np.array([], dtype=int)
        possible_spans = []
        for knot_idx, (knot_1, knot_2) in enumerate(zip(self.knot_vector[:-1], self.knot_vector[1:])):
            if knot_1 == knot_2:
                continue
            possible_span_indices = np.append(possible_span_indices, knot_idx)
            possible_spans.append([knot_1, knot_2])
        return np.array(possible_spans), possible_span_indices

    def _cox_de_boor(self, t: float, i: int, p: int) -> float:
        if p == 0:
            return 1.0 if i in self.possible_span_indices and self._find_span(t) == i else 0.0
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                f = (t - self.knot_vector[i]) / (self.knot_vector[i + p] - self.knot_vector[i])
                g = (self.knot_vector[i + p + 1] - t) / (self.knot_vector[i + p + 1] - self.knot_vector[i + 1])
                if np.isinf(f) or np.isnan(f):
                    f = 0.0
                if np.isinf(g) or np.isnan(g):
                    g = 0.0
                if f == 0.0 and g == 0.0:
                    return 0.0
                elif f != 0.0 and g == 0.0:
                    return f * self._cox_de_boor(t, i, p - 1)
                elif f == 0.0 and g != 0.0:
                    return g * self._cox_de_boor(t, i + 1, p - 1)
                else:
                    return f * self._cox_de_boor(t, i, p - 1) + g * self._cox_de_boor(t, i + 1, p - 1)

    def _basis_functions(self, t: float, p: int):
        """
        Compute the non-zero basis functions at parameter t
        """
        return np.array([self._cox_de_boor(t, i, p) for i in range(self.control_points.shape[0])])

    def _find_span(self, t: float):
        """
        Find the knot span index
        """
        # insertion_point = bisect.bisect_left(self.non_repeated_knots, t)
        # return self.possible_spans[insertion_point - 1]
        for knot_span, knot_span_idx in zip(self.possible_spans, self.possible_span_indices):
            if knot_span[0] <= t < knot_span[1]:
                return knot_span_idx
        if t == self.possible_spans[-1][1]:
            return self.possible_span_indices[-1]


class RationalBezierCurve3D(Geometry3D):

    def __init__(self,
                 control_points: typing.List[Point3D],
                 weights: np.ndarray):
        """
        Non-uniform rational B-spline (NURBS) curve evaluation class
        """
        self.control_points = control_points
        assert weights.ndim == 1
        assert len(control_points) == len(weights)

        # Negative weight check
        for weight in weights:
            if weight < 0:
                raise NegativeWeightError("All weights must be non-negative")

        self.dim = 3
        self.weights = np.array(weights)
        self.knot_vector = np.zeros(2 * len(control_points))
        self.knot_vector[len(control_points):] = 1.0
        self.degree = len(control_points) - 1
        assert self.knot_vector.ndim == 1
        assert len(self.knot_vector) == len(control_points) + self.degree + 1

    def to_iges(self, *args, **kwargs) -> aerocaps.iges.entity.IGESEntity:
        return aerocaps.iges.curves.RationalBSplineCurveIGES(
            knots=self.knot_vector,
            weights=self.weights,
            control_points_XYZ=self.get_control_point_array(),
            degree=self.degree
        )

    def reverse(self) -> "RationalBezierCurve3D":
        return self.__class__(self.control_points[::-1],
                              self.weights[::-1])

    def elevate_degree(self) -> "RationalBezierCurve3D":
        """
        Elevates the degree of the rational Bézier curve. Uses the same algorithm as degree elevation of a
        non-rational Bézier curve with a necessary additional step of conversion to/from
        `homogeneous coordinates <https://en.wikipedia.org/wiki/Homogeneous_coordinates>`_.

        .. figure:: ../images/quarter_circle_degree_elevation.*
            :width: 350
            :align: center

            Degree elevation of a quarter circle exactly represented by a rational Bézier curve

        Returns
        -------
        RationalBezierCurve3D
            A new rational Bézier curve with identical shape to the current one but with one additional control point.
        """
        n = self.degree
        Pw = self.get_homogeneous_control_points()

        # New array has one additional control point (current array only has n+1 control points)
        new_homogeneous_control_points = np.zeros((Pw.shape[0] + 1, Pw.shape[1]))

        # Set starting and ending control points to what they already were
        new_homogeneous_control_points[0, :] = Pw[0, :]
        new_homogeneous_control_points[-1, :] = Pw[-1, :]

        # Update all the other control points
        for i in range(1, n + 1):  # 1 <= i <= n
            new_homogeneous_control_points[i, :] = i / (n + 1) * Pw[i - 1, :] + (1 - i / (n + 1)) * Pw[i, :]

        # Project the homogeneous control points onto the w=1 hyperplane
        new_weights = new_homogeneous_control_points[:, -1]
        new_control_points = new_homogeneous_control_points[:, :-1] / np.repeat(new_weights[:, np.newaxis], 3, axis=1)

        return RationalBezierCurve3D.generate_from_array(new_control_points, new_weights)

    def evaluate_ndarray(self, t: float) -> np.ndarray:
        """
        Evaluate the NURBS curve at parameter t
        """
        return self.evaluate_simple(t).as_array()

    def get_control_point_array(self) -> np.ndarray:
        return np.array([p.as_array() for p in self.control_points])

    def get_homogeneous_control_points(self) -> np.ndarray:
        r"""
        Gets the array of control points in homogeneous coordinates, :math:`\mathbf{P}_i \cdot w_i`

        Returns
        -------
        numpy.ndarray
            Array of size :math:`(n + 1) \times 4`, where :math:`n` is the curve degree. The four columns, in order,
            represent the :math:`x`-coordinate, :math:`y`-coordinate, :math:`z`-coordinate, and weight of each
            control point.
        """
        return np.column_stack((
            self.get_control_point_array() * np.repeat(self.weights[:, np.newaxis], 3, axis=1),
            self.weights
        ))

    @classmethod
    def generate_from_array(cls, P: np.ndarray, weights: np.ndarray):
        return cls([Point3D(x=Length(m=xyz[0]), y=Length(m=xyz[1]), z=Length(m=xyz[2])) for xyz in P], weights)

    def evaluate_simple(self, t: float) -> Point3D:
        n_ctrl_points = len(self.control_points)
        degree = n_ctrl_points - 1
        P = self.get_control_point_array()

        # Evaluate the curve
        w_sum = sum([bernstein_poly(degree, i, t) * self.weights[i] for i in range(n_ctrl_points)])
        x = sum([P[i, 0] * bernstein_poly(degree, i, t) * self.weights[i] for i in range(n_ctrl_points)])
        y = sum([P[i, 1] * bernstein_poly(degree, i, t) * self.weights[i] for i in range(n_ctrl_points)])
        z = sum([P[i, 2] * bernstein_poly(degree, i, t) * self.weights[i] for i in range(n_ctrl_points)])

        return Point3D(x=Length(m=x / w_sum), y=Length(m=y / w_sum), z=Length(m=z / w_sum))
        # return Point3D(x=Length(m=x), y=Length(m=y), z=Length(m=z))

    def evaluate(self, t_vec: np.ndarray) -> np.ndarray:
        """
        Evaluates the NURBS curve at a vector of parameter values
        """
        points = np.array([self.evaluate_simple(t) for t in t_vec])
        return points

    def compute_t_corresponding_to_x(self, x_seek: float, t0: float = 0.5):
        def bez_root_find_func(t):
            point = self.evaluate_simple(t[0])
            return np.array([point.x.m - x_seek])

        return fsolve(bez_root_find_func, x0=np.array([t0]))[0]

    def compute_t_corresponding_to_y(self, y_seek: float, t0: float = 0.5):
        def bez_root_find_func(t):
            point = self.evaluate_simple(t[0])
            return np.array([point.y.m - y_seek])

        return fsolve(bez_root_find_func, x0=np.array([t0]))[0]

    def compute_t_corresponding_to_z(self, z_seek: float, t0: float = 0.5):
        def bez_root_find_func(t):
            point = self.evaluate_simple(t[0])
            return np.array([point.z.m - z_seek])

        return fsolve(bez_root_find_func, x0=np.array([t0]))[0]

    def plot(self, ax: plt.Axes or pv.Plotter, projection: str = None, t_vec: np.ndarray = None, **plt_kwargs):
        projection = "XYZ" if projection is None else projection
        t_vec = np.linspace(0.0, 1.0, 201) if t_vec is None else None
        data = self.evaluate(t_vec)
        args = tuple([data[:, _projection_dict[axis]] for axis in projection])

        if isinstance(ax, plt.Axes):
            ax.plot(*args, **plt_kwargs)
        elif isinstance(ax, pv.Plotter):
            arr = [data[0]]
            for row in data[1:-1]:
                arr.append(row)
                arr.append(row)
            arr.append(data[-1])
            ax.add_lines(np.array(arr), **plt_kwargs)

    def plot_control_points(self, ax: plt.Axes, projection: str = None, **plt_kwargs):
        projection = "XYZ" if projection is None else projection
        cps = self.get_control_point_array()
        args = tuple([cps[:, _projection_dict[axis]] for axis in projection])
        ax.plot(*args, **plt_kwargs)

    def compute_curvature_at_t0(self) -> float:
        """
        Computes the curvature at :math:`t=0` according to Eq. (20) in
        M. S. Floater, “Derivatives of Rational Bézier Curves,” Computer Aided Geometric Design, vol. 9, no. 3,
        pp. 161–174, Aug. 1992, issn: 0167-8396. doi: 10.1016/0167-8396(92)90014-G.

        Returns
        -------
        float
            The curvature at :math:`t=0`
        """
        R1 = (self.weights[0] * self.weights[2]) / self.weights[1]**2
        P01 = Vector3D(p0=self.control_points[0], p1=self.control_points[1])
        P12 = Vector3D(p0=self.control_points[1], p1=self.control_points[2])
        return (self.degree - 1) / self.degree * R1 * P01.cross(P12).mag().m / P01.mag().m**3

    def compute_curvature_at_t1(self) -> float:
        """
        Computes the curvature at :math:`t=1` according to Eq. (20) in
        M. S. Floater, “Derivatives of Rational Bézier Curves,” Computer Aided Geometric Design, vol. 9, no. 3,
        pp. 161–174, Aug. 1992, issn: 0167-8396. doi: 10.1016/0167-8396(92)90014-G.

        Returns
        -------
        float
            The curvature at :math:`t=1`
        """
        R1 = (self.weights[-1] * self.weights[-3]) / self.weights[-2] ** 2
        P_nm2_nm1 = Vector3D(p0=self.control_points[-3], p1=self.control_points[-2])
        P_nm1_n = Vector3D(p0=self.control_points[-2], p1=self.control_points[-1])
        return (self.degree - 1) / self.degree * R1 * P_nm2_nm1.cross(P_nm1_n).mag().m / P_nm1_n.mag().m ** 3

    def enforce_g0(self, other: "RationalBezierCurve3D"):
        other.control_points[0] = self.control_points[-1]

    def enforce_c0(self, other: "RationalBezierCurve3D"):
        self.enforce_g0(other)

    def enforce_g0g1(self, other: "RationalBezierCurve3D", f: float):
        self.enforce_g0(other)
        n_ratio = self.degree / other.degree
        w_ratio_a = self.weights[-2] / self.weights[-1]
        w_ratio_b = other.weights[0] / other.weights[1]
        other.control_points[1] = other.control_points[0] + f * n_ratio * w_ratio_a * w_ratio_b * (self.control_points[-1] - self.control_points[-2])

    def enforce_c0c1(self, other: "RationalBezierCurve3D"):
        self.enforce_g0g1(other, f=1.0)

    def enforce_g0g1g2(self, other: "RationalBezierCurve3D", f: float):
        self.enforce_g0g1(other, f)
        n_ratio_1 = self.degree / other.degree
        n_ratio_2 = (self.degree - 1) / (other.degree - 1)
        n_ratio_3 = 1 / (other.degree - 1)
        w_ratio_1 = self.weights[-3] / self.weights[-1]
        w_ratio_2 = other.weights[0] / other.weights[2]
        w_ratio_3 = self.weights[-2] / self.weights[-1]
        w_ratio_4 = other.weights[1] / other.weights[0]
        other.control_points[2] = other.control_points[1] + f ** 2 * n_ratio_1 * n_ratio_2 * w_ratio_1 * w_ratio_2 * (
                self.control_points[-3] - self.control_points[-2]) - f ** 2 * n_ratio_1 * n_ratio_3 * w_ratio_2 * (
                2 * self.degree * w_ratio_3**2 - (self.degree - 1) * w_ratio_1 - 2 * w_ratio_3) * (
                                          self.control_points[-2] - self.control_points[-1]) + n_ratio_3 * w_ratio_2 * (
                2 * other.degree * w_ratio_4**2 - (other.degree - 1) * w_ratio_2**-1 - 2 * w_ratio_4) * (
                                          other.control_points[1] - other.control_points[0])

    def enforce_c0c1c2(self, other: "RationalBezierCurve3D"):
        self.enforce_g0g1g2(other, f=1.0)


class BSpline3D(Geometry3D):
    def __init__(self,
                 control_points: np.ndarray,
                 knot_vector: np.ndarray,
                 degree: int):
        """
        Non-uniform rational B-spline (NURBS) curve evaluation class
        """
        assert control_points.ndim == 2
        assert knot_vector.ndim == 1
        assert len(knot_vector) == len(control_points) + degree + 1

        self.control_points = control_points
        self.dim = self.control_points.shape[1]
        self.knot_vector = np.array(knot_vector)
        self.weights = np.ones(self.control_points[:, 0].shape)
        self.degree = degree
        self.possible_spans, self.possible_span_indices = self._get_possible_spans()

    def to_iges(self, *args, **kwargs) -> aerocaps.iges.entity.IGESEntity:
        return aerocaps.iges.curves.RationalBSplineCurveIGES(
            knots=self.knot_vector,
            weights=self.weights,
            control_points_XYZ=self.control_points,
            degree=self.degree
        )

    def reverse(self) -> "BSpline3D":
        return self.__class__(np.flipud(self.control_points),
                              (1.0 - self.knot_vector)[::-1],
                              self.degree)

    def evaluate_ndarray(self, t: float) -> np.ndarray:
        """
        Evaluate the NURBS curve at parameter t
        """
        B = self._basis_functions(t, self.degree)
        point = np.dot(B, self.control_points)
        return point

    def evaluate_simple(self, t: float) -> Point3D:
        """
        Evaluates the NURBS curve at parameter t as a ``Point3D``

        Parameters
        ----------
        t: float
            Parameter value

        Returns
        -------
        Point3D
            Value of the NURBS curve at the specified parameter value
        """
        return Point3D.from_array(self.evaluate_ndarray(t))

    def evaluate(self, t_vec: np.ndarray) -> np.ndarray:
        """
        Evaluates the NURBS curve at a vector of parameter values
        """
        points = np.array([self.evaluate_ndarray(t) for t in t_vec])
        return points

    def _get_possible_spans(self) -> (np.ndarray, np.ndarray):
        possible_span_indices = np.array([], dtype=int)
        possible_spans = []
        for knot_idx, (knot_1, knot_2) in enumerate(zip(self.knot_vector[:-1], self.knot_vector[1:])):
            if knot_1 == knot_2:
                continue
            possible_span_indices = np.append(possible_span_indices, knot_idx)
            possible_spans.append([knot_1, knot_2])
        return np.array(possible_spans), possible_span_indices

    def _cox_de_boor(self, t: float, i: int, p: int) -> float:
        if p == 0:
            return 1.0 if i in self.possible_span_indices and self._find_span(t) == i else 0.0
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                f = (t - self.knot_vector[i]) / (self.knot_vector[i + p] - self.knot_vector[i])
                g = (self.knot_vector[i + p + 1] - t) / (self.knot_vector[i + p + 1] - self.knot_vector[i + 1])
                if np.isinf(f) or np.isnan(f):
                    f = 0.0
                if np.isinf(g) or np.isnan(g):
                    g = 0.0
                if f == 0.0 and g == 0.0:
                    return 0.0
                elif f != 0.0 and g == 0.0:
                    return f * self._cox_de_boor(t, i, p - 1)
                elif f == 0.0 and g != 0.0:
                    return g * self._cox_de_boor(t, i + 1, p - 1)
                else:
                    return f * self._cox_de_boor(t, i, p - 1) + g * self._cox_de_boor(t, i + 1, p - 1)

    def _basis_functions(self, t: float, p: int):
        """
        Compute the non-zero basis functions at parameter t
        """
        return np.array([self._cox_de_boor(t, i, p) for i in range(self.control_points.shape[0])])

    def _find_span(self, t: float):
        """
        Find the knot span index
        """
        # insertion_point = bisect.bisect_left(self.non_repeated_knots, t)
        # return self.possible_spans[insertion_point - 1]
        for knot_span, knot_span_idx in zip(self.possible_spans, self.possible_span_indices):
            if knot_span[0] <= t < knot_span[1]:
                return knot_span_idx
        if t == self.possible_spans[-1][1]:
            return self.possible_span_indices[-1]


class CompositeCurve3D(Geometry3D):
    def __init__(self, curves: typing.List[PCurve3D]):
        self.curves = curves

    def to_iges(self, curve_iges_entities: typing.List[aerocaps.iges.entity.IGESEntity],
                *args, **kwargs) -> aerocaps.iges.entity.IGESEntity:
        return aerocaps.iges.curves.CompositeCurveIGES(
            curve_iges_entities
        )


class CurveOnParametricSurface(Geometry3D):
    def __init__(self,
                 surface: aerocaps.geom.Surface,
                 parametric_curve: aerocaps.geom.Geometry3D,
                 model_space_curve: aerocaps.geom.Geometry3D):
        self.surface = surface
        self.parametric_curve = parametric_curve
        self.model_space_curve = model_space_curve

    def to_iges(self,
                surface_iges: aerocaps.iges.entity.IGESEntity,
                parametric_curve: aerocaps.iges.entity.IGESEntity,
                model_space_curve: aerocaps.iges.entity.IGESEntity,
                *args, **kwargs) -> aerocaps.iges.entity.IGESEntity:
        return aerocaps.iges.curves.CurveOnParametricSurfaceIGES(
            surface_iges,
            parametric_curve,
            model_space_curve
        )


def main():
    bspline = BSpline3D(np.array([
        [1.0, 0.05, 0.0],
        [0.8, 0.12, 0.0],
        [0.6, 0.2, 0.0],
        [0.2, 0.3, 0.0],
        [0.0, 0.05, 0.0],
        [0.0, -0.1, 0.0],
        [0.4, -0.4, 0.0],
        [0.6, -0.05, 0.0],
        [1.0, -0.05, 0.0]
    ]), knot_vector=np.array([0.0, 0.0, 0.0, 0.0, 0.2, 0.375, 0.5, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0]),
        degree=3
    )
    data = bspline.evaluate(np.linspace(0.0, 1.0, 301))
    plt.plot(data[:, 0], data[:, 1], color="steelblue")
    plt.plot(bspline.control_points[:, 0], bspline.control_points[:, 1], ls=":", color="grey", marker="o", mec="steelblue", mfc="none")
    plt.plot([data[75, 0], data[150, 0], data[225, 0]], [data[75, 1], data[150, 1], data[225, 1]], ls="none", marker="o", mfc="indianred", mec="indianred")
    plt.show()


if __name__ == "__main__":
    main()
