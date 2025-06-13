"""
Parametric curve classes (one-dimensional geometric objects defined by parameter :math:`t` that reside in
two- or three-dimensional space)
"""
import typing
from abc import abstractmethod
from copy import deepcopy

import numpy as np
import pyvista as pv
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
from rust_nurbs import *

import aerocaps.iges
import aerocaps.iges.curves
import aerocaps.iges.entity
from aerocaps.geom import Geometry2D, Geometry3D, NegativeWeightError
from aerocaps.geom.point import Point2D, Point3D
from aerocaps.geom.transformation import Transformation2D, Transformation3D
from aerocaps.geom.vector import Vector3D, Vector2D
from aerocaps.units.angle import Angle
from aerocaps.units.length import Length

__all__ = [
    "PCurveData2D",
    "PCurveData3D",
    "PCurve2D",
    "PCurve3D",
    "Line2D",
    "Line3D",
    "CircularArc2D",
    "BezierCurve2D",
    "BezierCurve3D",
    "BSplineCurve3D",
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
    """Data-processing class for 2-D parametric curves"""
    def __init__(self,
                 t: float or np.ndarray,
                 x: float or np.ndarray,
                 y: float or np.ndarray,
                 xp: float or np.ndarray,
                 yp: float or np.ndarray,
                 xpp: float or np.ndarray,
                 ypp: float or np.ndarray,
                 k: float or np.ndarray = None,
                 R: float or np.ndarray = None):
        r"""
        Data-processing class for 2-D parametric curves. Adds convenience methods for plotting, approximating
        arc length, and computing curvature combs.

        Parameters
        ----------
        t: float or numpy.ndarray
            :math:`t`-value or vector for the curve
        x: float or numpy.ndarray
            Value of the curve in the :math:`x`-direction
        y: float or numpy.ndarray
            Value of the curve in the :math:`y`-direction
        xp: float or numpy.ndarray
            First derivative :math:`x`-component of the curve with respect to :math:`t`
        yp: float or numpy.ndarray
            First derivative :math:`y`-component of the curve with respect to :math:`t`
        xpp: float or numpy.ndarray
            Second derivative :math:`x`-component of the curve with respect to :math:`t`
        ypp: float or numpy.ndarray
            Second derivative :math:`y`-component of the curve with respect to :math:`t`
        k: float or v.ndarray
            Curvature of the curve. If not specified, it will be computed. Default: ``None``
        R: float or numpy.ndarray
            Radius of curvature of the curve. If not specified, it will be computed. Default: ``None``
        """
        self.t = t
        self.x = x
        self.y = y
        self.xp = xp
        self.yp = yp
        self.xpp = xpp
        self.ypp = ypp
        self.k = k if k is not None else self._compute_curvature()
        with np.errstate(divide="ignore", invalid="ignore"):
            self.R = R if R is not None else 1 / self.k
            self.R_abs_min = np.min(np.abs(self.R))

    def _compute_curvature(self) -> float or np.ndarray:
        r"""
        Computes the curvature from the first and second derivatives

        Returns
        -------
        float or np.ndarray
            Same output type/shape as ``t``
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            A = self.xp * self.ypp - self.yp * self.xpp
            B = np.hypot(self.xp, self.yp) ** 3
            return A / B

    def plot(self, ax: plt.Axes, **kwargs):
        r"""
        Plots the curve on a :obj:`matplotlib.pyplot.Axes`

        .. note::

            If ``t`` is a :obj:`float`, specifying the ``marker`` keyword argument may be desirable as a line
            plot cannot be generated in this case

        Parameters
        ----------
        ax: plt.Axes
            Axis on which to plot
        kwargs: dict
            Keyword arguments to pass to :obj:`matplotlib.pyplot.Axes.plot`
        """
        if isinstance(self.t, float):
            ax.plot([self.x], [self.y], **kwargs)
        else:
            ax.plot(self.x, self.y, **kwargs)

    def get_curvature_comb(self, max_comb_length: float, interval: int = 1) -> (np.ndarray, np.ndarray):
        r"""
        Gets the curvature comb for the curve

        .. note::

            This method will raise an exception if ``t`` is a :obj:`float`

        Parameters
        ----------
        max_comb_length: float
            Length of the comb corresponding to the location of maximum unsigned curvature
        interval: int
            Interval between the output combs. If ``1``, the combs corresponding to every evaluated :math:`t`-value
            will be used. If ``2``, every other comb will be skipped, etc. The first and last combs are
            guaranteed to be output as long as ``len(t) > 1``. Default: ``1``

        Returns
        -------
        numpy.ndarray, numpy.ndarray
            2-D arrays corresponding to the comb heads and comb tails, respectively. Each array has outer dimension
            dependent on the interval (if ``interval=1``, the length of this dimension will be ``len(t)``) and
            inner dimension 2
        """
        if isinstance(self.t, float):
            raise ValueError(f"Curvature comb calculation is only available for array-type curve data")
        first_deriv_mag = np.hypot(self.xp, self.yp)
        abs_k = np.abs(self.k)
        normalized_k = abs_k / max(abs_k)
        comb_heads_x = self.x - self.yp / first_deriv_mag * normalized_k * max_comb_length
        comb_heads_y = self.y + self.xp / first_deriv_mag * normalized_k * max_comb_length
        # Stack the x and y columns (except for the last x and y values) horizontally and keep only the rows by the
        # specified interval:
        comb_tails = np.column_stack((self.x, self.y))[:-1:interval, :]
        comb_heads = np.column_stack((comb_heads_x, comb_heads_y))[:-1:interval, :]
        # Add the last x and y values onto the end (to make sure they do not get skipped with input interval)
        comb_tails = np.vstack((comb_tails, np.array([self.x[-1], self.y[-1]])))
        comb_heads = np.vstack((comb_heads, np.array([comb_heads_x[-1], comb_heads_y[-1]])))
        return comb_tails, comb_heads

    def approximate_arc_length(self) -> np.ndarray:
        r"""
        Approximates the arc-length of the curve by summing the arc-lengths of each segment of the evaluated
        polyline

        .. note::

            This method will raise an exception if ``t`` is a :obj:`float`

        Returns
        -------
        numpy.ndarray
            1-D array with :math:`(\text{len}(t) - 1)` elements
        """
        if isinstance(self.t, float):
            raise ValueError(f"Arc-length calculation is only available for array-type curve data")
        return np.sum(np.hypot(self.x[1:] - self.x[:-1], self.y[1:] - self.y[:-1]))


class PCurve2D(Geometry2D):
    """Two-dimensional abstract parametric curve class"""
    @abstractmethod
    def evaluate_point2d(self, t: float or int or np.ndarray) -> Point2D or typing.List[Point2D]:
        r"""
        Evaluates the line at one or more :math:`t`-values and returns a single point object or list of point objects

        Parameters
        ----------
        t: float or int or numpy.ndarray
            Either a single :math:`t`-value, a number of evenly spaced :math:`t`-values between 0 and 1, or
            a 1-D array of :math:`t`-values

        Returns
        -------
        Point2D or typing.List[Point2D]
            If ``t`` is a :obj:`float`, the output is a single point object. Otherwise, the output is a list of
            point objects
        """
        pass

    @abstractmethod
    def evaluate(self, t: float or int or np.ndarray) -> np.ndarray:
        r"""
        Evaluates the line at one or more :math:`t`-values

        Parameters
        ----------
        t: float or int or numpy.ndarray
            Either a single :math:`t`-value, a number of evenly spaced :math:`t`-values between 0 and 1, or
            a 1-D array of :math:`t`-values

        Returns
        -------
        numpy.ndarray
            If ``t`` is a :obj:`float`, the output is a 1-D array with two elements: the values of :math:`x` and
            :math:`y`. Otherwise, the output is an array of size :math:`\text{len}(t) \times 2`
        """
        pass

    @abstractmethod
    def dcdt(self, t: float or int or np.ndarray) -> np.ndarray:
        r"""
        Evaluates the first derivative of the curve with respect to :math:`t`

        Parameters
        ----------
        t: float or int or numpy.ndarray
            Either a single :math:`t`-value, a number of evenly spaced :math:`t`-values between 0 and 1, or
            a 1-D array of :math:`t`-values

        Returns
        -------
        numpy.ndarray
            If :math:`t` is a :obj:`float`, the output is a 1-D array containing two elements: the :math:`x`-
            and :math:`y`-components of the first derivative. Otherwise, the output is a 2-D array of size
            :math:`\text{len}(t) \times 2`
        """
        pass

    @abstractmethod
    def d2cdt2(self, t: float or int or np.ndarray) -> np.ndarray:
        r"""
        Evaluates the second derivative of the curve with respect to :math:`t`

        Parameters
        ----------
        t: float or int or numpy.ndarray
            Either a single :math:`t`-value, a number of evenly spaced :math:`t`-values between 0 and 1, or
            a 1-D array of :math:`t`-values

        Returns
        -------
        numpy.ndarray
            If :math:`t` is a :obj:`float`, the output is a 1-D array containing two elements: the :math:`x`-
            and :math:`y`-components of the second derivative. Otherwise, the output is a 2-D array of size
            :math:`\text{len}(t) \times 2`
        """
        pass

    @abstractmethod
    def evaluate_pcurvedata(self, t: float or int or np.ndarray) -> PCurveData2D:
        r"""
        Evaluates a verbose set of parametric curve data as a class based on an input parameter value or vector

        Parameters
        ----------
        t: float or int or numpy.ndarray
            Either a single :math:`t`-value, a number of evenly spaced :math:`t`-values between 0 and 1, or
            a 1-D array of :math:`t`-values

        Returns
        -------
        PCurveData2D
            Parametric curve information, including derivative and curvature data
        """
        pass

    @staticmethod
    def _get_linear_tvec(nt: int) -> np.ndarray:
        r"""
        Gets a linear vector of parameter values based on a number of points

        Parameters
        ----------
        nt: int
            Number of linearly-spaced :math:`t`-values to generate
        """
        return np.linspace(0.0, 1.0, nt)

    @staticmethod
    def _validate_and_convert_t(t: float or int or np.ndarray) -> float or np.ndarray:
        r"""
        Validates the :math:`t`-value and converts it to an array if necessary

        Parameters
        ----------
        t: float or int or numpy.ndarray
            Either a single :math:`t`-value, a number of evenly spaced :math:`t`-values between 0 and 1, or
            a 1-D array of :math:`t`-values

        Returns
        -------
        float or numpy.ndarray
            Either a single :math:`t`-value or a 1-D array of :math:`t`-values, depending on the input type
        """
        if isinstance(t, int) and t < 2:
            raise ValueError(f"If `t` is an integer, a value of at least 2 must be specified. If you are intending to "
                             f"evaluate the curve at t=0 or t=1, please use the float format for these numbers "
                             f"(0.0 or 1.0)")
        return np.linspace(0.0, 1.0, t) if isinstance(t, int) else t


class PCurveData3D:
    """Data-processing class for 3-D parametric curves"""
    def __init__(self,
                 t: float or np.ndarray,
                 x: float or np.ndarray,
                 y: float or np.ndarray,
                 z: float or np.ndarray,
                 xp: float or np.ndarray,
                 yp: float or np.ndarray,
                 zp: float or np.ndarray,
                 xpp: float or np.ndarray,
                 ypp: float or np.ndarray,
                 zpp: float or np.ndarray,
                 k: float or np.ndarray = None,
                 R: float or np.ndarray = None):
        r"""
        Data-processing class for 3-D parametric curves. Adds convenience methods for plotting, approximating
        arc length, and computing curvature combs.

        Parameters
        ----------
        t: float or numpy.ndarray
            :math:`t`-value or vector for the curve
        x: float or numpy.ndarray
            Value of the curve in the :math:`x`-direction
        y: float or numpy.ndarray
            Value of the curve in the :math:`y`-direction
        z: float or numpy.ndarray
            Value of the curve in the :math:`x`-direction
        xp: float or numpy.ndarray
            First derivative :math:`x`-component of the curve with respect to :math:`t`
        yp: float or numpy.ndarray
            First derivative :math:`y`-component of the curve with respect to :math:`t`
        zp: float or numpy.ndarray
            First derivative :math:`z`-component of the curve with respect to :math:`t`
        xpp: float or numpy.ndarray
            Second derivative :math:`x`-component of the curve with respect to :math:`t`
        ypp: float or numpy.ndarray
            Second derivative :math:`y`-component of the curve with respect to :math:`t`
        zpp: float or numpy.ndarray
            Second derivative :math:`z`-component of the curve with respect to :math:`t`
        k: float or numpy.ndarray
            Curvature of the curve. If not specified, it will be computed. Default: ``None``
        R: float or numpy.ndarray
            Radius of curvature of the curve. If not specified, it will be computed. Default: ``None``
        """
        self.t = t
        self.x = x
        self.y = y
        self.z = z
        self.xp = xp
        self.yp = yp
        self.zp = zp
        self.xpp = xpp
        self.ypp = ypp
        self.zpp = zpp
        self.k = k if k is not None else self._compute_curvature()
        with np.errstate(divide="ignore", invalid="ignore"):
            self.R = R if R is not None else 1 / self.k
            self.R_abs_min = np.min(np.abs(self.R))

    def _compute_curvature(self) -> float or np.ndarray:
        r"""
        Computes the curvature from the first and second derivatives

        Returns
        -------
        float or numpy.ndarray
            Same output type/shape as ``t``
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            A = self.zpp * self.yp - self.ypp * self.zp
            B = self.xpp * self.zp - self.zpp * self.xp
            C = self.ypp * self.xp - self.xpp * self.yp
            D = (self.xp ** 2 + self.yp ** 2 + self.zp ** 2) ** 1.5
            return np.sqrt(A ** 2 + B ** 2 + C ** 2) / D

    def plot(self, ax: plt.Axes, **kwargs):
        r"""
        Plots the curve on an :obj:`matplotlib.pyplot.Axes`

        .. note::

            If ``t`` is a :obj:`float`, specifying the ``marker`` keyword argument may be desirable as a line
            plot cannot be generated in this case

        Parameters
        ----------
        ax: plt.Axes
            Axis on which to plot
        kwargs: dict
            Keyword arguments to pass to :obj:`matplotlib.pyplot.Axes.plot3D`
        """
        ax.plot3D(self.x, self.y, self.z, **kwargs)

    def get_curvature_comb(self, max_comb_length: float, interval: int = 1):
        r"""
        Gets the curvature comb for the curve

        .. note::

            This method will raise an exception if ``t`` is a :obj:`float`

        Parameters
        ----------
        max_comb_length: float
            Length of the comb corresponding to the location of maximum unsigned curvature
        interval: int
            Interval between the output combs. If ``1``, the combs corresponding to every evaluated :math:`t`-value
            will be used. If ``2``, every other comb will be skipped, etc. The first and last combs are
            guaranteed to be output as long as ``len(t) > 1``. Default: ``1``

        Returns
        -------
        numpy.ndarray, numpy.ndarray
            2-D arrays corresponding to the comb heads and comb tails, respectively. Each array has outer dimension
            dependent on the interval (if ``interval=1``, the length of this dimension will be ``len(t)``) and
            inner dimension 3
        """
        if isinstance(self.t, float):
            raise ValueError(f"Curvature comb calculation is only available for array-type curve data")
        abs_k = np.abs(self.k)
        normalized_k = abs_k / max(abs_k)
        rp_vec = np.column_stack((self.xp, self.yp, self.zp))
        rpp_vec = np.column_stack((self.xpp, self.ypp, self.zpp))
        rp_mag = np.linalg.norm(rp_vec)
        T = rp_vec / rp_mag  # Normalized tangent vector
        with np.errstate(divide="ignore", invalid="ignore"):
            B_cross = np.cross(rp_vec, rpp_vec)
            B = B_cross / np.linalg.norm(B_cross)  # Normalized bi-normal vector
        # Handle the case where the curvature is zero
        for t_idx in range(len(self.t)):
            if not np.isinf(B[t_idx][0]):
                continue
            B[t_idx][0] = T[t_idx][0] + 0.5
            B[t_idx][1] = T[t_idx][1] + 0.5
            # Ensure that the dot product between the bi-normal and tangent vectors is zero
            B[t_idx][2] = -(B[t_idx][0] * T[t_idx][0] + B[t_idx][1] * T[t_idx][1]) / T[t_idx][2]
            # Normalize the resulting vector
            B[t_idx][:] /= np.linalg.norm(B[t_idx][:])
        N = np.cross(B, T)  # Normal vector
        comb_heads_x = self.x[:] + N[:, 0] * normalized_k * max_comb_length
        comb_heads_y = self.y[:] + N[:, 1] * normalized_k * max_comb_length
        comb_heads_z = self.z[:] + N[:, 2] * normalized_k * max_comb_length
        # Stack the x and y columns (except for the last x and y values) horizontally and keep only the rows by the
        # specified interval:
        comb_tails = np.column_stack((self.x, self.y, self.z))[:-1:interval, :]
        comb_heads = np.column_stack((comb_heads_x, comb_heads_y))[:-1:interval, :]
        # Add the last x and y values onto the end (to make sure they do not get skipped with input interval)
        comb_tails = np.vstack((comb_tails, np.array([self.x[-1], self.y[-1], self.z[-1]])))
        comb_heads = np.vstack((comb_heads, np.array([comb_heads_x[-1], comb_heads_y[-1], comb_heads_z[-1]])))
        return comb_tails, comb_heads

    def approximate_arc_length(self) -> np.ndarray:
        r"""
        Approximates the arc-length of the curve by summing the arc-lengths of each segment of the evaluated
        polyline

        .. note::

            This method will raise an exception if ``t`` is a :obj:`float`

        Returns
        -------
        numpy.ndarray
            1-D array with :math:`(\text{len}(t) - 1)` elements
        """
        if isinstance(self.t, float):
            raise ValueError(f"Arc-length calculation is only available for array-type curve data")
        return np.sum(np.sqrt(
            (self.x[1:] - self.x[:-1]) ** 2 +
            (self.y[1:] - self.y[:-1]) ** 2 +
            (self.z[1:] - self.z[:-1]) ** 2
        ))


class PCurve3D(Geometry3D):
    """Three-dimensional abstract parametric curve class"""
    @abstractmethod
    def evaluate_point3d(self, t: float or int or np.ndarray) -> Point3D or typing.List[Point3D]:
        r"""
        Evaluates the line at one or more :math:`t`-values and returns a single point object or list of point objects

        Parameters
        ----------
        t: float or int or numpy.ndarray
            Either a single :math:`t`-value, a number of evenly spaced :math:`t`-values between 0 and 1, or
            a 1-D array of :math:`t`-values

        Returns
        -------
        Point3D or typing.List[Point3D]
            If ``t`` is a :obj:`float`, the output is a single point object. Otherwise, the output is a list of
            point objects
        """
        pass

    @abstractmethod
    def evaluate(self, t: float or int or np.ndarray) -> float or np.ndarray:
        r"""
        Evaluates the line at one or more :math:`t`-values

        Parameters
        ----------
        t: float or int or numpy.ndarray
            Either a single :math:`t`-value, a number of evenly spaced :math:`t`-values between 0 and 1, or
            a 1-D array of :math:`t`-values

        Returns
        -------
        numpy.ndarray
            If ``t`` is a :obj:`float`, the output is a 1-D array with three elements: the values of :math:`x`,
            :math:`y`, and :math:`x`. Otherwise, the output is an array of size :math:`\text{len}(t) \times 3`
        """
        pass

    @abstractmethod
    def dcdt(self, t: float or int or np.ndarray) -> float or np.ndarray:
        r"""
        Evaluates the first derivative of the curve with respect to :math:`t`

        Parameters
        ----------
        t: float or int or numpy.ndarray
            Either a single :math:`t`-value, a number of evenly spaced :math:`t`-values between 0 and 1, or
            a 1-D array of :math:`t`-values

        Returns
        -------
        numpy.ndarray
            If :math:`t` is a :obj:`float`, the output is a 1-D array containing two elements: the :math:`x`-
            :math:`y`, and :math:`z`-components of the first derivative. Otherwise, the output is a 2-D array of size
            :math:`\text{len}(t) \times 3`
        """
        pass

    @abstractmethod
    def d2cdt2(self, t: float or int or np.ndarray) -> float or np.ndarray:
        r"""
        Evaluates the second derivative of the curve with respect to :math:`t`

        Parameters
        ----------
        t: float or int or numpy.ndarray
            Either a single :math:`t`-value, a number of evenly spaced :math:`t`-values between 0 and 1, or
            a 1-D array of :math:`t`-values

        Returns
        -------
        numpy.ndarray
            If :math:`t` is a :obj:`float`, the output is a 1-D array containing three elements: the :math:`x`-
            :math:`y`-, and :math:`z`-components of the second derivative. Otherwise, the output is a 2-D array of size
            :math:`\text{len}(t) \times 3`
        """
        pass

    @abstractmethod
    def evaluate_pcurvedata(self, t: float or int or np.ndarray) -> PCurveData3D:
        r"""
        Evaluates a verbose set of parametric curve data as a class based on an input parameter value or vector

        Parameters
        ----------
        t: float or int or numpy.ndarray
            Either a single :math:`t`-value, a number of evenly spaced :math:`t`-values between 0 and 1, or
            a 1-D array of :math:`t`-values

        Returns
        -------
        PCurveData3D
            Parametric curve information, including derivative and curvature data
        """
        pass

    @staticmethod
    def _get_linear_tvec(nt: int) -> np.ndarray:
        """
        Gets a linear vector of parameter values based on a number of points

        Parameters
        ----------
        nt: int
            Number of linearly-spaced :math:`t`-values to generate
        """
        return np.linspace(0.0, 1.0, nt)

    @staticmethod
    def _validate_and_convert_t(t: float or int or np.ndarray) -> float or np.ndarray:
        """

        Parameters
        ----------
        t: float or int or numpy.ndarray
            Either a single :math:`t`-value, a number of evenly spaced :math:`t`-values between 0 and 1, or
            a 1-D array of :math:`t`-values

        Returns
        -------
        float or numpy.ndarray
            Either a single :math:`t`-value or a 1-D array of :math:`t`-values, depending on the input type
        """
        if isinstance(t, int) and t < 2:
            raise ValueError(f"If `t` is an integer, a value of at least 2 must be specified. If you are intending to "
                             f"evaluate the curve at t=0 or t=1, please use the float format for these numbers "
                             f"(0.0 or 1.0)")
        return np.linspace(0.0, 1.0, t) if isinstance(t, int) else t
    
    @abstractmethod
    def transform(self, **transformation_kwargs) -> "PCurve3D":
        """
        Creates a transformed copy of the curve by transforming the control points

        Parameters
        ----------
        transformation_kwargs
            Keyword arguments passed to :obj:`~aerocaps.geom.transformation.Transformation3D`

        Returns
        -------
        PCurve3D
            Transformed curve
        """

class Line2D(PCurve2D):
    """
    Two-dimensional line class
    """
    def __init__(self,
                 p0: Point2D,
                 p1: Point2D = None,
                 theta: Angle = None,
                 d: Length = Length(m=1.0),
                 name: str = "Line2D",
                 construction: bool = False
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
        name: str
            Name of the geometric object. May be re-assigned a unique name when added to a
            :obj:`~aerocaps.geom.geometry_container.GeometryContainer`. Default: 'Line2D'
        construction: bool
            Whether this is a geometry used only for construction of other geometries. If ``True``, this
            geometry will not be exported or plotted. Default: ``False``
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
        super().__init__(name=name, construction=construction)

    def evaluate(self, t: float or int or np.ndarray) -> np.ndarray:
        t = self._validate_and_convert_t(t)

        if self.theta:
            x = self.p0.x.m + self.d.m * np.cos(self.theta.rad) * t
            y = self.p0.y.m + self.d.m * np.sin(self.theta.rad) * t
        else:
            x = self.p0.x.m + t * (self.p1.x.m - self.p0.x.m)
            y = self.p0.y.m + t * (self.p1.y.m - self.p0.y.m)

        return np.column_stack((x, y)) if isinstance(t, np.ndarray) else np.array([x, y])

    def evaluate_point2d(self, t: float or int or np.ndarray) -> Point2D or typing.List[Point2D]:
        t = self._validate_and_convert_t(t)
        curve = self.evaluate(t)
        if curve.ndim == 1:
            return Point2D.from_array(curve)
        return [Point2D.from_array(curve_point) for curve_point in curve]

    def dcdt(self, t: float or int or np.ndarray) -> np.ndarray:
        t = self._validate_and_convert_t(t)
        if isinstance(t, float):
            if self.theta:
                return self.d.m * np.array([np.cos(self.theta.rad), np.sin(self.theta.rad)])
            return (self.p1 - self.p0).as_array()
        if self.theta:
            return self.d.m * np.repeat(
                np.array([np.cos(self.theta.rad), np.sin(self.theta.rad)])[np.newaxis, :], t.shape[0], axis=0)
        return np.repeat((self.p1 - self.p0).as_array()[np.newaxis, :], t.shape[0], axis=0)

    def d2cdt2(self, t: float or int or np.ndarray) -> np.ndarray:
        t = self._validate_and_convert_t(t)
        if isinstance(t, float):
            return np.zeros(2)
        return np.zeros((t.shape[0], 2))

    def evaluate_pcurvedata(self, t: float or int or np.ndarray) -> PCurveData2D:
        t = self._validate_and_convert_t(t)
        zeros = np.zeros(t.shape) if isinstance(t, np.ndarray) else 0.0
        ones = np.ones(t.shape) if isinstance(t, np.ndarray) else 1.0
        if self.theta:
            x = self.p0.x.m + self.d.m * np.cos(self.theta.rad) * t
            y = self.p0.y.m + self.d.m * np.sin(self.theta.rad) * t
            xp = self.d.m * np.cos(self.theta.rad) * ones
            yp = self.d.m * np.sin(self.theta.rad) * ones
        else:
            x = self.p0.x.m + t * (self.p1.x.m - self.p0.x.m)
            y = self.p0.y.m + t * (self.p1.y.m - self.p0.y.m)
            xp = (self.p1.x.m - self.p0.x.m) * ones
            yp = (self.p1.y.m - self.p0.y.m) * ones

        xpp = zeros
        ypp = zeros
        R = np.inf * ones
        k = zeros
        return PCurveData2D(t=t, x=x, y=y, xp=xp, yp=yp, xpp=xpp, ypp=ypp, k=k, R=R)

    def get_vector(self) -> Vector2D:
        r"""
        Gets a vector object determined by the starting and ending points of the line

        Returns
        -------
        Vector2D
            Vector object
        """
        return Vector2D(p0=self.p0, p1=self.p1)

    def plot(self, ax: plt.Axes, nt: int = 10, **kwargs):
        r"""
        Plots the line on a :obj:`matplotlib.pyplot.Axes`

        Parameters
        ----------
        ax: plt.Axes
            Axis on which to plot the line
        nt: int
            Number of points along the line to output to the plot. Default: ``10``
        kwargs
            Additional keyword arguments to pass to :obj:`matplotlib.pyplot.Axes.plot`
        """
        xy = self.evaluate(nt)
        ax.plot(xy[:, 0], xy[:, 1], **kwargs)


class Line3D(PCurve3D):
    """
    Three-dimensional line class
    """
    def __init__(self,
                 p0: Point3D,
                 p1: Point3D = None,
                 theta: Angle = None,
                 phi: Angle = None,
                 d: Length = Length(m=1.0),
                 name: str = "Line3D",
                 construction: bool = False
                 ):
        """

        Parameters
        ----------
        p0
        p1
        theta
        phi
        d
        name: str
            Name of the geometric object. May be re-assigned a unique name when added to a
            :obj:`~aerocaps.geom.geometry_container.GeometryContainer`. Default: 'Line3D'
        construction: bool
            Whether this is a geometry used only for construction of other geometries. If ``True``, this
            geometry will not be exported or plotted. Default: ``False``
        """
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
        super().__init__(name=name, construction=construction)

    def to_iges(self, *args, **kwargs) -> aerocaps.iges.entity.IGESEntity:
        return aerocaps.iges.curves.LineIGES(self.p0.as_array(), self.p1.as_array())

    def get_control_point_array(self, unit: str = "m") -> np.ndarray:
        r"""
        Gets an array representation of the two points defining the line. Used for compatibility with
        NURBS curve methods.

        Parameters
        ----------
        unit: str
            Physical length units used to create the array from the point objects

        Returns
        -------
        numpy.ndarray
            Array of size :math:`2 \times 3`
        """
        return np.array([p.as_array(unit=unit) for p in self.control_points])

    def reverse(self) -> "Line3D":
        """
        Creates a copy of the line with the parametric direction reversed

        Returns
        -------
        Line3D
            Reversed line
        """
        return self.__class__(p0=self.p1, p1=self.p0)

    def project_onto_principal_plane(self, plane: str = "XY") -> Line2D:
        r"""
        Projects the line onto a principal plane

        Parameters
        ----------
        plane: str
            Plane on which to project the line. Either 'XY', 'YZ', or 'XZ'

        Returns
        -------
        Line2D
            Projected line
        """
        return Line2D(p0=self.p0.projection_on_principal_plane(plane), p1=self.p1.projection_on_principal_plane(plane))

    def evaluate(self, t: float or int or np.ndarray) -> np.ndarray:
        t = self._validate_and_convert_t(t)

        if self.theta:
            x = self.p0.x.m + self.d.m * np.cos(self.phi.rad) * np.cos(self.theta.rad) * t
            y = self.p0.y.m + self.d.m * np.cos(self.phi.rad) * np.sin(self.theta.rad) * t
            z = self.p0.z.m + self.d.m * np.sin(self.phi.rad) * t
        else:
            x = self.p0.x.m + t * (self.p1.x.m - self.p0.x.m)
            y = self.p0.y.m + t * (self.p1.y.m - self.p0.y.m)
            z = self.p0.z.m + t * (self.p1.z.m - self.p0.z.m)

        return np.column_stack((x, y, z)) if isinstance(t, np.ndarray) else np.array([x, y, z])

    def evaluate_point3d(self, t: float or int or np.ndarray) -> Point3D or typing.List[Point3D]:
        t = self._validate_and_convert_t(t)

        curve = self.evaluate(t)
        if curve.ndim == 1:
            return Point3D.from_array(curve)
        return [Point3D.from_array(curve_point) for curve_point in curve]

    def dcdt(self, t: float or int or np.ndarray) -> np.ndarray:
        t = self._validate_and_convert_t(t)
        if isinstance(t, float):
            if self.theta:
                return self.d.m * np.array([
                    np.cos(self.phi.rad) * np.cos(self.theta.rad),
                    np.cos(self.phi.rad) * np.sin(self.theta.rad),
                    np.sin(self.phi.rad)
                ])
            return (self.p1 - self.p0).as_array()
        if self.theta:
            return self.d.m * np.repeat(
                np.array([
                    np.cos(self.phi.rad) * np.cos(self.theta.rad),
                    np.cos(self.phi.rad) * np.sin(self.theta.rad),
                    np.sin(self.phi.rad)
                ])[np.newaxis, :], t.shape[0], axis=0)
        return np.repeat((self.p1 - self.p0).as_array()[np.newaxis, :], t.shape[0], axis=0)

    def d2cdt2(self, t: float or int or np.ndarray) -> np.ndarray:
        t = self._validate_and_convert_t(t)
        if isinstance(t, float):
            return np.zeros(3)
        return np.zeros((t.shape[0], 3))

    def evaluate_pcurvedata(self, t: float or int or np.ndarray) -> PCurveData3D:
        t = self._validate_and_convert_t(t)
        zeros = np.zeros(t.shape) if isinstance(t, np.ndarray) else 0.0
        ones = np.ones(t.shape) if isinstance(t, np.ndarray) else 1.0

        if self.theta:
            x = self.p0.x + self.d * np.cos(self.phi.rad) * np.cos(self.theta.rad) * t,
            y = self.p0.y + self.d * np.cos(self.phi.rad) * np.sin(self.theta.rad) * t,
            z = self.p0.z + self.d * np.sin(self.phi.rad) * t
            xp = self.d.m * np.cos(self.phi.rad) * np.cos(self.theta.rad) * ones
            yp = self.d.m * np.cos(self.phi.rad) * np.sin(self.theta.rad) * ones
            zp = self.d.m * np.sin(self.phi.rad) * ones
        else:
            x = self.p0.x.m + t * (self.p1.x.m - self.p0.x.m)
            y = self.p0.y.m + t * (self.p1.y.m - self.p0.y.m)
            z = self.p0.z.m + t * (self.p1.z.m - self.p0.z.m)
            xp = (self.p1.x.m - self.p0.x.m) * ones
            yp = (self.p1.y.m - self.p0.y.m) * ones
            zp = (self.p1.z.m - self.p0.z.m) * ones

        xpp = zeros
        ypp = zeros
        zpp = zeros
        R = np.inf * ones
        k = zeros
        return PCurveData3D(t=t, x=x, y=y, z=z, xp=xp, yp=yp, zp=zp, xpp=xpp, ypp=ypp, zpp=zpp, k=k, R=R)

    def get_vector(self) -> Vector3D:
        r"""
        Gets a vector object determined by the starting and ending points of the line

        Returns
        -------
        Vector3D
            Vector object
        """
        return Vector3D(p0=self.p0, p1=self.p1)

    def transform(self, **transformation_kwargs) -> "Line3D":
        """
        Creates a transformed copy of the curve by transforming the start and end points

        Parameters
        ----------
        transformation_kwargs
            Keyword arguments passed to :obj:`~aerocaps.geom.transformation.Transformation3D`

        Returns
        -------
        Line3D
            Transformed line
        """
        transformation = Transformation3D(**transformation_kwargs)
        new_points = transformation.transform(self.get_control_point_array())
        return Line3D(
            p0=Point3D.from_array(new_points[0]),
            p1=Point3D.from_array(new_points[1]),
            name=self.name, 
            construction=self.construction
        )

    def plot(self, plot: pv.Plotter = None, ax: plt.Axes = None, nt: int = 10, **kwargs):
        r"""
        Plots the line on either a 3-D :obj:`matplotlib.pyplot.Axes`

        Parameters
        ----------
        plot: pyvista.Plotter
            3-D :obj:`pyvista` plotting window
        ax: plt.Axes
            Axis on which to plot the line
        nt: int
            Number of points along the line to output to the plot. Default: ``10``
        kwargs
            Additional keyword arguments to pass to :obj:`matplotlib.pyplot.Axes.plot` or
            :obj:`pyvista.Plotter.add_lines`
        """
        if not bool(plot) ^ bool(ax):
            raise ValueError("Either `plot` or `ax` must be specified")
        if plot:
            line_arr = np.array([self.p0.as_array(), self.p1.as_array()])
            plot.add_lines(line_arr, **kwargs)
        if ax:
            xy = self.evaluate(nt)
            ax.plot(xy[:, 0], xy[:, 1], **kwargs)


class CircularArc2D(PCurve2D):
    """Two-dimensional circular arc class"""
    def __init__(self, center: Point2D, radius: Length, start_point: Point2D = None, end_point: Point2D = None,
                 start_angle: Angle = None, end_angle: Angle = None, complement: bool = False,
                 name: str = "CircularArc2D", construction: bool = False):
        """
        Creates a circular arc object.

        .. note::

            The center and radius must be specified, along with a combination of either ``start_point`` or
            ``start_angle`` and ``end_point`` or ``end_angle``. The starting and ending points are only used
            to determine the angle based on relationship to the center. These points do not override the value
            specified by ``radius``

        Parameters
        ----------
        center: Point2D
            Center of the arc
        radius: Length
            Arc radius
        start_point: Point2D
            Optional starting point for the arc. Default: ``None``
        end_point: Point2D
            Optional ending point for the arc. Default: ``None``
        start_angle: Angle
            Optional starting angle for the arc. Default: ``None``
        end_angle: Angle
            Optional ending angle for the arc. Default: ``None``
        complement: bool
            Whether to output the complement arc. Default: ``False``
        name: str
            Name of the geometric object. May be re-assigned a unique name when added to a
            :obj:`~aerocaps.geom.geometry_container.GeometryContainer`. Default: 'CircularArc2D'
        construction: bool
            Whether this is a geometry used only for construction of other geometries. If ``True``, this
            geometry will not be exported or plotted. Default: ``False``
        """
        if not bool(start_point) ^ bool(start_angle):
            raise ValueError("Must specify a starting point or angle")
        if not bool(end_point) ^ bool(end_angle):
            raise ValueError("Must specify an ennding point or angle")
        self.center = center
        self.radius = radius
        if start_angle is None:
            self.start_angle = Angle(rad=np.arctan2(start_point.y.m - center.y.m, start_point.x.m - center.x.m))
        else:
            self.start_angle = start_angle
        if end_angle is None:
            self.end_angle = Angle(rad=np.arctan2(end_point.y.m - center.y.m, end_point.x.m - center.x.m))
        else:
            self.end_angle = end_angle
        self.complement = complement
        super().__init__(name=name, construction=construction)

    def _map_t_to_angle(self, t: float or np.ndarray) -> float or np.ndarray:
        r"""
        Maps the :math:`t`-value to an angle along the circle

        Parameters
        ----------
        t: float or numpy.ndarray
            :math:`t`-value or array of :math:`t`-values to map

        Returns
        -------
        float or numpy.ndarray
            Mapped angles
        """
        if self.complement:
            return self.start_angle.rad - (2 * np.pi - (self.end_angle.rad - self.start_angle.rad)) * t
        else:
            return (self.end_angle.rad - self.start_angle.rad) * t + self.start_angle.rad

    def evaluate(self, t: float or int or np.ndarray) -> np.ndarray:
        t = self._validate_and_convert_t(t)

        x = self.center.x.m + self.radius.m * np.cos(self._map_t_to_angle(t))
        y = self.center.y.m + self.radius.m * np.sin(self._map_t_to_angle(t))

        return np.column_stack((x, y)) if isinstance(t, np.ndarray) else np.array([x, y])

    def evaluate_point2d(self, t: float or int or np.ndarray) -> Point2D or typing.List[Point2D]:
        t = self._validate_and_convert_t(t)
        curve = self.evaluate(t)
        if curve.ndim == 1:
            return Point2D.from_array(curve)
        return [Point2D.from_array(curve_point) for curve_point in curve]

    def dcdt(self, t: float or int or np.ndarray) -> np.ndarray:
        t = self._validate_and_convert_t(t)
        angle_range = self.end_angle.rad - self.start_angle.rad

        xp = -self.radius.m * angle_range * np.sin(self._map_t_to_angle(t))
        yp = self.radius.m * angle_range * np.cos(self._map_t_to_angle(t))
        if isinstance(t, float):
            return np.array([xp, yp])
        return np.column_stack((xp, yp))

    def d2cdt2(self, t: float or int or np.ndarray) -> np.ndarray:
        t = self._validate_and_convert_t(t)
        angle_range = self.end_angle.rad - self.start_angle.rad

        xpp = -self.radius.m * angle_range ** 2 * np.cos(self._map_t_to_angle(t))
        ypp = -self.radius.m * angle_range ** 2 * np.sin(self._map_t_to_angle(t))
        if isinstance(t, float):
            return np.array([xpp, ypp])
        return np.column_stack((xpp, ypp))

    def evaluate_pcurvedata(self, t: float or int or np.ndarray) -> PCurveData2D:
        t = self._validate_and_convert_t(t)
        ones = np.ones(t.shape) if isinstance(t, np.ndarray) else 1.0
        angle_range = self.end_angle.rad - self.start_angle.rad

        x = self.center.x.m + self.radius.m * np.cos(self._map_t_to_angle(t))
        y = self.center.y.m + self.radius.m * np.sin(self._map_t_to_angle(t))
        xp = -self.radius.m * angle_range * np.sin(self._map_t_to_angle(t))
        yp = self.radius.m * angle_range * np.cos(self._map_t_to_angle(t))
        xpp = -self.radius.m * angle_range ** 2 * np.cos(self._map_t_to_angle(t))
        ypp = -self.radius.m * angle_range ** 2 * np.sin(self._map_t_to_angle(t))
        R = self.radius.m * ones
        k = 1 / self.radius.m * ones
        return PCurveData2D(t=t, x=x, y=y, xp=xp, yp=yp, xpp=xpp, ypp=ypp, k=k, R=R)

    def plot(self, ax: plt.Axes, nt: int = 100, **kwargs):
        r"""
        Plots the line on a :obj:`matplotlib.pyplot.Axes`

        Parameters
        ----------
        ax: plt.Axes
            Axis on which to plot the arc
        nt: int
            Number of points along the arc to output to the plot. Default: ``100``
        kwargs
            Additional keyword arguments to pass to :obj:`matplotlib.pyplot.Axes.plot`
        """
        xy = self.evaluate(nt)
        ax.plot(xy[:, 0], xy[:, 1], **kwargs)


class BezierCurve2D(PCurve2D):
    """Two-dimensional Bézier curve class"""
    def __init__(self, control_points: typing.List[Point2D] or np.ndarray,
                 name: str = "BezierCurve2D", construction: bool = False):
        """
        Creates a two-dimensional Bézier curve objects from a list of control points

        Parameters
        ----------
        control_points: typing.List[Point2D] or numpy.ndarray
            Control points for the Bézier curve
        name: str
            Name of the geometric object. May be re-assigned a unique name when added to a
            :obj:`~aerocaps.geom.geometry_container.GeometryContainer`. Default: 'BezierCurve2D'
        construction: bool
            Whether this is a geometry used only for construction of other geometries. If ``True``, this
            geometry will not be exported or plotted. Default: ``False``
        """
        self.control_points = [Point2D.from_array(p) for p in control_points] if isinstance(
            control_points, np.ndarray) else control_points
        super().__init__(name=name, construction=construction)

    @property
    def degree(self) -> int:
        """Curve degree"""
        return len(self.control_points) - 1

    @degree.setter
    def degree(self, value):
        raise AttributeError("The 'degree' property is read-only. Use the Bezier2D.elevate_degree method to increase"
                             "the degree of the curve while retaining the shape, or manually add or remove control "
                             "points to change the degree directly.")

    def get_control_point_array(self, unit: str = "m") -> np.ndarray:
        r"""
        Gets an array representation of the control points

        Parameters
        ----------
        unit: str
            Physical length unit used to determine the output array. Default: ``"m"``

        Returns
        -------
        numpy.ndarray
            Array of size :math:`(n+1)\times 2` where :math:`n` is the curve degree
        """
        return np.array([p.as_array(unit=unit) for p in self.control_points])

    def evaluate(self, t: float or int or np.ndarray) -> np.ndarray:
        P = self.get_control_point_array()
        if isinstance(t, float):
            return np.array(bezier_curve_eval(P, t))
        if isinstance(t, int):
            return np.array(bezier_curve_eval_grid(P, t))
        return np.array(bezier_curve_eval_tvec(P, t))

    def evaluate_point2d(self, t: float or int or np.ndarray) -> Point2D or typing.List[Point2D]:
        t = self._validate_and_convert_t(t)
        curve = self.evaluate(t)
        if curve.ndim == 1:
            return Point2D.from_array(curve)
        return [Point2D.from_array(curve_point) for curve_point in curve]

    def dcdt(self, t: float or int or np.ndarray) -> np.ndarray:
        P = self.get_control_point_array()
        if isinstance(t, float):
            return np.array(bezier_curve_dcdt(P, t))
        if isinstance(t, int):
            return np.array(bezier_curve_dcdt_grid(P, t))
        return np.array(bezier_curve_dcdt_tvec(P, t))

    def d2cdt2(self, t: float or int or np.ndarray) -> np.ndarray:
        P = self.get_control_point_array()
        if isinstance(t, float):
            return np.array(bezier_curve_d2cdt2(P, t))
        if isinstance(t, int):
            return np.array(bezier_curve_d2cdt2_grid(P, t))
        return np.array(bezier_curve_d2cdt2_tvec(P, t))

    def evaluate_pcurvedata(self, t: float or int or np.ndarray) -> PCurveData2D:
        xy = self.evaluate(t)
        xpyp = self.dcdt(t)
        xppypp = self.d2cdt2(t)
        return PCurveData2D(
            t=t, x=xy[:, 0], y=xy[:, 1], xp=xpyp[:, 0], yp=xpyp[:, 1], xpp=xppypp[:, 0], ypp=xppypp[:, 1]
        )

    def compute_t_corresponding_to_x(self, x_seek: float, t0: float = 0.5) -> float:
        r"""
        Computes the :math:`t`-value corresponding to a given :math:`x`-value

        Parameters
        ----------
        x_seek: float
            :math:`x`-value
        t0: float
            Initial guess for the output :math:`t`-value. Default: ``0.5``

        Returns
        -------
        float
            :math:`t`-value corresponding to ``x_seek``
        """
        def bez_root_find_func(t):
            point = self.evaluate_point2d(t[0])
            return np.array([point.x.m - x_seek])

        return fsolve(bez_root_find_func, x0=np.array([t0]))[0]

    def compute_t_corresponding_to_y(self, y_seek: float, t0: float = 0.5) -> float:
        r"""
        Computes the :math:`t`-value corresponding to a given :math:`y`-value

        Parameters
        ----------
        y_seek: float
            :math:`y`-value
        t0: float
            Initial guess for the output :math:`t`-value. Default: ``0.5``

        Returns
        -------
        float
            :math:`t`-value corresponding to ``y_seek``
        """
        def bez_root_find_func(t):
            point = self.evaluate_point2d(t[0])
            return np.array([point.y.m - y_seek])

        return fsolve(bez_root_find_func, x0=np.array([t0]))[0]

    def convert_to_3d(self, plane: str = "XY") -> "BezierCurve3D":
        """
        Converts the 2-D Bézier curve to a 3-D Bézier curve by mapping it onto a principal plane. This is done
        by inserting a column of zeros.

        Parameters
        ----------
        plane: str
            Principal plane, one of 'XY', 'YZ', or 'XZ'

        Returns
        -------
        BezierCurve3D
            Planar 3-D Bézier curve
        """
        valid_planes = ["XY", "YZ", "XZ"]
        plane_axis_mapping = {k: v for k, v in zip(valid_planes, [2, 0, 1])}
        if plane not in valid_planes:
            raise ValueError(f"Plane must be one of {valid_planes}. Given plane was {plane}")
        P = self.get_control_point_array()
        new_P = np.insert(P, plane_axis_mapping[plane], 0.0, axis=1)
        return BezierCurve3D(new_P)

    def transform(self, **transformation_kwargs) -> "BezierCurve2D":
        """
        Creates a transformed copy of the curve by transforming each of the control points

        Parameters
        ----------
        transformation_kwargs
            Keyword arguments passed to :obj:`~aerocaps.geom.transformation.Transformation2D`

        Returns
        -------
        BezierCurve2D
            Transformed curve
        """
        transformation = Transformation2D(**transformation_kwargs)
        return BezierCurve2D(
            transformation.transform(self.get_control_point_array()),
            name=self.name, 
            construction=self.construction
        )

    def elevate_degree(self) -> "BezierCurve2D":
        """
        Elevates the degree of the Bézier curve. See algorithm source
        `here <https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/Bezier/bezier-elev.html>`_.

        Returns
        -------
        BezierCurve2D
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

        return BezierCurve2D(new_control_points)

    def split(self, t_split: float) -> ("BezierCurve2D", "BezierCurve2D"):
        r"""
        Splits the curve into two curves at a given :math:`t`-value by applying the de-Casteljau algorithm

        Parameters
        ----------
        t_split: float
            :math:`t`-value at which to split the curve

        Returns
        -------
        Bezier2D, Bezier2D
            Two new curves split at the input :math:`t`-value
        """
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
            BezierCurve2D(bez_1_points),
            BezierCurve2D(bez_2_points)
        )


class BezierCurve3D(PCurve3D):
    """Three-dimensional Bézier curve class"""
    def __init__(self, control_points: typing.List[Point3D] or np.ndarray,
                 name: str = "BezierCurve3D", construction: bool = False):
        """
        Creates a three-dimensional Bézier curve objects from a list of control points

        Parameters
        ----------
        control_points: typing.List[Point3D] or numpy.ndarray
            Control points for the Bézier curve
        name: str
            Name of the geometric object. May be re-assigned a unique name when added to a
            :obj:`~aerocaps.geom.geometry_container.GeometryContainer`. Default: 'BezierCurve3D'
        construction: bool
            Whether this is a geometry used only for construction of other geometries. If ``True``, this
            geometry will not be exported or plotted. Default: ``False``
        """
        self.control_points = [Point3D.from_array(p) for p in control_points] if isinstance(
            control_points, np.ndarray) else control_points
        super().__init__(name=name, construction=construction)

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

    def reverse(self) -> "BezierCurve3D":
        """
        Creates a copy of the curve with the parametric direction reversed. This is done by reversing the order
        of the control points

        Returns
        -------
        BezierCurve3D
            Reversed curve
        """
        return self.__class__(self.control_points[::-1])

    def to_rational_bezier_curve(self) -> "RationalBezierCurve3D":
        """
        Converts the curve to a rational Bézier curve by setting all weights to unity

        Returns
        -------
        RationalBezierCurve3D
            Rational version of the curve
        """
        return RationalBezierCurve3D(self.control_points, np.ones(self.degree + 1))

    def project_onto_principal_plane(self, plane: str = "XY") -> BezierCurve2D:
        r"""
        Projects the curve onto a principal plane

        Parameters
        ----------
        plane: str
            Plane on which to project the line. Either 'XY', 'YZ', or 'XZ'

        Returns
        -------
        BezierCurve2D
            Projected curve
        """
        return BezierCurve2D(control_points=[pt.projection_on_principal_plane(plane) for pt in self.control_points])

    def get_control_point_array(self, unit: str = "m") -> np.ndarray:
        r"""
        Gets an array representation of the control points

        Parameters
        ----------
        unit: str
            Physical length unit used to determine the output array. Default: ``"m"``

        Returns
        -------
        numpy.ndarray
            Array of size :math:`(n+1)\times 3` where :math:`n` is the curve degree
        """
        return np.array([p.as_array(unit=unit) for p in self.control_points])

    def evaluate(self, t: float or int or np.ndarray) -> np.ndarray:
        P = self.get_control_point_array()
        if isinstance(t, float):
            return np.array(bezier_curve_eval(P, t))
        if isinstance(t, int):
            return np.array(bezier_curve_eval_grid(P, t))
        return np.array(bezier_curve_eval_tvec(P, t))

    def evaluate_point3d(self, t: float or int or np.ndarray) -> Point3D or typing.List[Point3D]:
        t = self._validate_and_convert_t(t)
        curve = self.evaluate(t)
        if curve.ndim == 1:
            return Point3D.from_array(curve)
        return [Point3D.from_array(curve_point) for curve_point in curve]

    def dcdt(self, t: float or int or np.ndarray) -> np.ndarray:
        P = self.get_control_point_array()
        if isinstance(t, float):
            return np.array(bezier_curve_dcdt(P, t))
        if isinstance(t, int):
            return np.array(bezier_curve_dcdt_grid(P, t))
        return np.array(bezier_curve_dcdt_tvec(P, t))

    def d2cdt2(self, t: float or int or np.ndarray) -> np.ndarray:
        P = self.get_control_point_array()
        if isinstance(t, float):
            return np.array(bezier_curve_d2cdt2(P, t))
        if isinstance(t, int):
            return np.array(bezier_curve_d2cdt2_grid(P, t))
        return np.array(bezier_curve_d2cdt2_tvec(P, t))

    def evaluate_pcurvedata(self, t: float or int or np.ndarray) -> PCurveData3D:
        xyz = self.evaluate(t)
        xpypzp = self.dcdt(t)
        xppyppzpp = self.d2cdt2(t)
        return PCurveData3D(
            t=t,
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
            xp=xpypzp[:, 0], yp=xpypzp[:, 1], zp=xpypzp[:, 2],
            xpp=xppyppzpp[:, 0], ypp=xppyppzpp[:, 1], zpp=xppyppzpp[:, 2]
        )

    def compute_t_corresponding_to_x(self, x_seek: float, t0: float = 0.5):
        r"""
        Computes the :math:`t`-value corresponding to a given :math:`x`-value

        Parameters
        ----------
        x_seek: float
            :math:`x`-value
        t0: float
            Initial guess for the output :math:`t`-value. Default: ``0.5``

        Returns
        -------
        float
            :math:`t`-value corresponding to ``x_seek``
        """
        def bez_root_find_func(t):
            point = self.evaluate_point3d(t[0])
            return np.array([point.x.m - x_seek])

        return fsolve(bez_root_find_func, x0=np.array([t0]))[0]

    def compute_t_corresponding_to_y(self, y_seek: float, t0: float = 0.5):
        r"""
        Computes the :math:`t`-value corresponding to a given :math:`y`-value

        Parameters
        ----------
        y_seek: float
            :math:`y`-value
        t0: float
            Initial guess for the output :math:`t`-value. Default: ``0.5``

        Returns
        -------
        float
            :math:`t`-value corresponding to ``y_seek``
        """
        def bez_root_find_func(t):
            point = self.evaluate_point3d(t[0])
            return np.array([point.y.m - y_seek])

        return fsolve(bez_root_find_func, x0=np.array([t0]))[0]

    def compute_t_corresponding_to_z(self, z_seek: float, t0: float = 0.5):
        r"""
        Computes the :math:`t`-value corresponding to a given :math:`z`-value

        Parameters
        ----------
        z_seek: float
            :math:`z`-value
        t0: float
            Initial guess for the output :math:`t`-value. Default: ``0.5``

        Returns
        -------
        float
            :math:`t`-value corresponding to ``z_seek``
        """
        def bez_root_find_func(t):
            point = self.evaluate_point3d(t[0])
            return np.array([point.z.m - z_seek])

        return fsolve(bez_root_find_func, x0=np.array([t0]))[0]

    def transform(self, **transformation_kwargs) -> "BezierCurve3D":
        """
        Creates a transformed copy of the curve by transforming each of the control points

        Parameters
        ----------
        transformation_kwargs
            Keyword arguments passed to :obj:`~aerocaps.geom.transformation.Transformation3D`

        Returns
        -------
        BezierCurve3D
            Transformed curve
        """
        transformation = Transformation3D(**transformation_kwargs)
        return BezierCurve3D(
            transformation.transform(self.get_control_point_array()),
            name=self.name,
            construction=self.construction
        )

    def elevate_degree(self) -> "BezierCurve3D":
        """
        Elevates the degree of the Bézier curve. See algorithm source
        `here <https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/Bezier/bezier-elev.html>`_.

        Returns
        -------
        BezierCurve3D
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

        return BezierCurve3D(new_control_points)

    def split(self, t_split: float) -> ("BezierCurve3D", "BezierCurve3D"):
        r"""
        Splits the curve into two curves at a given :math:`t`-value by applying the de-Casteljau algorithm

        Parameters
        ----------
        t_split: float
            :math:`t`-value at which to split the curve

        Returns
        -------
        Bezier3D, Bezier3D
            Two new curves split at the input :math:`t`-value
        """
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
            BezierCurve3D(bez_1_points),
            BezierCurve3D(bez_2_points)
        )

    def plot(self, ax: plt.Axes or pv.Plotter, projection: str = None, nt: int = 201, **plt_kwargs):
        """
        Plots the curve on a :obj:`matplotlib.pyplot.Axes` or a `pyvista.Plotter` window

        Parameters
        ----------
        ax: plt.Axes or pv.Plotter
            Axes/window on which to plot
        projection: str
            Projection on which to plot (either 'XY', 'YZ', 'XZ', or 'XYZ' for a 3-D plot). Only used if
            ``ax`` is a ``plt.Axes``. Defaults to 'XYZ' if not specified. Default: ``None``
        nt: int
            Number of evenly-spaced parameter values to plot. Default: ``201``
        plt_kwargs
            Additional keyword arguments to pass to :obj:`matplotlib.pyplot.Axes.plot` or
            :obj:`pyvista.Plotter.add_lines`
        """
        projection = "XYZ" if projection is None else projection
        t_vec = np.linspace(0.0, 1.0, nt)
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


class RationalBezierCurve2D(PCurve2D):
    """Two-dimensional rational Bézier curve class"""
    def __init__(self, control_points: typing.List[Point2D] or np.ndarray, weights: np.ndarray,
                 name: str = "RationalBezierCurve3D", construction: bool = False):
        """
        Creates a two-dimensional rational Bézier curve objects from a list of control points and weights

        Parameters
        ----------
        control_points: typing.List[Point2D] or numpy.ndarray
            Control points for the rational Bézier curve
        weights: numpy.ndarray
            Weights for the control points. Must have the same length as the control point array
        name: str
            Name of the geometric object. May be re-assigned a unique name when added to a
            :obj:`~aerocaps.geom.geometry_container.GeometryContainer`. Default: 'RationalBezierCurve2D'
        construction: bool
            Whether this is a geometry used only for construction of other geometries. If ``True``, this
            geometry will not be exported or plotted. Default: ``False``
        """
        self.control_points = [Point2D.from_array(p) for p in control_points] if isinstance(
            control_points, np.ndarray) else control_points
        assert weights.ndim == 1
        assert len(control_points) == len(weights)

        # Negative weight check
        for weight in weights:
            if weight < 0:
                raise NegativeWeightError("All weights must be non-negative")

        self.dim = 2
        self.weights = np.array(weights)
        self.knot_vector = np.zeros(2 * len(control_points))
        self.knot_vector[len(control_points):] = 1.0
        self.degree = len(control_points) - 1
        assert self.knot_vector.ndim == 1
        assert len(self.knot_vector) == len(control_points) + self.degree + 1
        super().__init__(name=name, construction=construction)

    def reverse(self) -> "RationalBezierCurve2D":
        return self.__class__(self.control_points[::-1],
                              self.weights[::-1])

    def elevate_degree(self) -> "RationalBezierCurve2D":
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
        new_control_points = new_homogeneous_control_points[:, :-1] / np.repeat(new_weights[:, np.newaxis],
                                                                                self.dim, axis=1)

        return RationalBezierCurve2D(new_control_points, new_weights)

    def get_control_point_array(self, unit: str = "m") -> np.ndarray:
        r"""
        Gets an array representation of the control points

        Parameters
        ----------
        unit: str
            Physical length unit used to determine the output array. Default: ``"m"``

        Returns
        -------
        numpy.ndarray
            Array of size :math:`(n+1)\times 2` where :math:`n` is the curve degree
        """
        return np.array([p.as_array(unit=unit) for p in self.control_points])

    def get_homogeneous_control_points(self) -> np.ndarray:
        r"""
        Gets the array of control points in homogeneous coordinates, :math:`\mathbf{P}_i \cdot w_i`

        Returns
        -------
        numpy.ndarray
            Array of size :math:`(n + 1) \times 3`, where :math:`n` is the curve degree. The three columns, in order,
            represent the :math:`x`-coordinate, :math:`y`-coordinate, and weight of each
            control point.
        """
        return np.column_stack((
            self.get_control_point_array() * np.repeat(self.weights[:, np.newaxis], self.dim, axis=1),
            self.weights
        ))

    @classmethod
    def generate_from_array(cls, P: np.ndarray, weights: np.ndarray):
        return cls([Point2D(x=Length(m=xy[0]), y=Length(m=xy[1])) for xy in P], weights)

    def evaluate(self, t: float or int or np.ndarray) -> np.ndarray:
        P = self.get_control_point_array()
        w = self.weights
        if isinstance(t, float):
            return np.array(rational_bezier_curve_eval(P, w, t))
        if isinstance(t, int):
            return np.array(rational_bezier_curve_eval_grid(P, w, t))
        return np.array(rational_bezier_curve_eval_tvec(P, w, t))

    def evaluate_point2d(self, t: float or int or np.ndarray) -> Point2D or typing.List[Point2D]:
        curve = self.evaluate(t)
        if curve.ndim == 1:
            return Point2D.from_array(curve)
        return [Point2D.from_array(curve_point) for curve_point in curve]

    def dcdt(self, t: float or int or np.ndarray) -> np.ndarray:
        P = self.get_control_point_array()
        w = self.weights
        if isinstance(t, float):
            return np.array(rational_bezier_curve_dcdt(P, w, t))
        if isinstance(t, int):
            return np.array(rational_bezier_curve_dcdt_grid(P, w, t))
        return np.array(rational_bezier_curve_dcdt_tvec(P, w, t))

    def d2cdt2(self, t: float or int or np.ndarray) -> np.ndarray:
        P = self.get_control_point_array()
        w = self.weights
        if isinstance(t, float):
            return np.array(rational_bezier_curve_d2cdt2(P, w, t))
        if isinstance(t, int):
            return np.array(rational_bezier_curve_d2cdt2_grid(P, w, t))
        return np.array(rational_bezier_curve_d2cdt2_tvec(P, w, t))

    def evaluate_pcurvedata(self, t: float or int or np.ndarray) -> PCurveData2D:
        xy = self.evaluate(t)
        xpyp = self.dcdt(t)
        xppypp = self.d2cdt2(t)
        return PCurveData2D(
            t=t,
            x=xy[:, 0], y=xy[:, 1],
            xp=xpyp[:, 0], yp=xpyp[:, 1],
            xpp=xppypp[:, 0], ypp=xppypp[:, 1]
        )

    def compute_t_corresponding_to_x(self, x_seek: float, t0: float = 0.5):
        r"""
        Computes the :math:`t`-value corresponding to a given :math:`x`-value

        Parameters
        ----------
        x_seek: float
            :math:`x`-value
        t0: float
            Initial guess for the output :math:`t`-value. Default: ``0.5``

        Returns
        -------
        float
            :math:`t`-value corresponding to ``x_seek``
        """

        def bez_root_find_func(t):
            point = self.evaluate_point2d(t[0])
            return np.array([point.x.m - x_seek])

        return fsolve(bez_root_find_func, x0=np.array([t0]))[0]

    def compute_t_corresponding_to_y(self, y_seek: float, t0: float = 0.5):
        r"""
        Computes the :math:`t`-value corresponding to a given :math:`y`-value

        Parameters
        ----------
        y_seek: float
            :math:`y`-value
        t0: float
            Initial guess for the output :math:`t`-value. Default: ``0.5``

        Returns
        -------
        float
            :math:`t`-value corresponding to ``y_seek``
        """

        def bez_root_find_func(t):
            point = self.evaluate_point2d(t[0])
            return np.array([point.y.m - y_seek])

        return fsolve(bez_root_find_func, x0=np.array([t0]))[0]

    def transform(self, **transformation_kwargs) -> "RationalBezierCurve2D":
        """
        Creates a transformed copy of the curve by transforming each of the control points

        Parameters
        ----------
        transformation_kwargs
            Keyword arguments passed to :obj:`~aerocaps.geom.transformation.Transformation2D`

        Returns
        -------
        RationalBezierCurve2D
            Transformed curve
        """
        transformation = Transformation2D(**transformation_kwargs)
        return RationalBezierCurve2D(
            transformation.transform(self.get_control_point_array()),
            weights=deepcopy(self.weights), 
            name=self.name, 
            construction=self.construction
        )

    def plot(self, ax: plt.Axes or pv.Plotter, projection: str = None, nt: int = 201, **plt_kwargs):
        """
        Plots the curve on a :obj:`matplotlib.pyplot.Axes` or a `pyvista.Plotter` window

        Parameters
        ----------
        ax: plt.Axes or pv.Plotter
            Axes/window on which to plot
        projection: str
            Projection on which to plot (either 'XY', 'YZ', 'XZ', or 'XYZ' for a 3-D plot). Only used if
            ``ax`` is a ``plt.Axes``. Defaults to 'XYZ' if not specified. Default: ``None``
        nt: int
            Number of evenly-spaced parameter values to plot. Default: ``201``
        plt_kwargs
            Additional keyword arguments to pass to :obj:`matplotlib.pyplot.Axes.plot` or
            :obj:`pyvista.Plotter.add_lines`
        """
        projection = "XYZ" if projection is None else projection
        t_vec = np.linspace(0.0, 1.0, nt)
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
        """
        Plots the control points on a :obj:`matplotlib.pyplot.Axes`

        Parameters
        ----------
        ax: plt.Axes or pv.Plotter
            Axes/window on which to plot
        projection: str
            Projection on which to plot (either 'XY', 'YZ', 'XZ', or 'XYZ' for a 3-D plot). Only used if
            ``ax`` is a ``plt.Axes``. Defaults to 'XYZ' if not specified. Default: ``None``
        plt_kwargs
            Additional keyword arguments to pass to :obj:`matplotlib.pyplot.Axes.plot` or
            :obj:`pyvista.Plotter.add_lines`
        """
        projection = "XYZ" if projection is None else projection
        cps = self.get_control_point_array()
        args = tuple([cps[:, _projection_dict[axis]] for axis in projection])
        ax.plot(*args, **plt_kwargs)

    def enforce_g0(self, other: "RationalBezierCurve2D"):
        other.control_points[0] = self.control_points[-1]

    def enforce_c0(self, other: "RationalBezierCurve2D"):
        self.enforce_g0(other)

    def enforce_g0g1(self, other: "RationalBezierCurve2D", f: float):
        self.enforce_g0(other)
        n_ratio = self.degree / other.degree
        w_ratio_a = self.weights[-2] / self.weights[-1]
        w_ratio_b = other.weights[0] / other.weights[1]
        other.control_points[1] = other.control_points[0] + f * n_ratio * w_ratio_a * w_ratio_b * (self.control_points[-1] - self.control_points[-2])

    def enforce_c0c1(self, other: "RationalBezierCurve2D"):
        self.enforce_g0g1(other, f=1.0)

    def enforce_g0g1g2(self, other: "RationalBezierCurve2D", f: float):
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

    def enforce_c0c1c2(self, other: "RationalBezierCurve2D"):
        self.enforce_g0g1g2(other, f=1.0)


class RationalBezierCurve3D(PCurve3D):
    """Three-dimensional rational Bézier curve class"""
    def __init__(self, control_points: typing.List[Point3D] or np.ndarray, weights: np.ndarray,
                 name: str = "RationalBezierCurve3D", construction: bool = False):
        """
        Creates a three-dimensional rational Bézier curve objects from a list of control points and weights

        Parameters
        ----------
        control_points: typing.List[Point3D] or numpy.ndarray
            Control points for the Bézier curve
        weights: numpy.ndarray
            Weights for the control points. Must have the same length as the control point array
        name: str
            Name of the geometric object. May be re-assigned a unique name when added to a
            :obj:`~aerocaps.geom.geometry_container.GeometryContainer`. Default: 'RationalBezierCurve3D'
        construction: bool
            Whether this is a geometry used only for construction of other geometries. If ``True``, this
            geometry will not be exported or plotted. Default: ``False``
        """
        self.control_points = [Point3D.from_array(p) for p in control_points] if isinstance(
            control_points, np.ndarray) else control_points
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
        super().__init__(name=name, construction=construction)

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

        return RationalBezierCurve3D(new_control_points, new_weights)

    def get_control_point_array(self, unit: str = "m") -> np.ndarray:
        r"""
        Gets an array representation of the control points

        Parameters
        ----------
        unit: str
            Physical length unit used to determine the output array. Default: ``"m"``

        Returns
        -------
        numpy.ndarray
            Array of size :math:`(n+1)\times 3` where :math:`n` is the curve degree
        """
        return np.array([p.as_array(unit=unit) for p in self.control_points])

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

    def evaluate(self, t: float or int or np.ndarray) -> np.ndarray:
        P = self.get_control_point_array()
        w = self.weights
        if isinstance(t, float):
            return np.array(rational_bezier_curve_eval(P, w, t))
        if isinstance(t, int):
            return np.array(rational_bezier_curve_eval_grid(P, w, t))
        return np.array(rational_bezier_curve_eval_tvec(P, w, t))

    def evaluate_point3d(self, t: float or int or np.ndarray) -> Point3D or typing.List[Point3D]:
        curve = self.evaluate(t)
        if curve.ndim == 1:
            return Point3D.from_array(curve)
        return [Point3D.from_array(curve_point) for curve_point in curve]

    def dcdt(self, t: float or int or np.ndarray) -> np.ndarray:
        P = self.get_control_point_array()
        w = self.weights
        if isinstance(t, float):
            return np.array(rational_bezier_curve_dcdt(P, w, t))
        if isinstance(t, int):
            return np.array(rational_bezier_curve_dcdt_grid(P, w, t))
        return np.array(rational_bezier_curve_dcdt_tvec(P, w, t))

    def d2cdt2(self, t: float or int or np.ndarray) -> np.ndarray:
        P = self.get_control_point_array()
        w = self.weights
        if isinstance(t, float):
            return np.array(rational_bezier_curve_d2cdt2(P, w, t))
        if isinstance(t, int):
            return np.array(rational_bezier_curve_d2cdt2_grid(P, w, t))
        return np.array(rational_bezier_curve_d2cdt2_tvec(P, w, t))

    def evaluate_pcurvedata(self, t: float or int or np.ndarray) -> PCurveData3D:
        xyz = self.evaluate(t)
        xpypzp = self.dcdt(t)
        xppyppzpp = self.d2cdt2(t)
        return PCurveData3D(
            t=t,
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
            xp=xpypzp[:, 0], yp=xpypzp[:, 1], zp=xpypzp[:, 2],
            xpp=xppyppzpp[:, 0], ypp=xppyppzpp[:, 1], zpp=xppyppzpp[:, 2]
        )

    def compute_t_corresponding_to_x(self, x_seek: float, t0: float = 0.5):
        r"""
        Computes the :math:`t`-value corresponding to a given :math:`x`-value

        Parameters
        ----------
        x_seek: float
            :math:`x`-value
        t0: float
            Initial guess for the output :math:`t`-value. Default: ``0.5``

        Returns
        -------
        float
            :math:`t`-value corresponding to ``x_seek``
        """

        def bez_root_find_func(t):
            point = self.evaluate_point3d(t[0])
            return np.array([point.x.m - x_seek])

        return fsolve(bez_root_find_func, x0=np.array([t0]))[0]

    def compute_t_corresponding_to_y(self, y_seek: float, t0: float = 0.5):
        r"""
        Computes the :math:`t`-value corresponding to a given :math:`y`-value

        Parameters
        ----------
        y_seek: float
            :math:`y`-value
        t0: float
            Initial guess for the output :math:`t`-value. Default: ``0.5``

        Returns
        -------
        float
            :math:`t`-value corresponding to ``y_seek``
        """

        def bez_root_find_func(t):
            point = self.evaluate_point3d(t[0])
            return np.array([point.y.m - y_seek])

        return fsolve(bez_root_find_func, x0=np.array([t0]))[0]

    def compute_t_corresponding_to_z(self, z_seek: float, t0: float = 0.5):
        r"""
        Computes the :math:`t`-value corresponding to a given :math:`z`-value

        Parameters
        ----------
        z_seek: float
            :math:`z`-value
        t0: float
            Initial guess for the output :math:`t`-value. Default: ``0.5``

        Returns
        -------
        float
            :math:`t`-value corresponding to ``z_seek``
        """

        def bez_root_find_func(t):
            point = self.evaluate_point3d(t[0])
            return np.array([point.z.m - z_seek])

        return fsolve(bez_root_find_func, x0=np.array([t0]))[0]

    def transform(self, **transformation_kwargs) -> "RationalBezierCurve3D":
        """
        Creates a transformed copy of the curve by transforming each of the control points

        Parameters
        ----------
        transformation_kwargs
            Keyword arguments passed to :obj:`~aerocaps.geom.transformation.Transformation3D`

        Returns
        -------
        RationalBezierCurve3D
            Transformed curve
        """
        transformation = Transformation3D(**transformation_kwargs)
        return RationalBezierCurve3D(
            transformation.transform(self.get_control_point_array()),
            weights=deepcopy(self.weights), 
            name=self.name, 
            construction=self.construction
        )

    def plot(self, ax: plt.Axes or pv.Plotter, projection: str = None, nt: int = 201, **plt_kwargs):
        """
        Plots the curve on a :obj:`matplotlib.pyplot.Axes` or a `pyvista.Plotter` window

        Parameters
        ----------
        ax: plt.Axes or pv.Plotter
            Axes/window on which to plot
        projection: str
            Projection on which to plot (either 'XY', 'YZ', 'XZ', or 'XYZ' for a 3-D plot). Only used if
            ``ax`` is a ``plt.Axes``. Defaults to 'XYZ' if not specified. Default: ``None``
        nt: int
            Number of evenly-spaced parameter values to plot. Default: ``201``
        plt_kwargs
            Additional keyword arguments to pass to :obj:`matplotlib.pyplot.Axes.plot` or
            :obj:`pyvista.Plotter.add_lines`
        """
        projection = "XYZ" if projection is None else projection
        t_vec = np.linspace(0.0, 1.0, nt)
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
        """
        Plots the control points on a :obj:`matplotlib.pyplot.Axes`

        Parameters
        ----------
        ax: plt.Axes or pv.Plotter
            Axes/window on which to plot
        projection: str
            Projection on which to plot (either 'XY', 'YZ', 'XZ', or 'XYZ' for a 3-D plot). Only used if
            ``ax`` is a ``plt.Axes``. Defaults to 'XYZ' if not specified. Default: ``None``
        plt_kwargs
            Additional keyword arguments to pass to :obj:`matplotlib.pyplot.Axes.plot` or
            :obj:`pyvista.Plotter.add_lines`
        """
        projection = "XYZ" if projection is None else projection
        cps = self.get_control_point_array()
        args = tuple([cps[:, _projection_dict[axis]] for axis in projection])
        ax.plot(*args, **plt_kwargs)

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


class BSplineCurve3D(PCurve3D):
    """Three-dimensional B-spline curve class"""
    def __init__(self,
                 control_points: typing.List[Point3D] or np.ndarray,
                 knot_vector: np.ndarray,
                 degree: int,
                 name: str = "BSplineCurve3D",
                 construction: bool = False
                 ):
        """
        Three-dimensional B-spline curve class

        Parameters
        ----------
        control_points
        knot_vector
        degree
        name: str
            Name of the geometric object. May be re-assigned a unique name when added to a
            :obj:`~aerocaps.geom.geometry_container.GeometryContainer`. Default: 'BSplineCurve3D'
        construction: bool
            Whether this is a geometry used only for construction of other geometries. If ``True``, this
            geometry will not be exported or plotted. Default: ``False``
        """
        control_points = [Point3D.from_array(p) for p in control_points] if isinstance(
            control_points, np.ndarray) else control_points
        assert knot_vector.ndim == 1
        assert len(knot_vector) == len(control_points) + degree + 1

        self.control_points = control_points
        self.dim = 3
        self.knot_vector = np.array(knot_vector)
        self.weights = np.ones(len(self.control_points))
        self.degree = degree
        super().__init__(name=name, construction=construction)

    def to_iges(self, *args, **kwargs) -> aerocaps.iges.entity.IGESEntity:
        return aerocaps.iges.curves.RationalBSplineCurveIGES(
            knots=self.knot_vector,
            weights=self.weights,
            control_points_XYZ=self.control_points,
            degree=self.degree
        )

    def reverse(self) -> "BSplineCurve3D":
        return self.__class__(np.flipud(self.get_control_point_array()),
                              (1.0 - self.knot_vector)[::-1],
                              self.degree)

    def get_control_point_array(self, unit: str = "m") -> np.ndarray:
        r"""
        Gets an array representation of the control points

        Parameters
        ----------
        unit: str
            Physical length unit used to determine the output array. Default: ``"m"``

        Returns
        -------
        numpy.ndarray
            Array of size :math:`(n+1)\times 3` where :math:`n` is the curve degree
        """
        return np.array([p.as_array(unit=unit) for p in self.control_points])

    def evaluate(self, t: float or int or np.ndarray) -> np.ndarray:
        P = self.get_control_point_array()
        k = self.knot_vector
        if isinstance(t, float):
            return np.array(bspline_curve_eval(P, k, t))
        if isinstance(t, int):
            return np.array(bspline_curve_eval_grid(P, k, t))
        return np.array(bspline_curve_eval_tvec(P, k, t))

    def evaluate_point3d(self, t: float or int or np.ndarray) -> Point3D or typing.List[Point3D]:
        curve = self.evaluate(t)
        if curve.ndim == 1:
            return Point3D.from_array(curve)
        return [Point3D.from_array(curve_point) for curve_point in curve]

    def dcdt(self, t: float or int or np.ndarray) -> np.ndarray:
        P = self.get_control_point_array()
        k = self.knot_vector
        if isinstance(t, float):
            return np.array(bspline_curve_dcdt(P, k, t))
        if isinstance(t, int):
            return np.array(bspline_curve_dcdt_grid(P, k, t))
        return np.array(bspline_curve_dcdt_tvec(P, k, t))

    def d2cdt2(self, t: float or int or np.ndarray) -> np.ndarray:
        P = self.get_control_point_array()
        k = self.knot_vector
        if isinstance(t, float):
            return np.array(bspline_curve_d2cdt2(P, k, t))
        if isinstance(t, int):
            return np.array(bspline_curve_d2cdt2_grid(P, k, t))
        return np.array(bspline_curve_d2cdt2_tvec(P, k, t))

    def evaluate_pcurvedata(self, t: float or int or np.ndarray) -> PCurveData3D:
        xyz = self.evaluate(t)
        xpypzp = self.dcdt(t)
        xppyppzpp = self.d2cdt2(t)
        return PCurveData3D(
            t=t,
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
            xp=xpypzp[:, 0], yp=xpypzp[:, 1], zp=xpypzp[:, 2],
            xpp=xppyppzpp[:, 0], ypp=xppyppzpp[:, 1], zpp=xppyppzpp[:, 2]
        )

    def transform(self, **transformation_kwargs) -> "BSplineCurve3D":
        """
        Creates a transformed copy of the curve by transforming each of the control points

        Parameters
        ----------
        transformation_kwargs
            Keyword arguments passed to :obj:`~aerocaps.geom.transformation.Transformation3D`

        Returns
        -------
        BSplineCurve3D
            Transformed curve
        """
        transformation = Transformation3D(**transformation_kwargs)
        return BSplineCurve3D(
            transformation.transform(self.get_control_point_array()),
            knot_vector=deepcopy(self.knot_vector), 
            name=self.name, 
            construction=self.construction
        )

    def plot(self, ax: plt.Axes or pv.Plotter, projection: str = None, nt: int = 201, **plt_kwargs):
        """
        Plots the curve on a :obj:`matplotlib.pyplot.Axes` or a `pyvista.Plotter` window

        Parameters
        ----------
        ax: plt.Axes or pv.Plotter
            Axes/window on which to plot
        projection: str
            Projection on which to plot (either 'XY', 'YZ', 'XZ', or 'XYZ' for a 3-D plot). Only used if
            ``ax`` is a ``plt.Axes``. Defaults to 'XYZ' if not specified. Default: ``None``
        nt: int
            Number of evenly-spaced parameter values to plot. Default: ``201``
        plt_kwargs
            Additional keyword arguments to pass to :obj:`matplotlib.pyplot.Axes.plot` or
            :obj:`pyvista.Plotter.add_lines`
        """
        projection = "XYZ" if projection is None else projection
        t_vec = np.linspace(0.0, 1.0, nt)
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


class NURBSCurve3D(PCurve3D):
    """Three-dimensional Non-Uniform Rational B-Spline (NURBS) curve class"""
    def __init__(self,
                 control_points: typing.List[Point3D] or np.ndarray,
                 weights: np.ndarray,
                 knot_vector: np.ndarray,
                 degree: int,
                 name: str = "NURBSCurve3D",
                 construction: bool = False):
        """
        Non-uniform rational B-spline (NURBS) curve class

        .. warning::

            Need to make degree a get-only property and not a parameter in the ``__init__``

        Parameters
        ----------
        control_points
        weights
        knot_vector
        name: str
            Name of the geometric object. May be re-assigned a unique name when added to a
            :obj:`~aerocaps.geom.geometry_container.GeometryContainer`. Default: 'NURBSCurve3D'
        construction: bool
            Whether this is a geometry used only for construction of other geometries. If ``True``, this
            geometry will not be exported or plotted. Default: ``False``
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
        super().__init__(name=name, construction=construction)

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

    def get_control_point_array(self, unit: str = "m") -> np.ndarray:
        r"""
        Gets an array representation of the control points

        Parameters
        ----------
        unit: str
            Physical length unit used to determine the output array. Default: ``"m"``

        Returns
        -------
        numpy.ndarray
            Array of size :math:`(n+1)\times 3` where :math:`n` is the curve degree
        """
        return np.array([p.as_array(unit=unit) for p in self.control_points])

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

    def evaluate(self, t: float or int or np.ndarray) -> np.ndarray:
        P = self.get_control_point_array()
        w = self.weights
        k = self.knot_vector
        if isinstance(t, float):
            return np.array(nurbs_curve_eval(P, w, k, t))
        if isinstance(t, int):
            return np.array(nurbs_curve_eval_grid(P, w, k, t))
        return np.array(nurbs_curve_eval_tvec(P, w, k, t))

    def evaluate_point3d(self, t: float or int or np.ndarray) -> Point3D or typing.List[Point3D]:
        curve = self.evaluate(t)
        if curve.ndim == 1:
            return Point3D.from_array(curve)
        return [Point3D.from_array(curve_point) for curve_point in curve]

    def dcdt(self, t: float or int or np.ndarray) -> np.ndarray:
        P = self.get_control_point_array()
        w = self.weights
        k = self.knot_vector
        if isinstance(t, float):
            return np.array(nurbs_curve_dcdt(P, w, k, t))
        if isinstance(t, int):
            return np.array(nurbs_curve_dcdt_grid(P, w, k, t))
        return np.array(nurbs_curve_dcdt_tvec(P, w, k, t))

    def d2cdt2(self, t: float or int or np.ndarray) -> np.ndarray:
        P = self.get_control_point_array()
        w = self.weights
        k = self.knot_vector
        if isinstance(t, float):
            return np.array(nurbs_curve_d2cdt2(P, w, k, t))
        if isinstance(t, int):
            return np.array(nurbs_curve_d2cdt2_grid(P, w, k, t))
        return np.array(nurbs_curve_d2cdt2_tvec(P, w, k, t))

    def evaluate_pcurvedata(self, t: float or int or np.ndarray) -> PCurveData3D:
        xyz = self.evaluate(t)
        xpypzp = self.dcdt(t)
        xppyppzpp = self.d2cdt2(t)
        return PCurveData3D(
            t=t,
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
            xp=xpypzp[:, 0], yp=xpypzp[:, 1], zp=xpypzp[:, 2],
            xpp=xppyppzpp[:, 0], ypp=xppyppzpp[:, 1], zpp=xppyppzpp[:, 2]
        )

    def transform(self, **transformation_kwargs) -> "NURBSCurve3D":
        """
        Creates a transformed copy of the curve by transforming each of the control points

        Parameters
        ----------
        transformation_kwargs
            Keyword arguments passed to :obj:`~aerocaps.geom.transformation.Transformation3D`

        Returns
        -------
        NURBSCurve3D
            Transformed curve
        """
        transformation = Transformation3D(**transformation_kwargs)
        return NURBSCurve3D(
            transformation.transform(self.get_control_point_array()),
            weights=deepcopy(self.weights),
            knot_vector=deepcopy(self.knot_vector),
            name=self.name, 
            construction=self.construction
        )

    def plot(self, ax: plt.Axes or pv.Plotter, projection: str = None, nt: int = 201, **plt_kwargs):
        """
        Plots the curve on a :obj:`matplotlib.pyplot.Axes` or a `pyvista.Plotter` window

        Parameters
        ----------
        ax: plt.Axes or pv.Plotter
            Axes/window on which to plot
        projection: str
            Projection on which to plot (either 'XY', 'YZ', 'XZ', or 'XYZ' for a 3-D plot). Only used if
            ``ax`` is a ``plt.Axes``. Defaults to 'XYZ' if not specified. Default: ``None``
        nt: int
            Number of evenly-spaced parameter values to plot. Default: ``201``
        plt_kwargs
            Additional keyword arguments to pass to :obj:`matplotlib.pyplot.Axes.plot` or
            :obj:`pyvista.Plotter.add_lines`
        """
        projection = "XYZ" if projection is None else projection
        t_vec = np.linspace(0.0, 1.0, nt)
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


class CompositeCurve2D(Geometry2D):
    """Two-dimensional composite curve class"""
    def __init__(self, curves: typing.List[PCurve2D],
                 name: str = "CompositeCurve2D", construction: bool = False):
        """
        Two-dimensional composite curve (list of two-dimensional curves connected end-to-end)

        Parameters
        ----------
        curves: typing.List[PCurve2D]
            List of connected curves
        name: str
            Name of the geometric object. May be re-assigned a unique name when added to a
            :obj:`~aerocaps.geom.geometry_container.GeometryContainer`. Default: 'CompositeCurve2D'
        construction: bool
            Whether this is a geometry used only for construction of other geometries. If ``True``, this
            geometry will not be exported or plotted. Default: ``False``
        """
        self._validate(curves)
        self._unordered_curves = curves
        self._ordered_curves = self._get_ordered_curve_list()
        super().__init__(name=name, construction=construction)

    @property
    def unordered_curves(self) -> typing.List[PCurve2D]:
        return self._unordered_curves

    @property
    def ordered_curves(self) -> typing.List[PCurve2D]:
        return self._ordered_curves

    @staticmethod
    def _validate(unordered_curves: typing.List[PCurve2D]) -> bool:
        """
        Validates that the set of curves is connected end-to-end

        Parameters
        ----------
        unordered_curves: typing.List[PCurve2D]
            Unordered list of curves

        Returns
        -------
        bool
            Whether the list is closed (not whether the set of curves is connected; an
            exception is raised instead if this is not the case)
        """
        endpoints = {}
        for curve in unordered_curves:
            endpoint_1 = curve.evaluate_point2d(0.0)
            endpoint_2 = curve.evaluate_point2d(1.0)
            for endpoint_to_test in [endpoint_1, endpoint_2]:
                for endpoint in endpoints:
                    if endpoint.almost_equals(endpoint_to_test):
                        endpoints[endpoint] = True
                        break
                else:
                    endpoints[endpoint_to_test] = False

        unconnected_endpoint_counter = 0
        for v in endpoints.values():
            if v:
                continue
            else:
                unconnected_endpoint_counter += 1

        if unconnected_endpoint_counter > 1:
            raise ValueError("Curve loop is not connected end to end")
        elif unconnected_endpoint_counter == 1:
            return False
        return True

    def _get_ordered_curve_list(self) -> (
            typing.List[BezierCurve2D or RationalBezierCurve2D or Line2D]):
        # Copy a list of the curves
        curve_stack = deepcopy(self.unordered_curves[1:])
        ordered_curves = [deepcopy(self.unordered_curves[0])]

        while curve_stack:  # Loop until the curve stack is empty
            for curve_idx, curve in enumerate(curve_stack):
                if ordered_curves[-1].evaluate_point2d(1.0).almost_equals(
                        curve.evaluate_point2d(0.0)):
                    ordered_curves.append(curve)
                    break  # Go to the next curve in the stack
                elif ordered_curves[-1].evaluate_point2d(1.0).almost_equals(
                        curve.evaluate_point2d(1.0)):
                    ordered_curves.append(curve.reverse())
                    break  # Go to the next curve in the stack

        return ordered_curves

    def evaluate(self, Nt: int) -> np.ndarray:
        return np.vstack(tuple([
            curve.evaluate(Nt) if c_idx == 0 else curve.evaluate(Nt)[1:, :]
            for c_idx, curve in enumerate(self.ordered_curves)
        ]))


class CompositeCurve3D(Geometry3D):
    """Three-dimensional composite curve class"""

    def __init__(self, curves: typing.List[PCurve3D],
                 name: str = "CompositeCurve3D", construction: bool = False):
        """
        Three-dimensional composite curve (list of three-dimensional curves connected end-to-end)

        Parameters
        ----------
        curves: typing.List[PCurve3D]
            List of connected curves
        name: str
            Name of the geometric object. May be re-assigned a unique name when added to a
            :obj:`~aerocaps.geom.geometry_container.GeometryContainer`. Default: 'CompositeCurve3D'
        construction: bool
            Whether this is a geometry used only for construction of other geometries. If ``True``, this
            geometry will not be exported or plotted. Default: ``False``
        """
        self._validate(curves)
        self._unordered_curves = curves
        self._ordered_curves = self._get_ordered_curve_list()
        super().__init__(name=name, construction=construction)

    @property
    def unordered_curves(self) -> typing.List[PCurve3D]:
        return self._unordered_curves

    @property
    def ordered_curves(self) -> typing.List[PCurve3D]:
        return self._ordered_curves

    @staticmethod
    def _validate(unordered_curves: typing.List[PCurve3D]) -> bool:
        """
        Validates that the set of curves is connected end-to-end

        Parameters
        ----------
        unordered_curves: typing.List[PCurve3D]
            Unordered list of curves

        Returns
        -------
        bool
            Whether the list is closed (not whether the set of curves is connected; an
            exception is raised instead if this is not the case)
        """
        endpoints = {}
        for curve in unordered_curves:
            endpoint_1 = curve.evaluate_point3d(0.0)
            endpoint_2 = curve.evaluate_point3d(1.0)
            for endpoint_to_test in [endpoint_1, endpoint_2]:
                for endpoint in endpoints:
                    if endpoint.almost_equals(endpoint_to_test):
                        endpoints[endpoint] = True
                        break
                else:
                    endpoints[endpoint_to_test] = False

        unconnected_endpoint_counter = 0
        for v in endpoints.values():
            if v:
                continue
            else:
                unconnected_endpoint_counter += 1

        if unconnected_endpoint_counter > 1:
            raise ValueError("Curve loop is not connected end to end")
        elif unconnected_endpoint_counter == 1:
            return False
        return True

    def _get_ordered_curve_list(self) -> (
            typing.List[BezierCurve3D or RationalBezierCurve3D or BSplineCurve3D or NURBSCurve3D or Line3D]):
        # Copy a list of the curves
        curve_stack = deepcopy(self.unordered_curves[1:])
        ordered_curves = [deepcopy(self.unordered_curves[0])]

        while curve_stack:  # Loop until the curve stack is empty
            for curve_idx, curve in enumerate(curve_stack):
                if ordered_curves[-1].evaluate_point3d(1.0).almost_equals(
                        curve.evaluate_point3d(0.0)):
                    ordered_curves.append(curve)
                    curve_stack.pop(curve_idx)
                    break  # Go to the next curve in the stack
                elif ordered_curves[-1].evaluate_point3d(1.0).almost_equals(
                        curve.evaluate_point3d(1.0)):
                    ordered_curves.append(curve.reverse())
                    curve_stack.pop(curve_idx)
                    break  # Go to the next curve in the stack

        return ordered_curves
    
    def transform(self, **transformation_kwargs) -> "CompositeCurve3D":
        """
        Creates a transformed copy of the curve by transforming each of the child curves

        Parameters
        ----------
        transformation_kwargs
            Keyword arguments passed to :obj:`~aerocaps.geom.transformation.Transformation3D`

        Returns
        -------
        CompositeCurve3D
            Transformed curve
        """
        return CompositeCurve3D(
            [curve.transform(**transformation_kwargs) for curve in self.unordered_curves],
            name=self.name,
            construction=self.construction
        )

    def evaluate(self, Nt: int) -> np.ndarray:
        return np.vstack(tuple([
            curve.evaluate(Nt) if c_idx == 0 else curve.evaluate(Nt)[1:, :]
            for c_idx, curve in enumerate(self.ordered_curves)
        ]))

    def to_iges(self, curve_iges_entities: typing.List[aerocaps.iges.entity.IGESEntity],
                *args, **kwargs) -> aerocaps.iges.entity.IGESEntity:
        return aerocaps.iges.curves.CompositeCurveIGES(
            curve_iges_entities
        )


class CurveOnParametricSurface(Geometry3D):
    def __init__(self,
                 surface: aerocaps.geom.Surface,
                 parametric_curve: aerocaps.geom.Geometry3D,
                 model_space_curve: aerocaps.geom.Geometry3D,
                 name: str = "CurveOnParametricSurface",
                 construction: bool = False):
        """

        Parameters
        ----------
        surface
        parametric_curve
        model_space_curve
        name: str
            Name of the geometric object. May be re-assigned a unique name when added to a
            :obj:`~aerocaps.geom.geometry_container.GeometryContainer`. Default: 'CurveOnParametricSurface'
        construction: bool
            Whether this is a geometry used only for construction of other geometries. If ``True``, this
            geometry will not be exported or plotted. Default: ``False``
        """
        self.surface = surface
        self.parametric_curve = parametric_curve
        self.model_space_curve = model_space_curve
        super().__init__(name=name, construction=construction)

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
    bspline = BSplineCurve3D(np.array([
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
    p = bspline.get_control_point_array()
    plt.plot(data[:, 0], data[:, 1], color="steelblue")
    plt.plot(p[:, 0], p[:, 1], ls=":", color="grey", marker="o", mec="steelblue", mfc="none")
    plt.plot([data[75, 0], data[150, 0], data[225, 0]], [data[75, 1], data[150, 1], data[225, 1]], ls="none", marker="o", mfc="indianred", mec="indianred")
    plt.show()


if __name__ == "__main__":
    main()
