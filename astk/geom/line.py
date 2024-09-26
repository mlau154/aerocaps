import typing

import numpy as np
import pyvista as pv

import pyiges


from astk.geom import Geometry3D, Geometry2D
from astk.geom.point import Point3D, Point2D
from astk.geom.vector import Vector3D, Vector2D
import astk.iges.entity
import astk.iges.curves
from astk.units.angle import Angle
from astk.units.length import Length


class Line2D(Geometry2D):
    def __init__(self,
                 p0: Point2D,
                 theta: Angle = None,
                 p1: Point2D = None,
                 d: Length = Length(m=1.0)
                 ):
        if theta and p1:
            raise ValueError("Angle theta should not be specified if p1 is specified")
        if not theta and not p1:
            raise ValueError("Must specify either angle theta or p1")
        self.p0 = p0
        self.theta = theta
        from tpai.geometry.tools import measure_distance_between_points  # Avoid circular import
        self.d = d if not p1 else Length(m=measure_distance_between_points(p0, p1))
        self.p1 = self.evaluate(1.0) if not p1 else p1

    def _evaluate_single_t(self, t: float) -> Point2D:
        if self.theta:
            return Point2D(
                x=self.p0.x + self.d * np.cos(self.theta.rad) * t,
                y=self.p0.y + self.d * np.sin(self.theta.rad) * t
            )
        else:
            return self.p0 + t * (self.p1 - self.p0)

    def evaluate(self, t: float or typing.Iterable) -> Point2D or typing.List[Point2D]:
        if isinstance(t, float):
            return self._evaluate_single_t(t)
        elif isinstance(t, typing.Iterable):
            return [self._evaluate_single_t(t_val) for t_val in t]
        else:
            raise TypeError(f"t must be of type float or typing.Iterable")

    def get_vector(self) -> Vector2D:
        return Vector2D(p0=self.p0, p1=self.p1)

    def plot(self, plot: pv.Plotter, **line_kwargs):
        line_arr = np.array([self.p0.as_array(), self.p1.as_array()])
        plot.add_lines(line_arr, **line_kwargs)


class Line3D(Geometry3D):
    def __init__(self,
                 p0: Point3D,
                 theta: Angle = None,
                 phi: Angle = None,
                 p1: Point3D = None,
                 d: Length = Length(m=1.0)
                 ):
        if (theta and p1) or (phi and p1):
            raise ValueError("Angles should not be specified if p1 is specified")
        if (not theta and not p1) or (not phi and not p1):
            raise ValueError("Must specify either both angles, theta and phi, or p1")
        self.p0 = p0
        self.theta = theta
        self.phi = phi
        from tpai.geometry.tools import measure_distance_between_points  # Avoid circular import
        self.d = d if not p1 else Length(m=measure_distance_between_points(p0, p1))
        self.p1 = self.evaluate(1.0) if not p1 else p1

    def to_iges(self, *args, **kwargs) -> astk.iges.entity.Entity:
        return astk.iges.curves.Line(self.p0.as_array(), self.p1.as_array())

    def from_iges(self):
        pass

    def projection_on_principal_plane(self, plane: str = "XY") -> Line2D:
        return Line2D(p0=self.p0.projection_on_principal_plane(plane), p1=self.p1.projection_on_principal_plane(plane))

    def _evaluate_single_t(self, t: float) -> Point3D:
        if self.phi and self.theta:
            return Point3D(
                x=self.p0.x + self.d * np.cos(self.phi.rad) * np.cos(self.theta.rad) * t,
                y=self.p0.y + self.d * np.cos(self.phi.rad) * np.sin(self.theta.rad) * t,
                z=self.p0.z + self.d * np.sin(self.phi.rad) * t
            )
        else:
            return self.p0 + t * (self.p1 - self.p0)

    def evaluate(self, t: float or typing.Iterable) -> Point3D or typing.List[Point3D]:
        if isinstance(t, float):
            return self._evaluate_single_t(t)
        elif isinstance(t, typing.Iterable):
            return [self._evaluate_single_t(t_val) for t_val in t]
        else:
            raise TypeError(f"t must be of type float or typing.Iterable")

    def get_vector(self) -> Vector3D:
        return Vector3D(p0=self.p0, p1=self.p1)

    def plot(self, plot: pv.Plotter, **line_kwargs):
        line_arr = np.array([self.p0.as_array(), self.p1.as_array()])
        plot.add_lines(line_arr, **line_kwargs)
