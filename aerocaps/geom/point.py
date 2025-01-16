from copy import deepcopy

import numpy as np
import pyvista as pv

from aerocaps.units.length import Length
from aerocaps.geom import Geometry2D, Geometry3D
import aerocaps.iges.entity
import aerocaps.iges.point


__all__ = [
    "Point2D",
    "Point3D",
    "Origin2D",
    "Origin3D"
]


class Point2D(Geometry2D):
    """Two-dimensional point class"""
    def __init__(self, x: Length, y: Length, name: str = "Point2D", construction: bool = False):
        """
        Two-dimensional point class

        Parameters
        ----------
        x: Length
        y: Length
        name: str
            Name of the geometric object. May be re-assigned a unique name when added to a
            :obj:`~aerocaps.geom.geometry_container.GeometryContainer`. Default: 'Point2D'
        construction: bool
            Whether this is a geometry used only for construction of other geometries. If ``True``, this
            geometry will not be exported or plotted. Default: ``False``
        """
        self.x = x
        self.y = y
        super().__init__(name=name, construction=construction)

    def as_array(self, unit: str = "m"):
        return np.array([getattr(self.x, unit), getattr(self.y, unit)])

    @classmethod
    def from_array(cls, arr: np.ndarray, unit: str = "m"):
        return cls(x=Length(**{unit: arr[0]}), y=Length(**{unit: arr[1]}))

    def __add__(self, other):
        return Point2D(x=self.x + other.x, y=self.y + other.y)

    def __sub__(self, other):
        return Point2D(x=self.x - other.x, y=self.y - other.y)

    def __mul__(self, other):
        if isinstance(other, float):
            return Point2D(x=self.x * other, y=self.y * other)
        else:
            raise ValueError("Only multiplication between points and scalars is currently supported")

    def __rmul__(self, other):
        return self.__mul__(other)


class Point3D(Geometry3D):
    """
    Three-dimensional point class
    """
    def __init__(self, x: Length, y: Length, z: Length, name: str = "Point3D", construction: bool = False):
        """
        Three-dimensional point class

        Parameters
        ----------
        x: Length
        y: Length
        z: Length
        name: str
            Name of the geometric object. May be re-assigned a unique name when added to a
            :obj:`~aerocaps.geom.geometry_container.GeometryContainer`. Default: 'Point3D'
        construction: bool
            Whether this is a geometry used only for construction of other geometries. If ``True``, this
            geometry will not be exported or plotted. Default: ``False``
        """
        self.x = x
        self.y = y
        self.z = z
        super().__init__(name=name, construction=construction)

    def to_iges(self, *args, **kwargs) -> aerocaps.iges.entity.IGESEntity:
        return aerocaps.iges.point.PointIGES(self.as_array())

    def projection_on_principal_plane(self, plane: str = "XY") -> Point2D:
        if plane == "XY":
            return Point2D(x=deepcopy(self.x), y=deepcopy(self.y))
        elif plane == "XZ":
            return Point2D(x=deepcopy(self.x), y=deepcopy(self.z))
        elif plane == "YZ":
            return Point2D(x=deepcopy(self.y), y=deepcopy(self.z))
        else:
            raise ValueError("Invalid plane")

    def as_array(self, unit: str = "m"):
        return np.array([getattr(self.x, unit), getattr(self.y, unit), getattr(self.z, unit)])

    @classmethod
    def from_array(cls, arr: np.ndarray, unit: str = "m"):
        return cls(x=Length(**{unit: arr[0]}), y=Length(**{unit: arr[1]}), z=Length(**{unit: arr[2]}))

    def almost_equals(self, other: "Point3D"):
        return all([np.isclose(self_xyz, other_xyz) for self_xyz, other_xyz in zip(self.as_array(), other.as_array())])

    def plot(self, plot: pv.Plotter, **point_kwargs):
        plot.add_points(np.array([self.as_array()]), **point_kwargs)

    def __add__(self, other):
        return Point3D(x=self.x + other.x, y=self.y + other.y, z=self.z + other.z)

    def __sub__(self, other):
        return Point3D(x=self.x - other.x, y=self.y - other.y, z=self.z - other.z)

    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return Point3D(x=self.x * other, y=self.y * other, z=self.z * other)
        else:
            raise ValueError("Only multiplication between points and scalars is currently supported")

    def __rmul__(self, other):
        return self.__mul__(other)


class Origin2D(Point2D):
    """Two-dimensional origin point class"""
    def __init__(self, name: str = "Origin2D", construction: bool = False):
        """
        Point located at :math:`(x,y)=(0,0)`

        Parameters
        ----------
        name: str
            Name of the geometric object. May be re-assigned a unique name when added to a
            :obj:`~aerocaps.geom.geometry_container.GeometryContainer`. Default: 'Origin2D'
        construction: bool
            Whether this is a geometry used only for construction of other geometries. If ``True``, this
            geometry will not be exported or plotted. Default: ``False``
        """
        super().__init__(x=Length(m=0), y=Length(m=0), name=name, construction=construction)


class Origin3D(Point3D):
    """Three-dimensional origin point class"""
    def __init__(self, name: str = "Origin3D", construction: bool = False):
        """
        Point located at :math:`(x,y,z)=(0,0,0)`

        Parameters
        ----------
        name: str
            Name of the geometric object. May be re-assigned a unique name when added to a
            :obj:`~aerocaps.geom.geometry_container.GeometryContainer`. Default: 'Origin3D'
        construction: bool
            Whether this is a geometry used only for construction of other geometries. If ``True``, this
            geometry will not be exported or plotted. Default: ``False``
        """
        super().__init__(x=Length(m=0), y=Length(m=0), z=Length(m=0), name=name, construction=construction)
