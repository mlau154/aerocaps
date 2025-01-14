from aerocaps.geom.point import Point3D, Origin3D
from aerocaps.geom.vector import Vector3D
from aerocaps.units.length import Length

import pyvista as pv


__all__ = [
    "Plane",
    "PlaneX",
    "PlaneY",
    "PlaneZ"
]


class Plane:
    def __init__(self, p0: Point3D, p1: Point3D, p2: Point3D):
        """
        Creates a plane from three points

        Parameters
        ----------
        p0: Point3D
            First point (origin of the plane)
        p1: Point3D
            Second point
        p2: Point3D
            Third point
        """
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2

    @property
    def origin(self) -> Point3D:
        return self.p0

    def compute_normal(self) -> Vector3D:
        """
        Computes the unit vector normal to the plane

        Returns
        -------
        Vector3D
            Unit normal vector
        """
        v01 = Vector3D(p0=self.p0, p1=self.p1)
        v02 = Vector3D(p0=self.p0, p1=self.p2)
        v_out = v01.cross(v02)
        return v_out.get_normalized_vector()

    @classmethod
    def plane_parallel_X(cls, distance_from_origin: Length):
        return cls(
            p0=Point3D(x=distance_from_origin, y=Length(m=0.0), z=Length(m=0.0)),
            p1=Point3D(x=distance_from_origin, y=Length(m=1.0), z=Length(m=0.0)),
            p2=Point3D(x=distance_from_origin, y=Length(m=0.0), z=Length(m=1.0))
        )

    @classmethod
    def plane_parallel_Y(cls, distance_from_origin: Length):
        return cls(
            p0=Point3D(x=Length(m=0.0), y=distance_from_origin, z=Length(m=0.0)),
            p1=Point3D(x=Length(m=0.0), y=distance_from_origin, z=Length(m=1.0)),
            p2=Point3D(x=Length(m=1.0), y=distance_from_origin, z=Length(m=0.0))
        )

    @classmethod
    def plane_parallel_Z(cls, distance_from_origin: Length):
        return cls(
            p0=Point3D(x=Length(m=0.0), y=Length(m=0.0), z=distance_from_origin),
            p1=Point3D(x=Length(m=1.0), y=Length(m=0.0), z=distance_from_origin),
            p2=Point3D(x=Length(m=0.0), y=Length(m=1.0), z=distance_from_origin)
        )

    def plot(self, plot: pv.Plotter, plane_kwargs: dict = None, mesh_kwargs: dict = None) -> pv.PolyData:
        """
        Plots the plane on the scene

        Parameters
        ----------
        plot:
            :obj:`pyvista.Plotter` instance
        plane_kwargs:
            Keyword arguments to pass to the :obj:`pyvista.Plane` constructor. Default: ``None``
        mesh_kwargs:
            Keyword arguments to pass to :obj:`pyvista.Plotter.add_mesh`. Default: ``None``

        Returns
        -------
        pv.PolyData
            The plane mesh object
        """
        mesh = pv.Plane(self.origin.as_array(), self.compute_normal().as_array(),
                        **plane_kwargs if plane_kwargs is not None else dict())
        plot.add_mesh(mesh, **mesh_kwargs if mesh_kwargs is not None else dict())
        return mesh


class PlaneX(Plane):
    def __init__(self):
        super().__init__(
            p0=Origin3D(),
            p1=Point3D(x=Length(m=0.0), y=Length(m=1.0), z=Length(m=0.0)),
            p2=Point3D(x=Length(m=0.0), y=Length(m=0.0), z=Length(m=1.0))
        )


class PlaneY(Plane):
    def __init__(self):
        super().__init__(
            p0=Origin3D(),
            p1=Point3D(x=Length(m=0.0), y=Length(m=0.0), z=Length(m=1.0)),
            p2=Point3D(x=Length(m=1.0), y=Length(m=0.0), z=Length(m=0.0))
        )


class PlaneZ(Plane):
    def __init__(self):
        super().__init__(
            p0=Origin3D(),
            p1=Point3D(x=Length(m=1.0), y=Length(m=0.0), z=Length(m=0.0)),
            p2=Point3D(x=Length(m=0.0), y=Length(m=1.0), z=Length(m=0.0))
        )
