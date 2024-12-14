from aerocaps.geom.point import Point3D, Origin3D
from aerocaps.units.length import Length


__all__ = [
    "Plane",
    "PlaneX",
    "PlaneY",
    "PlaneZ"
]


class Plane:
    def __init__(self, p0: Point3D, p1: Point3D, p2: Point3D):
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2

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
            p1=Point3D(x=Length(m=1.0), y=distance_from_origin, z=Length(m=0.0)),
            p2=Point3D(x=Length(m=0.0), y=distance_from_origin, z=Length(m=1.0))
        )

    @classmethod
    def plane_parallel_Z(cls, distance_from_origin: Length):
        return cls(
            p0=Point3D(x=Length(m=0.0), y=Length(m=0.0), z=distance_from_origin),
            p1=Point3D(x=Length(m=0.0), y=Length(m=1.0), z=distance_from_origin),
            p2=Point3D(x=Length(m=1.0), y=Length(m=0.0), z=distance_from_origin)
        )


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
            p1=Point3D(x=Length(m=1.0), y=Length(m=0.0), z=Length(m=0.0)),
            p2=Point3D(x=Length(m=0.0), y=Length(m=0.0), z=Length(m=1.0))
        )


class PlaneZ(Plane):
    def __init__(self):
        super().__init__(
            p0=Origin3D(),
            p1=Point3D(x=Length(m=0.0), y=Length(m=1.0), z=Length(m=0.0)),
            p2=Point3D(x=Length(m=1.0), y=Length(m=0.0), z=Length(m=0.0))
        )
