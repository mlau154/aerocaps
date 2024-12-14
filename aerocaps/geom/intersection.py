from aerocaps.geom.curves import Line3D
from aerocaps.geom.plane import Plane
from aerocaps.geom.point import Point3D
from aerocaps.geom.vector import Vector3D


__all__ = [
    "intersection_of_line_and_plane"
]


def intersection_of_line_and_plane(line: Line3D, plane: Plane):
    p01 = Vector3D(p0=plane.p0, p1=plane.p1)
    p02 = Vector3D(p0=plane.p0, p1=plane.p2)
    cross_vec = p01.cross(p02)
    l_a_p0 = Vector3D(p0=plane.p0, p1=line.p0)
    l_ab = Vector3D(p0=line.p0, p1=line.p1)
    numerator = cross_vec.dot(l_a_p0)
    denominator = -(l_ab.dot(cross_vec))
    t = numerator / denominator
    l_ab_val = l_ab.value()
    l_ab_t = Point3D(x=l_ab_val[0] * t, y=l_ab_val[1] * t, z=l_ab_val[2] * t)
    return line.p0 + l_ab_t
