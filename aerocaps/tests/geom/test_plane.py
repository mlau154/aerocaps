import numpy as np

from aerocaps.geom.point import Point3D
from aerocaps.geom.plane import Plane, PlaneX, PlaneY, PlaneZ
from aerocaps.units.length import Length

import pyvista as pv


_SHOW_PLOTS = False


def test_compute_normal():
    p0 = Point3D.from_array(np.array([0.0, 0.0, 0.0]))
    p1 = Point3D.from_array(np.array([0.0, 1.0, 1.0]))
    p2 = Point3D.from_array(np.array([1.0, 0.0, 1.0]))
    plane = Plane(p0, p1, p2)
    normal = plane.compute_normal()
    assert all(np.isclose(normal.as_array(), np.array([1.0, 1.0, -1.0]) / np.sqrt(3.0)))


def test_plane_parallel_x():
    plane = Plane.plane_parallel_X(distance_from_origin=Length(m=0.5))
    normal = plane.compute_normal()
    assert all(np.isclose(normal.as_array(), np.array([1.0, 0.0, 0.0])))


def test_plane_parallel_y():
    plane = Plane.plane_parallel_Y(distance_from_origin=Length(m=0.5))
    normal = plane.compute_normal()
    assert all(np.isclose(normal.as_array(), np.array([0.0, 1.0, 0.0])))


def test_plane_parallel_z():
    plane = Plane.plane_parallel_Z(distance_from_origin=Length(m=0.5))
    normal = plane.compute_normal()
    assert all(np.isclose(normal.as_array(), np.array([0.0, 0.0, 1.0])))


def test_plane_x():
    plane = PlaneX()
    normal = plane.compute_normal()
    assert all(np.isclose(normal.as_array(), np.array([1.0, 0.0, 0.0])))


def test_plane_y():
    plane = PlaneY()
    normal = plane.compute_normal()
    assert all(np.isclose(normal.as_array(), np.array([0.0, 1.0, 0.0])))


def test_plane_z():
    plane = PlaneZ()
    normal = plane.compute_normal()
    assert all(np.isclose(normal.as_array(), np.array([0.0, 0.0, 1.0])))


def test_plot():
    p0 = Point3D.from_array(np.array([0.0, 0.0, 0.0]))
    p1 = Point3D.from_array(np.array([0.0, 1.0, 1.0]))
    p2 = Point3D.from_array(np.array([1.0, 0.0, 1.0]))
    plane = Plane(p0, p1, p2)
    plot = pv.Plotter()
    plane.plot(plot, mesh_kwargs=dict(opacity=0.4, color="red"))
    plot.add_axes_at_origin()
    if _SHOW_PLOTS:
        plot.show()
