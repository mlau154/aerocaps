import os

import numpy as np
import pytest

from aerocaps.geom.geometry_container import GeometryContainer
from aerocaps.geom.curves import BezierCurve3D
from aerocaps.geom.surfaces import BezierSurface
from aerocaps.geom.point import Point3D


_SHOW_PLOTS = False


@pytest.fixture
def geometry_container():
    point = Point3D.from_array(np.array([
        0.2, 0.1, 0.3
    ]))
    curve = BezierCurve3D(np.array([
        [0.0, 0.0, 0.0],
        [0.3, 0.2, 0.1],
        [0.6, 0.1, 0.3],
        [1.0, -0.1, 0.2]
    ]))
    surf = BezierSurface(np.array([
        [
            [0.0, 0.0, 0.0],
            [0.3, 0.2, 0.1],
            [0.6, 0.1, 0.3],
            [1.0, -0.1, 0.2]
        ],
        [
            [0.0, 0.0, 1.0],
            [0.3, 0.4, 1.1],
            [0.6, 0.2, 1.3],
            [1.0, -0.3, 1.2]
        ]
    ]))
    container = GeometryContainer()
    container.add_geometry(point)
    container.add_geometry(curve)
    container.add_geometry(surf)
    return container


def test__get_max_index_associated_with_name(geometry_container):
    # A non-existent key should raise an index of zero (its existence is handled by add_geometry)
    assert geometry_container._get_max_index_associated_with_name("RandomName") == 0

    # Make sure that the index for a name with no duplicates is zero
    assert geometry_container._get_max_index_associated_with_name("BezierCurve3D") == 0

    # Add another point and make sure that the index gets raise
    new_point = Point3D.from_array(np.array([-1.0, -2.0, 3.0]))
    geometry_container.add_geometry(new_point)
    assert geometry_container._get_max_index_associated_with_name("Point3D") == 1


def test_add_geometry(geometry_container):
    curve = BezierCurve3D(np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [-1.0, -1.0, -1.0],
        [0.0, 1.0, 2.0]
    ]), name="MyCurve", construction=True)
    geometry_container.add_geometry(curve)
    assert isinstance(geometry_container.geometry_by_name("MyCurve"), BezierCurve3D)

    curve = BezierCurve3D(np.array([
        [0.0, 0.3, 0.0],
        [1.0, 1.2, 1.0],
        [-1.0, -1.0, -1.0],
        [0.0, 1.0, 2.0]
    ]), name="MyCurve", construction=True)
    geometry_container.add_geometry(curve)
    assert geometry_container.geometry_by_name("MyCurve-1") is not None


def test_remove_geometry(geometry_container):
    # Remove the surface from the geometry
    geometry_container.remove_geometry("BezierSurface")
    assert len(geometry_container.geometry_name_list()) == 2

    # Add a curve to the geometry
    curve = BezierCurve3D(np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [-1.0, -1.0, -1.0],
        [0.0, 1.0, 2.0]
    ]), name="MyCurve", construction=True)
    geometry_container.add_geometry(curve)
    assert len(geometry_container.geometry_name_list()) == 3

    # Now, remove the curve by reference
    geometry_container.remove_geometry(curve)
    assert len(geometry_container.geometry_name_list()) == 2


def test_geometry_by_name(geometry_container):
    assert isinstance(geometry_container.geometry_by_name("Point3D"), Point3D)


def test_geometry_name_list(geometry_container):
    assert len(geometry_container.geometry_name_list()) == 3
    assert len(geometry_container.geometry_name_list(Point3D)) == 1


def test_plot(geometry_container):
    geometry_container.plot(show=_SHOW_PLOTS)


def test_export_iges(geometry_container):
    file_name = "test_iges_export_10295876681345053.igs"
    geometry_container.export_iges(file_name)
    if os.path.exists(file_name):
        os.remove(file_name)
