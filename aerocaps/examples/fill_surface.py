import typing

import numpy as np

import aerocaps as ac


def fill_surface_four_sided(displacement_degree: int = 3) -> (ac.RationalBezierSurface, typing.List[ac.BezierCurve3D or ac.RationalBezierCurve3D]):
    """
    Creates a four-sided fill surface from a combination of rational and non-rational Bézier curves.

    .. figure:: ../images/fill_surface.*
        :width: 600
        :align: center

        Fill surface from four curve boundaries

    Parameters
    ----------
    displacement_degree: int
        Displacement degree to use for the fill surface top and bottom boundary insertion

    Returns
    -------
    ac.RationalBezierSurface, typing.List[ac.Bezier3D or ac.RationalBezierCurve3D]
        Fill surface and the list of the four boundary curves
    """
    left_curve = ac.RationalBezierCurve3D(
        np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 1.0, 0.0]
        ]),
        np.array([1.0, 0.7, 1.0])
    )
    right_curve = ac.BezierCurve3D(
        np.array([
            [2.0, 0.1, 0.0],
            [2.1, 0.9, 0.0]
        ])
    )
    top_curve = ac.BezierCurve3D(
        np.array([
            left_curve.control_points[-1].as_array(),
            [0.5, 1.2, 0.0],
            [1.5, 0.8, 0.0],
            right_curve.control_points[-1].as_array()
        ])
    )
    bottom_curve = ac.BezierCurve3D(
        np.array([
            left_curve.control_points[0].as_array(),
            [1.2, 0.2, 0.0],
            right_curve.control_points[0].as_array()
        ])
    )
    return (
        ac.RationalBezierSurface.fill_surface_from_four_boundaries(left_curve, right_curve, top_curve, bottom_curve,
                                                                   displacement_degree=displacement_degree),
        [left_curve, right_curve, top_curve, bottom_curve]
    )
