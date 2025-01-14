import numpy as np

import aerocaps as ac


def bezier_surface_2x3() -> ac.BezierSurface:
    r"""
    Creates a Bézier surface with degrees :math:`n=2` (three rows of control points in the :math:`u`-direction)
    and :math:`m=3` (four rows of control points in the :math:`v`-direction).

    Returns
    -------
    BezierSurface
        The :math:`2 \times 3` Bézier surface
    """
    # Create three rows of control points that each influence the surface shape in the v-direction
    row_1 = np.array([
        [0.0, 0.0, 0.0],
        [0.3, 0.0, 0.1],
        [0.6, 0.0, -0.1],
        [1.0, 0.0, 0.0]
    ])
    row_2 = np.array([
        [0.0, 0.35, 0.0],
        [0.3, 0.25, 0.2],
        [0.6, 0.35, -0.2],
        [1.0, 0.30, 0.3]
    ])
    row_3 = np.array([
        [0.0, 0.7, 0.2],
        [0.3, 0.6, -0.1],
        [0.6, 0.65, 0.3],
        [1.0, 0.7, 0.2]
    ])

    # Stack the rows into a matrix
    control_points = np.stack((row_1, row_2, row_3), axis=0)

    # Generate the surface from the array
    surf = ac.BezierSurface(control_points)

    return surf
