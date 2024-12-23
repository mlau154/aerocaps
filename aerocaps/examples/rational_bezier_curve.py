import numpy as np

import aerocaps as ac


def quarter_circle_rational_bezier() -> ac.RationalBezierCurve3D:
    points = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0]
    ])
    weights = np.array([1.0, 1 / np.sqrt(2.0), 1.0])

    curve = ac.RationalBezierCurve3D.generate_from_array(points, weights)
    return curve
