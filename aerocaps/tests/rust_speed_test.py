from aerocaps import bernstein_poly as rust_bern
from aerocaps import bezier_curve_eval, bezier_surf_eval, bspline_curve_eval
from aerocaps.utils.math import bernstein_poly as py_bern
import aerocaps as ac
import time
import numpy.random as rand
import numpy as np


def main():
    ts = rand.uniform(0.0, 1.0, 30000)
    start_py = time.perf_counter()
    for t in ts:
        py_bern(10, 4, t)
    end_py = time.perf_counter()

    start_rust = time.perf_counter()
    for t in ts:
        rust_bern(10, 4, t)
    end_rust = time.perf_counter()

    print(f"Bernstein poly. "
          f"Python time: {end_py - start_py:.3f} seconds. "
          f"Rust time: {end_rust - start_rust:.3f} seconds.")

    P = np.array([[0.0, 0.0, 0.0], [0.3, 0.5, 0.0], [0.1, -0.2, 0.3], [0.5, 0.1, 0.2], [0.6, 1.0, 2.0]])
    start_py = time.perf_counter()
    for t in ts:
        bez = ac.Bezier3D.generate_from_array(P)
        bez.evaluate(t)
    end_py = time.perf_counter()

    start_rust = time.perf_counter()
    for t in ts:
        bezier_curve_eval(P, t)
    end_rust = time.perf_counter()

    print(f"Bezier curve evaluation with {ts.shape[0]} t-values. "
          f"Python time: {end_py - start_py:.3f} seconds. "
          f"Rust time: {end_rust - start_rust:.3f} seconds.")

    P = np.array([[0.0, 0.0, 0.0], [0.3, 0.5, 0.0], [0.1, -0.2, 0.3], [0.5, 0.1, 0.2], [0.6, 1.0, 2.0]])
    knots = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0])
    start_py = time.perf_counter()
    for t in ts:
        bez = ac.BSpline3D(P, knots, degree=3)
        bez.evaluate_ndarray(t)
    end_py = time.perf_counter()

    start_rust = time.perf_counter()
    for t in ts:
        bspline_curve_eval(P, knots, t)
    end_rust = time.perf_counter()

    print(f"B-spline evaluation with {ts.shape[0]} t-values. "
          f"Python time: {end_py - start_py:.3f} seconds. "
          f"Rust time: {end_rust - start_rust:.3f} seconds.")

    us = rand.uniform(0.0, 1.0, 30000)
    vs = rand.uniform(0.0, 1.0, 30000)
    P = np.array([
        [
            [0.0, 0.0, 0.0],
            [0.3, 0.5, 0.0],
            [0.1, -0.2, 0.3],
            [0.5, 0.1, 0.2],
            [0.6, 1.0, 2.0]
        ],
        [
            [0.0, 1.0, 0.5],
            [0.3, 1.5, 0.3],
            [0.1, 0.8, 0.3],
            [0.5, 1.1, 0.6],
            [0.6, 2.0, 3.0]
        ]
    ])
    start_py = time.perf_counter()
    for (u, v) in zip(us, vs):
        bez = ac.BezierSurface.generate_from_array(P)
        bez.evaluate_ndarray(u, v)
    end_py = time.perf_counter()

    start_rust = time.perf_counter()
    for (u, v) in zip(us, vs):
        bezier_surf_eval(P, u, v)
    end_rust = time.perf_counter()

    print(f"Bezier surface evaluation with {us.shape[0]} uv-pairs. "
          f"Python time: {end_py - start_py:.3f} seconds. "
          f"Rust time: {end_rust - start_rust:.3f} seconds.")


if __name__ == "__main__":
    main()
