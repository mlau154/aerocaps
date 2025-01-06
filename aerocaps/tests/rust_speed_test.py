from aerocaps import bernstein_poly as rust_bern
from aerocaps import bezier_curve_eval
from aerocaps.utils.math import bernstein_poly as py_bern
import aerocaps as ac
import time
import numpy.random as rand
import numpy as np


def main():
    nums = rand.uniform(0.0, 1.0, 100000)
    start_py = time.perf_counter()
    for num in nums:
        py_bern(10, 4, num)
    end_py = time.perf_counter()

    start_rust = time.perf_counter()
    for num in nums:
        rust_bern(10, 4, num)
    end_rust = time.perf_counter()

    print(f"Bernstein poly. "
          f"Python time: {end_py - start_py:.3f} seconds. "
          f"Rust time: {end_rust - start_rust:.3f} seconds.")

    P = np.array([[0.0, 0.0, 0.0], [0.3, 0.5, 0.0], [0.1, -0.2, 0.3], [0.5, 0.1, 0.2], [0.6, 1.0, 2.0]])
    start_py = time.perf_counter()
    for num in nums:
        bez = ac.Bezier3D.generate_from_array(P)
        bez.evaluate(num)
    end_py = time.perf_counter()

    start_rust = time.perf_counter()
    for num in nums:
        bezier_curve_eval(P, num)
    end_rust = time.perf_counter()

    print(f"Bezier evaluation. "
          f"Python time: {end_py - start_py:.3f} seconds. "
          f"Rust time: {end_rust - start_rust:.3f} seconds.")


if __name__ == "__main__":
    main()
