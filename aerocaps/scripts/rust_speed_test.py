import time
import typing

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rand
import rust_nurbs

import aerocaps as ac
from aerocaps.geom import nurbs_purepython


def case_1(N: int) -> (typing.List[float], str):
    """Bernstein polynomial evaluation at many random :math:`t`-values"""
    ts = rand.uniform(0.0, 1.0, N)
    start_py = time.perf_counter()
    for t in ts:
        nurbs_purepython.bernstein_poly(10, 4, t)
    end_py = time.perf_counter()

    start_rust = time.perf_counter()
    for t in ts:
        rust_nurbs.bernstein_poly(10, 4, t)
    end_rust = time.perf_counter()

    return [end_py - start_py, end_rust - start_rust], "Case 1"


def case_2(N: int) -> (typing.List[float], str):
    """Bézier curve evaluation"""
    ts = rand.uniform(0.0, 1.0, N)
    P = np.array([[0.0, 0.0, 0.0], [0.3, 0.5, 0.0], [0.1, -0.2, 0.3], [0.5, 0.1, 0.2], [0.6, 1.0, 2.0]])
    P_list = P.tolist()
    start_py = time.perf_counter()
    for t in ts:
        nurbs_purepython.bezier_curve_eval(P_list, t)
    end_py = time.perf_counter()

    start_rust = time.perf_counter()
    for t in ts:
        rust_nurbs.bezier_curve_eval(P, t)
    end_rust = time.perf_counter()

    return [end_py - start_py, end_rust - start_rust], "Case 2"


def case_3(N: int) -> (typing.List[float], str):
    """B-spline curve evaluation"""
    ts = rand.uniform(0.0, 1.0, N)
    P = np.array([[0.0, 0.0, 0.0], [0.3, 0.5, 0.0], [0.1, -0.2, 0.3], [0.5, 0.1, 0.2], [0.6, 1.0, 2.0]])
    P_list = P.tolist()
    knots = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0])
    knot_list = knots.tolist()
    start_py = time.perf_counter()
    for t in ts:
        nurbs_purepython.bspline_curve_eval(P_list, knot_list, t)
    end_py = time.perf_counter()

    start_rust = time.perf_counter()
    for t in ts:
        rust_nurbs.bspline_curve_eval(P, knots, t)
    end_rust = time.perf_counter()

    return [end_py - start_py, end_rust - start_rust], "Case 3"


def case_4(N: int) -> (typing.List[float], str):
    """Bezier surface evaluation at a bunch of :math:`(u,v)`-pairs"""
    us = rand.uniform(0.0, 1.0, N)
    vs = rand.uniform(0.0, 1.0, N)
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
    P_list = P.tolist()
    start_py = time.perf_counter()
    for (u, v) in zip(us, vs):
        nurbs_purepython.bezier_surf_eval(P_list, u, v)
    end_py = time.perf_counter()

    start_rust = time.perf_counter()
    for (u, v) in zip(us, vs):
        rust_nurbs.bezier_surf_eval(P, u, v)
    end_rust = time.perf_counter()

    return [end_py - start_py, end_rust - start_rust], "Case 4"


def case_5(N: int) -> (typing.List[float], str):
    """High-resolution Bezier surface grid evaluation"""
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
    P_list = P.tolist()
    Nu, Nv = N, N
    start_py = time.perf_counter()
    nurbs_purepython.bezier_surf_eval_grid(P_list, Nu, Nv)
    end_py = time.perf_counter()

    start_rust = time.perf_counter()
    rust_nurbs.bezier_surf_eval_grid(P, Nu, Nv)
    end_rust = time.perf_counter()

    return [end_py - start_py, end_rust - start_rust], "Case 5"


def case_6(N: int) -> (typing.List[float], str):
    """High-resolution NURBS surface of revolution grid evaluation"""
    bez_P = np.array([
        [0.0, 1.0, 0.0],
        [0.2, 1.5, 0.0],
        [0.5, 0.8, 0.0],
        [0.8, 1.2, 0.0],
        [1.0, 1.1, 0.0]
    ])
    bez = ac.BezierCurve3D(bez_P)
    ax = ac.Line3D(p0=ac.Point3D.from_array(np.array([0.0, 0.0, 0.0])),
                   p1=ac.Point3D.from_array(np.array([1.0, 0.0, 0.0])))
    Nu, Nv = N, N
    surf = ac.NURBSSurface.from_bezier_revolve(bez, ax, start_angle=ac.Angle(deg=0.0), end_angle=ac.Angle(deg=360.0))
    P = surf.get_control_point_array()
    P_list = P.tolist()
    w = surf.weights
    w_list = w.tolist()
    ku = surf.knots_u
    ku_list = ku.tolist()
    kv = surf.knots_v
    kv_list = kv.tolist()

    start_py = time.perf_counter()
    nurbs_purepython.nurbs_surf_eval_grid(P_list, w_list, ku_list, kv_list, Nu, Nv)
    end_py = time.perf_counter()

    start_rust = time.perf_counter()
    rust_nurbs.nurbs_surf_eval_grid(P, w, ku, kv, Nu, Nv)
    end_rust = time.perf_counter()

    return [end_py - start_py, end_rust - start_rust], "Case 6"


def case_7(N: int) -> (typing.List[float], str):
    """High-resolution NURBS surface of revolution grid evaluation with higher degree in the :math:`v`-direction"""
    bez_P = np.array([
        [0.0, 1.0, 0.0],
        [0.2, 1.5, 0.0],
        [0.3, 1.2, 0.0],
        [0.4, 1.3, 0.0],
        [0.5, 0.8, 0.0],
        [0.6, 0.9, 0.0],
        [0.7, 0.8, 0.0],
        [0.8, 1.2, 0.0],
        [0.9, 1.1, 0.0],
        [1.0, 1.1, 0.0]
    ])
    bez = ac.BezierCurve3D(bez_P)
    ax = ac.Line3D(p0=ac.Point3D.from_array(np.array([0.0, 0.0, 0.0])),
                   p1=ac.Point3D.from_array(np.array([1.0, 0.0, 0.0])))
    Nu, Nv = N, N
    surf = ac.NURBSSurface.from_bezier_revolve(bez, ax, start_angle=ac.Angle(deg=0.0), end_angle=ac.Angle(deg=360.0))
    P = surf.get_control_point_array()
    P_list = P.tolist()
    w = surf.weights
    w_list = w.tolist()
    ku = surf.knots_u
    ku_list = ku.tolist()
    kv = surf.knots_v
    kv_list = kv.tolist()

    start_py = time.perf_counter()
    nurbs_purepython.nurbs_surf_eval_grid(P_list, w_list, ku_list, kv_list, Nu, Nv)
    end_py = time.perf_counter()

    start_rust = time.perf_counter()
    rust_nurbs.nurbs_surf_eval_grid(P, w, ku, kv, Nu, Nv)
    end_rust = time.perf_counter()

    return [end_py - start_py, end_rust - start_rust], "Case 7"


def main():

    sizes = [500000, 500000, 500000, 500000, 500, 500, 500]
    case_names = []
    time_arr = []
    for i, size in enumerate(sizes, start=1):
        times, name = globals()[f"case_{i}"](size)
        print(f"Completed {name}. Pure-Python: {times[0]:.3f} seconds. Rust-Python: {times[1]:.3f} seconds.")
        case_names.append(name)
        time_arr.append(times)

    relative_times = [[t[0] / t[1], 1.0] for t in time_arr]

    x = np.arange(len(case_names))
    width = 0.35

    fig, ax = plt.subplots()
    plt.grid(axis="y", ls=":")
    ax.bar(x - width / 2, [t[0] for t in relative_times], width, label="Pure Python")
    ax.bar(x + width / 2, [t[1] for t in relative_times], width, label="Rust-Python")

    ax.set_ylabel("Time Relative to Rust-Python", font=dict(size=16))
    for tick in ax.get_yticklabels():
        tick.set_fontsize(14)
    ax.set_xticks(x)
    ax.set_xticklabels(case_names, font=dict(size=14))
    ax.legend(prop=dict(size=16))

    fig.set_tight_layout(True)
    # fig.savefig("python-rust-speed-comparison.png", dpi=300, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()
