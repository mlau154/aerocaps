"""
Pure-python implementation of NURBS evaluation (no :obj:`numpy`).

.. warning::

    The functions in this module are purely for comparison purposes and calling of these functions from higher-level
    functions or methods is discouraged since the much faster :obj:`rust_nurbs` library is available.
"""
from decimal import Decimal
from math import factorial
from typing import List

__all__ = [
    "bernstein_poly",
    "bezier_curve_eval",
    "bezier_surf_eval",
    "bezier_surf_eval_grid",
    "rational_bezier_curve_eval",
    "rational_bezier_surf_eval",
    "rational_bezier_surf_eval_grid",
    "bspline_curve_eval",
    "bspline_surf_eval",
    "bspline_surf_eval_grid",
    "nurbs_curve_eval",
    "nurbs_surf_eval",
    "nurbs_surf_eval_grid"
]


def nchoosek(n: int, k: int):
    r"""
    Computes the mathematical combination

    .. math::

      n \choose k

    Parameters
    ----------
    n: int
      Number of elements in the set
    k: int
      Number of items to select from the set

    Returns
    -------
    :math:`n \choose k`
    """

    return float(Decimal(factorial(n)) / Decimal(factorial(k)) / Decimal(factorial(n - k)))


def bernstein_poly(n: int, i: int, t: float) -> float:
    r"""
    Evaluates the Bernstein polynomial at a single :math:`t`-value. The Bernstein polynomial is given by

    .. math::

        B_{i,n}(t)={n \choose i} t^i (1-t)^{n-i}

    Parameters
    ----------
    n: int
        Degree of the polynomial
    i: int
        Index
    t: float
        Parameter value :math:`t` at which to evaluate

    Returns
    -------
    float
        Value of the Bernstein polynomial at :math:`t`
    """
    if not 0 <= i <= n:
        return 0.0
    return nchoosek(n, i) * t ** i * (1.0 - t) ** (n - i)


def bezier_curve_eval(p: List[List[float]], t: float) -> List[float]:
    r"""
    Evaluates a Bézier curve with :math:`n+1` control points at a single :math:`t`-value according to

    .. math::

        \mathbf{C}(t) = \sum\limits_{i=0}^n B_{i,n}(t) \mathbf{P}_i

    where :math:`B_{i,n}(t)` is the Bernstein polynomial.

    Parameters
    ----------
    p: List[List[float]]
        2-D list or array of control points where the inner dimension can have any size, but typical
        sizes include ``2`` (:math:`x`-:math:`y` space), ``3`` (:math:`x`-:math:`y`-:math:`z` space) and
        ``4`` (:math:`x`-:math:`y`-:math:`z`-:math:`w` space)
    t: float
        Parameter value :math:`t` at which to evaluate

    Returns
    -------
    List[float]
        Value of the Bézier curve at :math:`t`. Has the same size as the inner dimension of ``p``
    """
    n = len(p) - 1
    dim = len(p[0])
    evaluated_point = [0.0] * dim
    for i in range(n + 1):
        b_poly = bernstein_poly(n, i, t)
        for j in range(dim):
            evaluated_point[j] += p[i][j] * b_poly
    return evaluated_point


def bezier_surf_eval(p: List[List[List[float]]], u: float, v: float) -> List[float]:
    r"""
    Evaluates a Bézier surface with :math:`n+1` control points in the :math:`u`-direction
    and :math:`m+1` control points in the :math:`v`-direction at a :math:`(u,v)` parameter pair according to

    .. math::

        \mathbf{S}(u,v) = \sum\limits_{i=0}^n \sum\limits_{j=0}^m B_{i,n}(u) B_{j,m}(v) \mathbf{P}_{i,j}

    Parameters
    ----------
    p: List[List[List[float]]]
        3-D list or array of control points where the innermost dimension can have any size, but typical
        sizes include ``2`` (:math:`x`-:math:`y` space), ``3`` (:math:`x`-:math:`y`-:math:`z` space) and
        ``4`` (:math:`x`-:math:`y`-:math:`z`-:math:`w` space)
    u: float
        Parameter value in the :math:`u`-direction at which to evaluate the surface
    v: float
        Parameter value in the :math:`v`-direction at which to evaluate the surface

    Returns
    -------
    List[float]
        Value of the Bézier surface at :math:`(u,v)`. Has the same size as the innermost dimension of ``p``
    """
    n = len(p) - 1
    m = len(p[0]) - 1
    dim = len(p[0][0])
    evaluated_point = [0.0] * dim
    for i in range(n + 1):
        b_poly_u = bernstein_poly(n, i, u)
        for j in range(m + 1):
            b_poly_v = bernstein_poly(m, j, v)
            b_poly_prod = b_poly_u * b_poly_v
            for k in range(dim):
                evaluated_point[k] += p[i][j][k] * b_poly_prod
    return evaluated_point


def bezier_surf_eval_grid(p: List[List[List[float]]], nu: int, nv: int) -> List[List[List[float]]]:
    r"""
    Evaluates a Bézier surface with :math:`n+1` control points in the :math:`u`-direction
    and :math:`m+1` control points in the :math:`v`-direction at :math:`N_u \times N_v` points
    along a linearly-spaced rectangular grid in :math:`(u,v)`-space according to

    .. math::

        \mathbf{S}(u,v) = \sum\limits_{i=0}^n \sum\limits_{j=0}^m B_{i,n}(u) B_{j,m}(v) \mathbf{P}_{i,j}

    Parameters
    ----------
    p: List[List[List[float]]]
        3-D list or array of control points where the innermost dimension can have any size, but typical
        sizes include ``2`` (:math:`x`-:math:`y` space), ``3`` (:math:`x`-:math:`y`-:math:`z` space) and
        ``4`` (:math:`x`-:math:`y`-:math:`z`-:math:`w` space)
    nu: int
        Number of linearly-spaced points in the :math:`u`-direction. E.g., ``nu=3`` outputs
        the evaluation of the surface at :math:`u=0.0`, :math:`u=0.5`, and :math:`u=1.0`.
    nv: int
        Number of linearly-spaced points in the :math:`v`-direction. E.g., ``nv=3`` outputs
        the evaluation of the surface at :math:`v=0.0`, :math:`v=0.5`, and :math:`v=1.0`.

    Returns
    -------
    List[List[List[float]]]
        Values of :math:`N_u \times N_v` points on the Bézier surface at :math:`(u,v)`.
        Output array has size :math:`N_u \times N_v \times d`, where :math:`d` is the spatial dimension
        (usually either ``2``, ``3``, or ``4``)
    """
    n = len(p) - 1
    m = len(p[0]) - 1
    dim = len(p[0][0])
    evaluated_points = [[[0.0] * dim] * nv] * nu
    for u_idx in range(nu):
        u = float(u_idx) * 1.0 / (float(nu) - 1.0)
        for v_idx in range(nv):
            v = float(v_idx) * 1.0 / (float(nv) - 1.0)
            for i in range(n + 1):
                b_poly_u = bernstein_poly(n, i, u)
                for j in range(m + 1):
                    b_poly_v = bernstein_poly(m, j, v)
                    b_poly_prod = b_poly_u * b_poly_v
                    for k in range(dim):
                        evaluated_points[u_idx][v_idx][k] += p[i][j][k] * b_poly_prod
    return evaluated_points


def rational_bezier_curve_eval(p: List[List[float]], w: List[float], t: float) -> List[float]:
    r"""
    Evaluates a rational Bézier curve with :math:`n+1` control points at a single :math:`t`-value according to

    .. math::

        \mathbf{C}(t) = \frac{\sum_{i=0}^n B_{i,n}(t) w_i \mathbf{P}_i}{\sum_{i=0}^n B_{i,n}(t) w_i}

    where :math:`B_{i,n}(t)` is the Bernstein polynomial.

    Parameters
    ----------
    p: List[List[float]]
        2-D list or array of control points where the inner dimension can have any size, but the typical
        size is ``3`` (:math:`x`-:math:`y`-:math:`z` space)
    w: List[float]
        1-D list or array of weights corresponding to each of control points. Must have length
        equal to the outer dimension of ``p``.
    t: float
        Parameter value :math:`t` at which to evaluate

    Returns
    -------
    List[float]
        Value of the rational Bézier curve at :math:`t`. Has the same size as the inner dimension of ``p``
    """
    n = len(p) - 1
    dim = len(p[0])
    evaluated_point = [0.0] * dim
    w_sum = 0.0
    for i in range(n + 1):
        b_poly = bernstein_poly(n, i, t)
        w_sum += w[i] * b_poly
        for j in range(dim):
            evaluated_point[j] += p[i][j] * w[i] * b_poly
    for j in range(dim):
        evaluated_point[j] /= w_sum
    return evaluated_point


def rational_bezier_surf_eval(p: List[List[List[float]]], w: List[List[float]], u: float, v: float) -> List[float]:
    r"""
    Evaluates a rational Bézier surface with :math:`n+1` control points in the :math:`u`-direction
    and :math:`m+1` control points in the :math:`v`-direction at a :math:`(u,v)` parameter pair according to

    .. math::

        \mathbf{S}(u,v) = \frac{\sum_{i=0}^n \sum_{j=0}^m B_{i,n}(u) B_{j,m}(v) w_{i,j} \mathbf{P}_{i,j}}{\sum_{i=0}^n \sum_{j=0}^m B_{i,n}(u) B_{j,m}(v) w_{i,j}}

    Parameters
    ----------
    p: List[List[List[float]]]
        3-D list or array of control points where the innermost dimension can have any size, but the typical
        size is ``3`` (:math:`x`-:math:`y`-:math:`z` space)
    w: List[List[float]]
        2-D list or array of weights corresponding to each of control points. The size of the array must be
        equal to the size of the first two dimensions of ``p`` (:math:`n+1 \times m+1`)
    u: float
        Parameter value in the :math:`u`-direction at which to evaluate the surface
    v: float
        Parameter value in the :math:`v`-direction at which to evaluate the surface

    Returns
    -------
    List[float]
        Value of the rational Bézier surface at :math:`(u,v)`. Has the same size as the innermost dimension of ``p``
    """
    n = len(p) - 1
    m = len(p[0]) - 1
    dim = len(p[0][0])
    evaluated_point = [0.0] * dim
    w_sum = 0.0
    for i in range(n + 1):
        b_poly_u = bernstein_poly(n, i, u)
        for j in range(m + 1):
            b_poly_v = bernstein_poly(m, j, v)
            b_poly_prod = b_poly_u * b_poly_v
            w_sum += w[i][j] * b_poly_prod
            for k in range(dim):
                evaluated_point[k] += p[i][j][k] * w[i][j] * b_poly_prod
    for k in range(dim):
        evaluated_point[k] /= w_sum
    return evaluated_point


def rational_bezier_surf_eval_grid(p: List[List[List[float]]],
                                   w: List[List[float]], nu: int, nv: int) -> List[List[List[float]]]:
    r"""
    Evaluates a rational Bézier surface with :math:`n+1` control points in the :math:`u`-direction
    and :math:`m+1` control points in the :math:`v`-direction at :math:`N_u \times N_v` points along a
    linearly-spaced rectangular grid in :math:`(u,v)`-space according to

    .. math::

        \mathbf{S}(u,v) = \frac{\sum_{i=0}^n \sum_{j=0}^m B_{i,n}(u) B_{j,m}(v) w_{i,j} \mathbf{P}_{i,j}}{\sum_{i=0}^n \sum_{j=0}^m B_{i,n}(u) B_{j,m}(v) w_{i,j}}

    Parameters
    ----------
    p: List[List[List[float]]]
        3-D list or array of control points where the innermost dimension can have any size, but the typical
        size is ``3`` (:math:`x`-:math:`y`-:math:`z` space)
    w: List[List[float]]
        2-D list or array of weights corresponding to each of control points. The size of the array must be
        equal to the size of the first two dimensions of ``p`` (:math:`n+1 \times m+1`)
    nu: int
        Number of linearly-spaced points in the :math:`u`-direction. E.g., ``nu=3`` outputs
        the evaluation of the surface at :math:`u=0.0`, :math:`u=0.5`, and :math:`u=1.0`.
    nv: int
        Number of linearly-spaced points in the :math:`v`-direction. E.g., ``nv=3`` outputs
        the evaluation of the surface at :math:`v=0.0`, :math:`v=0.5`, and :math:`v=1.0`.

    Returns
    -------
    List[List[List[float]]]
        Values of :math:`N_u \times N_v` points on the rational Bézier surface at :math:`(u,v)`.
        Output array has size :math:`N_u \times N_v \times d`, where :math:`d` is the spatial dimension
        (usually ``3``)
    """
    n = len(p) - 1
    m = len(p[0]) - 1
    dim = len(p[0][0])
    evaluated_points = [[[0.0] * dim] * nv] * nu
    for u_idx in range(nu):
        u = float(u_idx) * 1.0 / (float(nu) - 1.0)
        for v_idx in range(nv):
            v = float(v_idx) * 1.0 / (float(nv) - 1.0)
            w_sum = 0.0
            for i in range(n + 1):
                b_poly_u = bernstein_poly(n, i, u)
                for j in range(m + 1):
                    b_poly_v = bernstein_poly(m, j, v)
                    b_poly_prod = b_poly_u * b_poly_v
                    w_sum += w[i][j] * b_poly_prod
                    for k in range(dim):
                        evaluated_points[u_idx][v_idx][k] += p[i][j][k] * w[i][j] * b_poly_prod
            for k in range(dim):
                evaluated_points[u_idx][v_idx][k] /= w_sum
    return evaluated_points


def _get_possible_span_indices(k: List[float]) -> List[int]:
    """
    Gets the list of possible knot span indices (those that have a non-zero width)

    Parameters
    ----------
    k: List[float]
        Knot vector

    Returns
    -------
    List[int]
        Possible knot span indices (each value represents the index at the start of the knot span)
    """
    possible_span_indices = []
    num_knots = len(k)
    for i in range(num_knots - 1):
        if k[i] == k[i + 1]:
            continue
        possible_span_indices.append(i)
    return possible_span_indices


def _find_span(k: List[float], possible_span_indices: List[int], t: float) -> int:
    """
    Finds the knot span on which the parameter value :math:`t` lies

    Parameters
    ----------
    k: List[float]
        Knot vector
    possible_span_indices: List[int]
        List of possible span indices along which :math:`t` can lie (usually called from ``_get_possible_span_indices``)
    t: float
        Parameter value :math:`t`

    Returns
    -------
    int
        Index corresponding to the start of the knot span on which the parameter value :math:`t` lies
    """
    for knot_span_idx in possible_span_indices:
        if k[knot_span_idx] <= t < k[knot_span_idx + 1]:
            return knot_span_idx
    if t == k[-1]:
        return possible_span_indices[-1]
    raise ValueError(f"Parameter value {t = } out of bounds for knot vector with "
                     f"first knot {k[0]} and last knot {k[-1]}")


def _cox_de_boor(k: List[float], possible_span_indices: List[int], degree: int, i: int, t: float) -> float:
    """
    Implements the Cox de Boor algorithm for computing the B-spline basis function

    Parameters
    ----------
    k: List[float]
        Knot vector
    possible_span_indices: List[int]
        List of possible span indices along which :math:`t` can lie (usually called from ``_get_possible_span_indices``)
    degree: int
        B-spline basis function degree
    i: int
        B-spline basis index
    t: float
        Parameter value :math:`t`

    Returns
    -------
    float
        Value of the B-spline basis function at a particular value of :math:`t`
    """
    if degree == 0:
        if i in possible_span_indices and _find_span(k, possible_span_indices, t) == i:
            return 1.0
        return 0.0
    f = 0.0
    g = 0.0
    if k[i + degree] - k[i] != 0.0:
        f = (t - k[i]) / (k[i + degree + 1])
    if k[i + degree + 1] - k[i + 1] != 0.0:
        g = (k[i + degree + 1] - t) / (k[i + degree + 1] - k[i + 1])
    if f == 0.0 and g == 0.0:
        return 0.0
    if g == 0.0:
        return f * _cox_de_boor(k, possible_span_indices, degree - 1, i, t)
    if f == 0.0:
        return g * _cox_de_boor(k, possible_span_indices, degree - 1, i + 1, t)
    return f * _cox_de_boor(k, possible_span_indices, degree - 1, i, t) + g * _cox_de_boor(
        k, possible_span_indices, degree - 1, i + 1, t)


def bspline_curve_eval(p: List[List[float]], k: List[float], t: float) -> List[float]:
    r"""
    Evaluates a B-spline curve with :math:`n+1` control points at a single :math:`t`-value according to

    .. math::

        \mathbf{C}(t) = \sum\limits_{i=0}^n N_{i,q}(t) \mathbf{P}_i

    where :math:`N_{i,q}(t)` is the B-spline basis function of degree :math:`q`, defined recursively as

    .. math::

        N_{i,q} = \frac{t - t_i}{t_{i+q} - t_i} N_{i,q-1}(t) + \frac{t_{i+q+1} - t}{t_{i+q+1} - t_{i+1}} N_{i+1, q-1}(t)

    with base case

    .. math::

        N_{i,0} = \begin{cases}
            1, & \text{if } t_i \leq t < t_{i+1} \text{ and } t_i < t_{i+1} \\
            0, & \text{otherwise}
        \end{cases}

    The degree of the B-spline is computed as ``q = k.len() - len(p) - 1``.

    Parameters
    ----------
    p: List[List[float]]
        2-D list or array of control points where the inner dimension can have any size, but the typical
        size is ``3`` (:math:`x`-:math:`y`-:math:`z` space)
    k: List[float]
        1-D list or array of knots
    t: float
        Parameter value :math:`t` at which to evaluate

    Returns
    -------
    List[float]
        Value of the B-spline curve at :math:`t`. Has the same size as the inner dimension of ``p``
    """
    n = len(p) - 1
    num_knots = len(k)
    q = num_knots - n - 2
    possible_span_indices = _get_possible_span_indices(k)
    dim = len(p[0])
    evaluated_point = [0.0] * dim
    for i in range(n + 1):
        bspline_basis = _cox_de_boor(k, possible_span_indices, q, i, t)
        for j in range(dim):
            evaluated_point[j] += p[i][j] * bspline_basis
    return evaluated_point


def bspline_surf_eval(p: List[List[List[float]]],
                      ku: List[float], kv: List[float], u: float, v: float) -> List[float]:
    r"""
    Evaluates a B-spline surface with :math:`n+1` control points in the :math:`u`-direction
    and :math:`m+1` control points in the :math:`v`-direction at a :math:`(u,v)` parameter pair according to

    .. math::

        \mathbf{S}(u,v) = \sum\limits_{i=0}^n \sum\limits_{j=0}^m N_{i,q}(u) N_{j,r}(v) \mathbf{P}_{i,j}

    where :math:`N_{i,q}(t)` is the B-spline basis function of degree :math:`q`. The degree of the B-spline
    in the :math:`u`-direction is computed as ``q = len(ku) - len(p) - 1``, and the degree of the B-spline
    surface in the :math:`v`-direction is computed as ``r = len(kv) - len(p[0]) - 1``.

    Parameters
    ----------
    p: List[List[List[float]]]
        3-D list or array of control points where the innermost dimension can have any size, but the typical
        size is ``3`` (:math:`x`-:math:`y`-:math:`z` space)
    ku: List[float]
        1-D list or array of knots in the :math:`u`-parametric direction
    kv: List[float]
        1-D list or array of knots in the :math:`v`-parametric direction
    u: float
        Parameter value in the :math:`u`-direction at which to evaluate the surface
    v: float
        Parameter value in the :math:`v`-direction at which to evaluate the surface

    Returns
    -------
    List[float]
        Value of the B-spline surface at :math:`(u,v)`. Has the same size as the innermost dimension of ``p``
    """
    n = len(p) - 1  # Number of control points in the u-direction minus 1
    m = len(p[0]) - 1  # Number of control points in the v-direction minus 1
    num_knots_u = len(ku)  # Number of knots in the u-direction
    num_knots_v = len(kv)  # Number of knots in the v-direction
    q = num_knots_u - n - 2  # Degree in the u-direction
    r = num_knots_v - m - 2  # Degree in the v-direction
    possible_span_indices_u = _get_possible_span_indices(ku)
    possible_span_indices_v = _get_possible_span_indices(kv)
    dim = len(p[0][0])  # Number of spatial dimensions
    evaluated_point = [0.0] * dim
    for i in range(n + 1):
        bspline_basis_u = _cox_de_boor(ku, possible_span_indices_u, q, i, u)
        for j in range(m + 1):
            bspline_basis_v = _cox_de_boor(kv, possible_span_indices_v, r, j, v)
            bspline_basis_prod = bspline_basis_u * bspline_basis_v
            for k in range(dim):
                evaluated_point[k] += p[i][j][k] * bspline_basis_prod
    return evaluated_point


def bspline_surf_eval_grid(p: List[List[List[float]]],
                           ku: List[float], kv: List[float], nu: int, nv: int) -> List[List[List[float]]]:
    r"""
    Evaluates a B-spline surface with :math:`n+1` control points in the :math:`u`-direction
    and :math:`m+1` control points in the :math:`v`-direction at :math:`N_u \times N_v`
    points along a linearly-spaced rectangular grid in :math:`(u,v)`-space according to

    .. math::

        \mathbf{S}(u,v) = \sum\limits_{i=0}^n \sum\limits_{j=0}^m N_{i,q}(u) N_{j,r}(v) \mathbf{P}_{i,j}

    where :math:`N_{i,q}(t)` is the B-spline basis function of degree :math:`q`. The degree of the B-spline
    in the :math:`u`-direction is computed as ``q = len(ku) - len(p) - 1``, and the degree of the B-spline
    surface in the :math:`v`-direction is computed as ``r = len(kv) - len(p[0]) - 1``.

    Parameters
    ----------
    p: List[List[List[float]]]
        3-D list or array of control points where the innermost dimension can have any size, but the typical
        size is ``3`` (:math:`x`-:math:`y`-:math:`z` space)
    ku: List[float]
        1-D list or array of knots in the :math:`u`-parametric direction
    kv: List[float]
        1-D list or array of knots in the :math:`v`-parametric direction
    nu: int
        Number of linearly-spaced points in the :math:`u`-direction. E.g., ``nu=3`` outputs
        the evaluation of the surface at :math:`u=0.0`, :math:`u=0.5`, and :math:`u=1.0`.
    nv: int
        Number of linearly-spaced points in the :math:`v`-direction. E.g., ``nv=3`` outputs
        the evaluation of the surface at :math:`v=0.0`, :math:`v=0.5`, and :math:`v=1.0`.

    Returns
    -------
    List[List[List[float]]]
        Values of :math:`N_u \times N_v` points on the B-spline surface at :math:`(u,v)`.
        Output array has size :math:`N_u \times N_v \times d`, where :math:`d` is the spatial dimension
        (usually ``3``)
    """
    n = len(p) - 1  # Number of control points in the u-direction minus 1
    m = len(p[0]) - 1  # Number of control points in the v-direction minus 1
    num_knots_u = len(ku)  # Number of knots in the u-direction
    num_knots_v = len(kv)  # Number of knots in the v-direction
    q = num_knots_u - n - 2  # Degree in the u-direction
    r = num_knots_v - m - 2  # Degree in the v-direction
    possible_span_indices_u = _get_possible_span_indices(ku)
    possible_span_indices_v = _get_possible_span_indices(kv)
    dim = len(p[0][0])  # Number of spatial dimensions
    evaluated_points = [[[0.0] * dim] * nv] * nu
    for u_idx in range(nu):
        u = float(u_idx) * 1.0 / (float(nu) - 1.0)
        for v_idx in range(nv):
            v = float(v_idx) * 1.0 / (float(nv) - 1.0)
            for i in range(n + 1):
                bspline_basis_u = _cox_de_boor(ku, possible_span_indices_u, q, i, u)
                for j in range(m + 1):
                    bspline_basis_v = _cox_de_boor(kv, possible_span_indices_v, r, j, v)
                    bspline_basis_prod = bspline_basis_u * bspline_basis_v
                    for k in range(dim):
                        evaluated_points[u_idx][v_idx][k] += p[i][j][k] * bspline_basis_prod
    return evaluated_points


def nurbs_curve_eval(p: List[List[float]], w: List[float], k: List[float], t: float) -> List[float]:
    r"""
    Evaluates a Non-Uniform Rational B-Spline (NURBS) curve with :math:`n+1` control points at a
    single :math:`t`-value according to

    .. math::

        \mathbf{C}(t) = \frac{\sum_{i=0}^n N_{i,q}(t) w_i \mathbf{P}_i}{\sum_{i=0}^n N_{i,q}(t) w_i}

    where :math:`N_{i,q}(t)` is the B-spline basis function of degree :math:`q`.
    The degree of the B-spline is computed as ``q = len(k) - len(p) - 1``.

    Parameters
    ----------
    p: List[List[float]]
        2-D list or array of control points where the inner dimension can have any size, but the typical
        size is ``3`` (:math:`x`-:math:`y`-:math:`z` space)
    w: List[float]
        1-D list or array of weights corresponding to each of control points. Must have length
        equal to the outer dimension of ``p``.
    k: List[float]
        1-D list or array of knots
    t: float
        Parameter value :math:`t` at which to evaluate

    Returns
    -------
    List[float]
        Value of the NURBS curve at :math:`t`. Has the same size as the inner dimension of ``p``
    """
    n = len(p) - 1  # Number of control points minus 1
    num_knots = len(k)
    q = num_knots - n - 2
    possible_span_indices = _get_possible_span_indices(k)
    dim = len(p[0])
    evaluated_point = [0.0] * dim
    w_sum = 0.0
    for i in range(n + 1):
        bspline_basis = _cox_de_boor(k, possible_span_indices, q, i, t)
        w_sum += w[i] * bspline_basis
        for j in range(dim):
            evaluated_point[j] += p[i][j] * w[i] * bspline_basis
    for j in range(dim):
        evaluated_point[j] /= w_sum
    return evaluated_point


def nurbs_surf_eval(p: List[List[List[float]]], w: List[List[float]],
                    ku: List[float], kv: List[float], u: float, v: float) -> List[float]:
    r"""
    Evaluates a Non-Uniform Rational B-Spline (NURBS) surface with :math:`n+1` control points in the :math:`u`-direction
    and :math:`m+1` control points in the :math:`v`-direction at a :math:`(u,v)` parameter pair according to

    .. math::

        \mathbf{S}(u,v) = \frac{\sum_{i=0}^n \sum_{j=0}^m N_{i,q}(u) N_{j,r}(v) w_{i,j} \mathbf{P}_{i,j}}{\sum_{i=0}^n \sum_{j=0}^m N_{i,q}(u) N_{j,r}(v) w_{i,j}}

    where :math:`N_{i,q}(t)` is the B-spline basis function of degree :math:`q`. The degree of the B-spline
    in the :math:`u`-direction is computed as ``q = len(ku) - len(p) - 1``, and the degree of the B-spline
    surface in the :math:`v`-direction is computed as ``r = len(kv) - len(p[0]) - 1``.

    Parameters
    ----------
    p: List[List[List[float]]]
        3-D list or array of control points where the innermost dimension can have any size, but the typical
        size is ``3`` (:math:`x`-:math:`y`-:math:`z` space)
    w: List[List[float]]
        2-D list or array of weights corresponding to each of control points. The size of the array must be
        equal to the size of the first two dimensions of ``p`` (:math:`n+1 \times m+1`)
    ku: List[float]
        1-D list or array of knots in the :math:`u`-parametric direction
    kv: List[float]
        1-D list or array of knots in the :math:`v`-parametric direction
    u: float
        Parameter value in the :math:`u`-direction at which to evaluate the surface
    v: float
        Parameter value in the :math:`v`-direction at which to evaluate the surface

    Returns
    -------
    List[float]
        Value of the NURBS surface at :math:`(u,v)`. Has the same size as the innermost dimension of ``p``
    """
    n = len(p) - 1  # Number of control points in the u-direction minus 1
    m = len(p[0]) - 1  # Number of control points in the v-direction minus 1
    num_knots_u = len(ku)  # Number of knots in the u-direction
    num_knots_v = len(kv)  # Number of knots in the v-direction
    q = num_knots_u - n - 2  # Degree in the u-direction
    r = num_knots_v - m - 2  # Degree in the v-direction
    possible_span_indices_u = _get_possible_span_indices(ku)
    possible_span_indices_v = _get_possible_span_indices(kv)
    dim = len(p[0][0])  # Number of spatial dimensions
    evaluated_point = [0.0] * dim
    w_sum = 0.0
    for i in range(n + 1):
        bspline_basis_u = _cox_de_boor(ku, possible_span_indices_u, q, i, u)
        for j in range(m + 1):
            bspline_basis_v = _cox_de_boor(kv, possible_span_indices_v, r, j, v)
            bspline_basis_prod = bspline_basis_u * bspline_basis_v
            w_sum += w[i][j] * bspline_basis_prod
            for k in range(dim):
                evaluated_point[k] += p[i][j][k] * w[i][j] * bspline_basis_prod
    for k in range(dim):
        evaluated_point[k] /= w_sum
    return evaluated_point


def nurbs_surf_eval_grid(p: List[List[List[float]]], w: List[List[float]],
                         ku: List[float], kv: List[float], nu: int, nv: int) -> List[List[List[float]]]:
    r"""
    Evaluates a Non-Uniform Rational B-Spline (NURBS) surface with :math:`n+1` control points in the :math:`u`-direction
    and :math:`m+1` control points in the :math:`v`-direction at :math:`N_u \times N_v`
    points along a linearly-spaced rectangular grid in :math:`(u,v)`-space according to

    .. math::

        \mathbf{S}(u,v) = \frac{\sum_{i=0}^n \sum_{j=0}^m N_{i,q}(u) N_{j,r}(v) w_{i,j} \mathbf{P}_{i,j}}{\sum_{i=0}^n \sum_{j=0}^m N_{i,q}(u) N_{j,r}(v) w_{i,j}}

    where :math:`N_{i,q}(t)` is the B-spline basis function of degree :math:`q`. The degree of the B-spline
    in the :math:`u`-direction is computed as ``q = len(ku) - len(p) - 1``, and the degree of the B-spline
    surface in the :math:`v`-direction is computed as ``r = len(kv) - len(p[0]) - 1``.

    Parameters
    ----------
    p: List[List[List[float]]]
        3-D list or array of control points where the innermost dimension can have any size, but the typical
        size is ``3`` (:math:`x`-:math:`y`-:math:`z` space)
    w: List[List[float]]
        2-D list or array of weights corresponding to each of control points. The size of the array must be
        equal to the size of the first two dimensions of ``p`` (:math:`n+1 \times m+1`)
    ku: List[float]
        1-D list or array of knots in the :math:`u`-parametric direction
    kv: List[float]
        1-D list or array of knots in the :math:`v`-parametric direction
    nu: int
        Number of linearly-spaced points in the :math:`u`-direction. E.g., ``nu=3`` outputs
        the evaluation of the surface at :math:`u=0.0`, :math:`u=0.5`, and :math:`u=1.0`.
    nv: int
        Number of linearly-spaced points in the :math:`v`-direction. E.g., ``nv=3`` outputs
        the evaluation of the surface at :math:`v=0.0`, :math:`v=0.5`, and :math:`v=1.0`.

    Returns
    -------
    List[List[List[float]]]
        Values of :math:`N_u \times N_v` points on the NURBS surface at :math:`(u,v)`.
        Output array has size :math:`N_u \times N_v \times d`, where :math:`d` is the spatial dimension
        (usually ``3``)
    """
    n = len(p) - 1  # Number of control points in the u-direction minus 1
    m = len(p[0]) - 1  # Number of control points in the v-direction minus 1
    num_knots_u = len(ku)  # Number of knots in the u-direction
    num_knots_v = len(kv)  # Number of knots in the v-direction
    q = num_knots_u - n - 2  # Degree in the u-direction
    r = num_knots_v - m - 2  # Degree in the v-direction
    possible_span_indices_u = _get_possible_span_indices(ku)
    possible_span_indices_v = _get_possible_span_indices(kv)
    dim = len(p[0][0])  # Number of spatial dimensions
    evaluated_points = [[[0.0] * dim] * nv] * nu
    for u_idx in range(nu):
        u = float(u_idx) * 1.0 / (float(nu) - 1.0)
        for v_idx in range(nv):
            v = float(v_idx) * 1.0 / (float(nv) - 1.0)
            w_sum = 0.0
            for i in range(n + 1):
                bspline_basis_u = _cox_de_boor(ku, possible_span_indices_u, q, i, u)
                for j in range(m + 1):
                    bspline_basis_v = _cox_de_boor(kv, possible_span_indices_v, r, j, v)
                    bspline_basis_prod = bspline_basis_u * bspline_basis_v
                    w_sum += w[i][j] * bspline_basis_prod
                    for k in range(dim):
                        evaluated_points[u_idx][v_idx][k] += p[i][j][k] * w[i][j] * bspline_basis_prod
            for k in range(dim):
                evaluated_points[u_idx][v_idx][k] /= w_sum
    return evaluated_points
