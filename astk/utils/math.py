from math import factorial
from decimal import Decimal

import numpy as np


def nchoosek(n: int, k: int):
    r"""
    Computes the mathematical combination

    .. math::

      n \choose k

    Parameters
    ==========
    n: int
      Number of elements in the set

    k: int
      Number of items to select from the set

    Returns
    =======
    :math:`n \choose k`
    """

    return float(Decimal(factorial(n)) / Decimal(factorial(k)) / Decimal(factorial(n - k)))



def bernstein_poly(n: int, i: int, t: int or float or np.ndarray):
    r"""
    Calculates the Bernstein polynomial for a given Bézier curve order, index, and parameter vector. The
    Bernstein polynomial is described by

    .. math::

        B_{i,n}(t)={n \choose i} t^i (1-t)^{n-i}

    Arguments
    =========
    n: int
        Bézier curve degree (one less than the number of control points in the Bézier curve)
    i: int
        Bézier curve index
    t: int, float, or np.ndarray
        Parameter vector for the Bézier curve

    Returns
    =======
    np.ndarray
        Array of values of the Bernstein polynomial evaluated for each point in the parameter vector
    """
    if not 0 <= i <= n:
        return 0.0 if isinstance(t, float) else np.zeros(t.shape)
    return nchoosek(n, i) * t ** i * (1.0 - t) ** (n - i)
