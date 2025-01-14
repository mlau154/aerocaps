import numpy as np


def unique_with_tolerance(arr: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8) -> np.ndarray:
    """
    Find unique elements in an array with a specified tolerance.
    """
    assert arr.ndim == 2
    unique_values = {0: arr[0]}
    for idx, value in enumerate(arr[1:], start=1):
        for unique_value in unique_values.values():
            if all([np.isclose(v1, v2, rtol=rtol, atol=atol) for v1, v2 in zip(value, unique_value)]):
                break
        else:
            unique_values[idx] = value

    return np.array(list(unique_values.values()))
