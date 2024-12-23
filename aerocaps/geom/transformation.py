import typing

import numpy as np
import numpy.linalg as la

from aerocaps.geom.vector import Vector3D

__all__ = [
    "Transformation2D",
    "Transformation3D",
    "transform_points_into_coordinate_system",
]


class Transformation2D:
    def __init__(self, tx: list = 0.0, ty: list = 0.0, r: list = 0.0, sx: list = 1.0,
                 sy: list = 1.0, rotation_units: str = 'rad', order='r,s,t'):
        """
        Allows for arbitrary 2D transformations on a set of coordinates of size 2 x N"""

        self.tx = tx
        self.ty = ty
        self.r = r
        self.sx = sx
        self.sy = sy

        if rotation_units == 'rad':
            self.r_rad = self.r
            self.r_deg = np.rad2deg(self.r)
        elif rotation_units == 'deg':
            self.r_deg = self.r
            self.r_rad = np.deg2rad(self.r)

        self.order = order

        self.r_mat = None
        self.s_mat = None
        self.t_mat = None

        self.M = np.eye(3)  # 3 x 3 identity matrix

        self.r_mat = self.generate_rotation_matrix()
        self.s_mat = self.generate_scale_matrix()
        self.t_mat = self.generate_translation_matrix()
        self.generate_transformation_matrix()

    def generate_rotation_matrix(self) -> np.ndarray:
        return np.array([[np.cos(self.r), -np.sin(self.r), 0],
                        [np.sin(self.r), np.cos(self.r), 0],
                        [0, 0, 1]])

    def generate_scale_matrix(self) -> np.ndarray:
        return np.array([[self.sx, 0, 0],
                         [0, self.sy, 0],
                         [0, 0, 1]])

    def generate_translation_matrix(self) -> np.ndarray:
        return np.array([[1, 0, self.tx],
                         [0, 1, self.ty],
                         [0, 0, 1]])

    def generate_transformation_matrix(self):
        for idx, operation in enumerate(self.order.split(',')):
            if operation == 'r':
                self.M = self.M @ self.r_mat.T
            elif operation == 's':
                self.M = self.M @ self.s_mat.T
            elif operation == 't':
                self.M = self.M @ self.t_mat.T
            else:
                raise TransformationError(f'Invalid value \'{operation}\' found in 2-D transformation '
                                          f'(must be \'r\', \'s\', or \'t\'')

    def transform(self, coordinates: np.ndarray):
        r"""
        Computes the transformation of the coordinates.

        Parameters
        ----------
        coordinates: np.ndarray
          Size :math:`N \times 2`, where N is the number of coordinates. The columns represent :math:`x` and :math:`y`.
        """
        return (np.column_stack((coordinates, np.ones(len(coordinates)))) @ self.M)[:, :2]  # x' = x * M


class Transformation3D:
    def __init__(self, tx: float = 0.0, ty: float = 0.0, tz: float = 0.0, rx: float = 0.0, ry: float = 0.0,
                 rz: float = 0.0, sx: float = 1.0, sy: float = 1.0, sz: float = 1.0,
                 rotation_units: str = 'rad', order='rx,ry,rz,s,t'):
        r"""
        Allows for arbitrary 3D transformations on a set of coordinates of size :math:`3 \times N`
        """

        self.tx = tx
        self.ty = ty
        self.tz = tz
        self.rx = rx
        self.ry = ry
        self.rz = rz
        self.sx = sx
        self.sy = sy
        self.sz = sz

        if rotation_units == 'rad':
            self.rx_rad = self.rx
            self.rx_deg = np.rad2deg(self.rx)
            self.ry_rad = self.ry
            self.ry_deg = np.rad2deg(self.ry)
            self.rz_rad = self.rz
            self.rz_deg = np.rad2deg(self.rz)
        elif rotation_units == 'deg':
            self.rx_deg = self.rx
            self.rx_rad = np.deg2rad(self.rx)
            self.ry_deg = self.ry
            self.ry_rad = np.deg2rad(self.ry)
            self.rz_deg = self.rz
            self.rz_rad = np.deg2rad(self.rz)

        self.order = order.split(",")

        self.rx_mat = None
        self.ry_mat = None
        self.rz_mat = None
        self.s_mat = None
        self.t_mat = None

        self.M = np.eye(4)  # 4 x 4 identity matrix

        self.rx_mat = self.generate_rotation_matrix_x()
        self.ry_mat = self.generate_rotation_matrix_y()
        self.rz_mat = self.generate_rotation_matrix_z()
        self.s_mat = self.generate_scale_matrix()
        self.t_mat = self.generate_translation_matrix()
        self.generate_transformation_matrix()

    def generate_rotation_matrix_x(self) -> np.ndarray:
        return np.array([[1, 0, 0, 0],
                         [0, np.cos(self.rx_rad), np.sin(self.rx_rad), 0],
                         [0, -np.sin(self.rx_rad), np.cos(self.rx_rad), 0],
                         [0, 0, 0, 1]])

    def generate_rotation_matrix_y(self) -> np.ndarray:
        return np.array([[np.cos(self.ry_rad), 0, -np.sin(self.ry_rad), 0],
                         [0, 1, 0, 0],
                         [np.sin(self.ry_rad), 0, np.cos(self.ry_rad), 0],
                         [0, 0, 0, 1]])

    def generate_rotation_matrix_z(self) -> np.ndarray:
        return np.array([[np.cos(self.rz_rad), -np.sin(self.rz_rad), 0, 0],
                         [np.sin(self.rz_rad), np.cos(self.rz_rad), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

    def generate_scale_matrix(self) -> np.ndarray:
        return np.array([[self.sx, 0, 0, 0],
                         [0, self.sy, 0, 0],
                         [0, 0, self.sz, 0],
                         [0, 0, 0, 1]])

    def generate_translation_matrix(self) -> np.ndarray:
        return np.array([[1, 0, 0, self.tx],
                         [0, 1, 0, self.ty],
                         [0, 0, 1, self.tz],
                         [0, 0, 0, 1]])

    def generate_transformation_matrix(self):
        for idx, operation in enumerate(self.order):
            if operation == 'rx':
                self.M = self.M @ self.rx_mat.T
            elif operation == 'ry':
                self.M = self.M @ self.ry_mat.T
            elif operation == 'rz':
                self.M = self.M @ self.rz_mat.T
            elif operation == 's':
                self.M = self.M @ self.s_mat.T
            elif operation == 't':
                self.M = self.M @ self.t_mat.T
            else:
                raise TransformationError(f'Invalid value \'{operation}\' found in 3-D transformation order '
                                          f'(must be \'rx\', \'ry\', \'rz\', \'s\', or \'t\'')

    def transform(self, coordinates: np.ndarray):
        """
        Computes the transformation of the coordinates.

        Parameters
        ----------
        coordinates: np.ndarray
          Size N x 3, where N is the number of coordinates. The columns represent :math:`x` and :math:`y`.
        """
        return (np.column_stack((coordinates, np.ones(len(coordinates)))) @ self.M)[:, :3]  # x' = x * M


def _convert_list_of_csys_vectors_to_homogeneous_matrix(csys_vectors: typing.List[Vector3D]) -> np.ndarray:
    r"""
    Converts a list of unit vectors describing a coordinate system to a matrix using homogeneous coordinates.

    Parameters
    ----------
    csys_vectors: typing.List[Vector3D]
        Vectors describing the coordinate system. Should be in the order :math:`(i,j,k)`. The vectors are normalized
        automatically.

    Returns
    -------
    np.ndarray
        Homogeneous matrix of size :math:`4 \times 4`
    """
    mat = np.zeros((4, 4))
    mat[:3, 0] = csys_vectors[0].get_normalized_vector().as_array()
    mat[:3, 1] = csys_vectors[1].get_normalized_vector().as_array()
    mat[:3, 2] = csys_vectors[2].get_normalized_vector().as_array()
    mat[3, 3] = 1.0
    return mat


def transform_points_into_coordinate_system(points_in_current_csys: np.ndarray,
                                            current_csys: typing.List[Vector3D],
                                            primed_csys: typing.List[Vector3D]) -> np.ndarray:
    r"""
    Transforms a set of points from one coordinate system into another.

    Parameters
    ----------
    points_in_current_csys: np.ndarray
        Array of points, :math:`N \times 3`
    current_csys: typing.List[Vector3D]
        List of vector objects describing the coordinate system the points currently reside in. Should be in
        the order :math:`(i,j,k)`. The vectors are normalized automatically by the code.
    primed_csys: typing.List[Vector3D]
        List of vector objects describing the coordinate system into which the points should be transformed.
        Should be in the order :math:`(i,j,k)`. The vectors are normalized automatically by the code.

    Returns
    -------
    np.ndarray
        Transformed points
    """
    mat_current = _convert_list_of_csys_vectors_to_homogeneous_matrix(current_csys)
    mat_prime = _convert_list_of_csys_vectors_to_homogeneous_matrix(primed_csys)
    mat_current_inverse = la.inv(mat_current)
    # homogeneous_transposed_points = np.column_stack((
    #     points_in_current_csys, np.ones(points_in_current_csys.shape[0])
    # )).T
    homogeneous_transposed_points = np.insert(
        points_in_current_csys, 3, 1.0, axis=1
    ).T
    homogeneous_transformed_points = mat_prime @ (mat_current_inverse @ homogeneous_transposed_points)
    return homogeneous_transformed_points.T[:, :3]


class TransformationError(Exception):
    pass
