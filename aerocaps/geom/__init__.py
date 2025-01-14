from abc import abstractmethod

import numpy as np

import aerocaps.iges.entity


class Geometry2D:
    pass


class Geometry3D:
    @abstractmethod
    def to_iges(self, *args, **kwargs) -> aerocaps.iges.entity.IGESEntity:
        """
        Converts the geometric object to an IGES entity. To add this IGES entity to an ``.igs`` file,
        use an :obj:`~aerocaps.iges.iges_generator.IGESGenerator`.
        """
        pass


class Surface(Geometry3D):
    @abstractmethod
    def evaluate(self, u: float, v: float) -> np.ndarray:
        pass

    @abstractmethod
    def evaluate_point3d(self, u: float, v: float):
        pass

    @abstractmethod
    def evaluate_grid(self, Nu: int, Nv: int) -> np.ndarray:
        pass


class InvalidGeometryError(Exception):
    pass


class NegativeWeightError(Exception):
    pass
