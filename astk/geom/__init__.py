from abc import abstractmethod

import numpy as np

import astk.iges.entity


class Geometry2D:
    pass


class Geometry3D:
    @abstractmethod
    def to_iges(self, *args, **kwargs) -> astk.iges.entity.IGESEntity:
        pass


class Surface(Geometry3D):
    @abstractmethod
    def evaluate(self, Nu: int, Nv: int) -> np.ndarray:
        pass


class InvalidGeometryError(Exception):
    pass


class NegativeWeightError(Exception):
    pass
