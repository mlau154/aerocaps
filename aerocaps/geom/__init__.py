from abc import abstractmethod

import numpy as np

import aerocaps.iges.entity


class Geometry:
    """Abstract geometry class"""
    def __init__(self, name: str, construction: bool = False):
        """
        Abstract geometry class

        Parameters
        ----------
        name: str
            Name of the geometric object. May be re-assigned a unique name when added to a
            :obj:`~aerocaps.geom.geometry_container.GeometryContainer`
        construction: bool
            Whether this is a geometry used only for construction of other geometries. If ``True``, this
            geometry will not be exported or plotted. Default: ``False``
        """
        self._name = self._validate_name(name)
        self._construction = construction
        self.container = None  # Geometry container associated with this geometry. This will be assigned after
        # calling GeometryContainer.add_geometry

    @property
    def name(self) -> str:
        """
        Gets the name assigned to the geometry

        Returns
        -------
        str
            Name of the geometry
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """
        Sets the name of the geometry. This name might get modified with a higher index when adding to
        a :obj:`~aerocaps.geom.geometry_container.GeometryContainer`

        Parameters
        ----------
        name: str
            Name to assign
        """
        self._name = self._validate_name(name)

    @staticmethod
    def _validate_name(name: str) -> str:
        """
        Validates that the name is suitable for addition to a
        :obj:`~aerocaps.geom.geometry_container.GeometryContainer`. If not, an exception is raised

        Parameters
        ----------
        name: str
            Name to check

        Returns
        -------
        name: str
            Name of the object if an exception was not raised
        """
        if "-" in name:
            raise ValueError("Name must not contain dashes. Dashes are reserved for GeometryContainer indexing")
        return name

    @property
    def construction(self) -> bool:
        """Whether this is a construction geometry (if so, it will not be plotted or exported)"""
        return self._construction

    @construction.setter
    def construction(self, construction: bool):
        """Sets the construction property"""
        self._construction = construction


class Geometry2D(Geometry):
    """Two-dimensional abstract geometry class"""
    pass


class Geometry3D(Geometry):
    """Three-dimensional abstract geometry class"""
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
