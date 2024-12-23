import numpy as np

from aerocaps.iges.entity import IGESEntity
from aerocaps.iges.iges_param import IGESParam


class PointIGES(IGESEntity):
    def __init__(self, xyz: np.ndarray):
        """
        IGES IGESEntity #116
        """
        parameter_data = [
            IGESParam(xyz[0], "real"),
            IGESParam(xyz[1], "real"),
            IGESParam(xyz[2], "real"),
            IGESParam(0, "pointer")
        ]
        super().__init__(116, parameter_data)
