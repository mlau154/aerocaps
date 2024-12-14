import typing

import numpy as np

from aerocaps.iges.entity import IGESEntity
from aerocaps.iges.iges_param import IGESParam
from aerocaps.units.angle import Angle
from aerocaps.units.length import Length


class CircularArcIGES(IGESEntity):
    """
    IGES IGESEntity #100
    """
    def __init__(self, radius: Length, start_angle: Angle, end_angle: Angle):
        x2 = radius.m * np.cos(start_angle.rad)
        y2 = radius.m * np.sin(start_angle.rad)
        x3 = radius.m * np.cos(end_angle.rad)
        y3 = radius.m * np.sin(end_angle.rad)
        parameter_data = [
            IGESParam(0.0, "real"),
            IGESParam(0.0, "real"),
            IGESParam(0.0, "real"),
            IGESParam(x2, "real"),
            IGESParam(y2, "real"),
            IGESParam(x3, "real"),
            IGESParam(y3, "real")
        ]
        super().__init__(100, parameter_data)


class LineIGES(IGESEntity):
    """
    IGES IGESEntity #110
    """
    def __init__(self, start_point: np.ndarray, end_point: np.ndarray):
        parameter_data = [
            IGESParam(start_point[0], "real"),
            IGESParam(start_point[1], "real"),
            IGESParam(start_point[2], "real"),
            IGESParam(end_point[0], "real"),
            IGESParam(end_point[1], "real"),
            IGESParam(end_point[2], "real")
        ]
        super().__init__(110, parameter_data)


class RationalBSplineCurveIGES(IGESEntity):
    """
    IGES IGESEntity #126
    """
    def __init__(self, knots: np.ndarray, weights: np.ndarray,
                 control_points_XYZ: np.ndarray, degree: int, start_parameter_value=0.0, end_parameter_value=1.0,
                 unit_normal_x=0.0, unit_normal_y=0.0, unit_normal_z=0.0, planar_flag: bool = False,
                 closed_flag: bool = False, polynomial_flag: bool = False, periodic_flag: bool = False,
                 **entity_kwargs):
        self.upper_index_of_sum = control_points_XYZ.shape[0] - 1
        self.degree = degree
        self.flag1 = int(planar_flag)
        self.flag2 = int(closed_flag)
        self.flag3 = int(polynomial_flag)
        self.flag4 = int(periodic_flag)
        self.knots = knots
        self.weights = weights
        self.control_points = control_points_XYZ
        self.v0 = start_parameter_value
        self.v1 = end_parameter_value
        self.XN = unit_normal_x
        self.YN = unit_normal_y
        self.ZN = unit_normal_z
        parameter_data = [
            IGESParam(self.upper_index_of_sum, "int"),
            IGESParam(self.degree, "int"),
            IGESParam(self.flag1, "int"),
            IGESParam(self.flag2, "int"),
            IGESParam(self.flag3, "int"),
            IGESParam(self.flag4, "int"),
            *[IGESParam(k, "real") for k in self.knots],
            *[IGESParam(w, "real") for w in self.weights],
            *[IGESParam(xyz, "real") for xyz in self.control_points.flatten()],
            IGESParam(self.v0, "real"),
            IGESParam(self.v1, "real"),
            IGESParam(self.XN, "real"),
            IGESParam(self.YN, "real"),
            IGESParam(self.ZN, "real"),
        ]
        super().__init__(126, parameter_data, **entity_kwargs)


class BezierIGES(RationalBSplineCurveIGES):
    def __init__(self, control_points_XYZ: np.ndarray, start_parameter_value=0.0, end_parameter_value=1.0,
                 unit_normal_x=0.0, unit_normal_y=0.0, unit_normal_z=0.0, planar_flag: bool = False,
                 closed_flag: bool = False, periodic_flag: bool = False, **entity_kwargs):
        order = len(control_points_XYZ)
        degree = order - 1
        knots = np.concatenate((np.zeros(order), np.ones(order)))
        weights = np.ones(order)
        polynomial_flag = True
        super().__init__(knots=knots, weights=weights, control_points_XYZ=control_points_XYZ, degree=degree,
                         start_parameter_value=start_parameter_value, end_parameter_value=end_parameter_value,
                         unit_normal_x=unit_normal_x, unit_normal_y=unit_normal_y, unit_normal_z=unit_normal_z,
                         planar_flag=planar_flag, closed_flag=closed_flag, polynomial_flag=polynomial_flag,
                         periodic_flag=periodic_flag, **entity_kwargs)


class BoundaryCurveIGES(IGESEntity):
    def __init__(self,
                 untrimmed_surface: IGESEntity,
                 curves: typing.Dict[IGESEntity, typing.List[IGESEntity]],
                 preferred_representation: int = 3,
                 curves_needing_reversal: typing.List[int] = None,
                 **entity_kwargs
                 ):
        """
        IGES Type 141
        """
        if curves_needing_reversal is None:
            curves_needing_reversal = []
        number_parameter_space_curves = sum([len(parameter_space_curves) for parameter_space_curves in curves.values()])
        parameter_data = [
            IGESParam(0 if number_parameter_space_curves == 0 else 1, "int"),
            IGESParam(preferred_representation, "int"),  # 0 = unspecified, 1 = model space, 2 = parameter space,
                                                         # 3 = representations are of equal preference
            IGESParam(untrimmed_surface, "pointer"),
            IGESParam(len(curves) + number_parameter_space_curves, "int")
        ]
        for curve_idx, (model_space_curve, parameter_space_curves) in enumerate(curves.items()):
            parameter_data.append(IGESParam(model_space_curve, "pointer"))
            parameter_data.append(IGESParam(2 if curve_idx in curves_needing_reversal else 1, "int"))
            parameter_data.append(IGESParam(len(parameter_space_curves), "int"))
            for parameter_space_curve in parameter_space_curves:
                parameter_data.append(IGESParam(parameter_space_curve, "pointer"))

        super().__init__(141, parameter_data, **entity_kwargs)


class CurveOnParametricSurfaceIGES(IGESEntity):
    def __init__(self, surface: IGESEntity, parametric_curve: IGESEntity, model_space_curve: IGESEntity, **entity_kwargs):
        """
        IGES Type 142
        """
        parametric_curve.status_number.value = 5
        parameter_data = [
            IGESParam(0, "int"),
            IGESParam(surface, "pointer"),
            IGESParam(parametric_curve, "pointer"),
            IGESParam(model_space_curve, "pointer"),
            IGESParam(3, "int")
        ]
        super().__init__(142, parameter_data, **entity_kwargs)


class CompositeCurveIGES(IGESEntity):
    def __init__(self, curves: typing.List[IGESEntity], **entity_kwargs):
        """
        IGES Type 102
        """
        parameter_data = [
            IGESParam(len(curves), "int"),
            *[IGESParam(curve, "pointer") for curve in curves]
        ]
        super().__init__(102, parameter_data, **entity_kwargs)
