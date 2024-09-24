import typing

import numpy as np

from astk.iges.curves import BoundaryCurveIGES
from astk.units.angle import Angle
from astk.iges.entity import IGESEntity
from astk.iges.iges_param import IGESParam


class SurfaceOfRevolutionIGES(IGESEntity):
    def __init__(self, axis_of_revolution: IGESEntity, curve: IGESEntity, start_angle: Angle, end_angle: Angle,
                 **entity_kwargs):
        parameter_data = [
            IGESParam(axis_of_revolution, "pointer"),
            IGESParam(curve, "pointer"),
            IGESParam(start_angle.rad, "real"),
            IGESParam(end_angle.rad, "real")
        ]
        super().__init__(120, parameter_data, **entity_kwargs)


class RuledSurfaceIGES(IGESEntity):
    def __init__(self, curve_1: IGESEntity, curve_2: IGESEntity, **entity_kwargs):
        parameter_data = [
            IGESParam(curve_1, "pointer"),
            IGESParam(curve_2, "pointer"),
            IGESParam(0, "int"),
            IGESParam(1, "int")
        ]
        super().__init__(118, parameter_data, **entity_kwargs)


class BoundedSurfaceIGES(IGESEntity):
    def __init__(self,
                 untrimmed_surface: IGESEntity,
                 boundary_curves: typing.List[BoundaryCurveIGES],
                 **entity_kwargs):
        parameter_data = [
            IGESParam(int(any(
                [boundary_curve.parameter_data[0].value == 1 for boundary_curve in boundary_curves])), "int"
            ),
            IGESParam(untrimmed_surface, "pointer"),
            IGESParam(len(boundary_curves), "int"),
            *[IGESParam(boundary_curve, "pointer") for boundary_curve in boundary_curves]
        ]
        super().__init__(143, parameter_data, **entity_kwargs)


class RationalBSplineSurfaceIGES(IGESEntity):
    def __init__(self,
                 control_points: np.ndarray,
                 knots_u: np.ndarray,
                 knots_v: np.ndarray,
                 weights: np.ndarray,
                 degree_u: int, degree_v: int,
                 closed_u: bool = True, closed_v: bool = True,
                 rational: bool = True,
                 periodic_u: bool = False, periodic_v: bool = False,
                 start_u: float = 0.0, end_u: float = 1.0,
                 start_v: float = 0.0, end_v: float = 1.0,
                 **entity_kwargs
                 ):
        assert control_points.ndim == 3
        assert knots_u.ndim == 1
        assert knots_v.ndim == 1
        assert weights.ndim == 2
        self.control_points = control_points
        self.knots_u = knots_u
        self.knots_v = knots_v
        self.weights = weights
        parameter_data = [
            IGESParam(control_points.shape[0] - 1, "int"),
            IGESParam(control_points.shape[1] - 1, "int"),
            IGESParam(degree_u, "int"),
            IGESParam(degree_v, "int"),
            IGESParam(int(closed_u), "int"),
            IGESParam(int(closed_v), "int"),
            IGESParam(int(not rational), "int"),
            IGESParam(int(periodic_u), "int"),
            IGESParam(int(periodic_v), "int"),
            *[IGESParam(k, "real") for k in self.knots_u],
            *[IGESParam(k, "real") for k in self.knots_v],
            *self._flatten_weights(),
            *self._flatten_control_points(),
            IGESParam(start_u, "real"),
            IGESParam(end_u, "real"),
            IGESParam(start_v, "real"),
            IGESParam(end_v, "real")
        ]
        super().__init__(128, parameter_data, **entity_kwargs)

    def _flatten_control_points(self):
        control_point_params_flattened = []
        for j in range(self.control_points.shape[1]):
            for i in range(self.control_points.shape[0]):
                for k in range(self.control_points.shape[2]):
                    control_point_params_flattened.append(IGESParam(self.control_points[i, j, k], "real"))
        return control_point_params_flattened

    def _flatten_weights(self):
        weights_flattened = []
        for j in range(self.weights.shape[1]):
            for i in range(self.weights.shape[0]):
                weights_flattened.append(IGESParam(self.weights[i, j], "real"))
        return weights_flattened


class BezierSurfaceIGES(RationalBSplineSurfaceIGES):
    def __init__(self, control_points: np.ndarray, **entity_kwargs):
        order_u = control_points.shape[0]
        order_v = control_points.shape[1]
        degree_u = order_u - 1
        degree_v = order_v - 1
        knots_u = np.concatenate((np.zeros(order_u), np.ones(order_u)))
        knots_v = np.concatenate((np.zeros(order_v), np.ones(order_v)))
        weights = np.ones(shape=(order_u, order_v))
        super().__init__(control_points=control_points, knots_u=knots_u, knots_v=knots_v, weights=weights,
                         degree_u=degree_u, degree_v=degree_v, **entity_kwargs)
