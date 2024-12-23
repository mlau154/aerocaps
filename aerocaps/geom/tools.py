import numpy as np
from scipy.optimize import minimize_scalar

from aerocaps.geom.curves import Bezier3D, PCurve2D, PCurve3D, Line3D
from aerocaps.geom.point import Point3D, Point2D
from aerocaps.geom.transformation import Transformation3D
from aerocaps.geom.vector import Vector3D
from aerocaps.units.angle import Angle


__all__ = [
    "measure_distance_between_points",
    "measure_pitch_angle",
    "measure_distance_point_line",
    "add_vector_to_point",
    "project_point_onto_line",
    "find_t_corresponding_to_minimum_distance_to_point2d",
    "find_t_corresponding_to_minimum_distance_to_point3d",
    "sweep_along_curve",
    "rotate_about_axis",
    "rotate_point_about_axis"
]


def measure_distance_between_points(point_1: Point2D or Point3D or np.ndarray,
                                    point_2: Point2D or Point3D or np.ndarray) -> float:
    if isinstance(point_1, Point2D) or isinstance(point_1, Point3D):
        point_1 = point_1.as_array()
    if isinstance(point_2, Point2D) or isinstance(point_2, Point3D):
        point_2 = point_2.as_array()
    if len(point_1) != len(point_2):
        raise ValueError("Cannot calculate distance between two points of different dimensionality")
    if len(point_1) == 2 and len(point_2) == 2:
        return np.sqrt((point_2[0] - point_1[0]) ** 2 + (point_2[1] - point_1[1]) ** 2)
    elif len(point_1) == 3 and len(point_2) == 3:
        return np.sqrt((point_2[0] - point_1[0]) ** 2 + (point_2[1] - point_1[1]) ** 2 + (point_2[2] - point_1[2]) ** 2)
    else:
        raise ValueError("Points must have dimension 2 or 3")


def measure_pitch_angle(point_1: Point3D or np.ndarray, point_2: Point3D or np.ndarray) -> Angle:
    """
    Translates the two points such that the first point is at the origin and then measures the angle the line
    connecting the two points makes on the X-Z plane
    """
    if isinstance(point_1, Point3D):
        point_1 = point_1.as_array()
    if isinstance(point_2, Point3D):
        point_2 = point_2.as_array()
    point_2_translated = np.array([point_2[0] - point_1[0], point_2[1] - point_1[1], point_2[2] - point_1[2]])
    return Angle(rad=np.arctan2(point_2_translated[2], point_2_translated[0]))


def measure_distance_point_line(point: Point3D, line: Line3D) -> float:
    return measure_distance_between_points(point, project_point_onto_line(point, line))


def add_vector_to_point(vector: Vector3D, point: Point3D) -> Point3D:
    return point + (vector.p1 - vector.p0)


def project_point_onto_line(point: Point3D, line: Line3D) -> Point3D:
    vAB = Vector3D(p0=line.p0, p1=line.p1)
    vAC = Vector3D(p0=line.p0, p1=point)
    vAD_value = vAB.as_array() * (vAB.dot(vAC) / vAB.dot(vAB))
    vAD = Vector3D.from_array(vAD_value)
    return add_vector_to_point(vector=vAD, point=line.p0)


def find_t_corresponding_to_minimum_distance_to_point2d(curve: PCurve2D, point: np.ndarray or Point2D) -> (
        float, float):

    point = point.as_array() if isinstance(point, Point2D) else point

    def minimize_func(t):
        curve_point = curve.evaluate_single_t(t)
        return (curve_point[0] - point[0]) ** 2 + (curve_point[1] - point[1]) ** 2

    res = minimize_scalar(minimize_func, bounds=[0.0, 1.0])
    return res.x, np.sqrt(res.fun)


def find_t_corresponding_to_minimum_distance_to_point3d(curve: PCurve3D, point: np.ndarray or Point3D) -> (
        float, float):

    point = point.as_array() if isinstance(point, Point3D) else point

    def minimize_func(t):
        curve_point = curve.evaluate_single_t(t)
        return (curve_point[0] - point[0]) ** 2 + (curve_point[1] - point[1]) ** 2 + (curve_point[2] - point[2]) ** 2

    res = minimize_scalar(minimize_func, bounds=[0.0, 1.0])
    return res.x, np.sqrt(res.fun)


def sweep_along_curve(primary_curve: Bezier3D, guide_curve: Bezier3D):
    point_list_3d = [primary_curve.control_points]
    for prev_point, current_point in zip(guide_curve.control_points[:-1], guide_curve.control_points[1:]):
        point_list_3d.append([primary_point + current_point - prev_point for primary_point in point_list_3d[-1]])
    return np.array([[[point.x.m, point.y.m, point.z.m] for point in v] for v in point_list_3d])


def rotate_about_axis(points: np.ndarray, axis: Vector3D, angle: Angle) -> np.ndarray:
    ux, uy, uz = axis.normalized_value()
    c = np.cos(angle.rad)
    s = np.sin(angle.rad)
    rotation_matrix = np.array([
        [c + ux**2 * (1 - c), ux * uy * (1 - c) - uz * s, ux * uz * (1 - c) + uy * s],
        [uy * ux * (1 - c) + uz * s, c + uy**2 * (1 - c), uy * uz * (1 - c) - ux * s],
        [uz * ux * (1 - c) - uy * s, uz * uy * (1 - c) + ux * s, c + uz**2 * (1 - c)]
    ])
    return (rotation_matrix @ points.T).T


def rotate_point_about_axis(p: Point3D, ax: Line3D, angle: Angle) -> Point3D:
    reverse_transformation = Transformation3D(tx=-ax.p0.x.m, ty=-ax.p0.y.m, tz=-ax.p0.z.m)
    forward_transformation = Transformation3D(tx=ax.p0.x.m, ty=ax.p0.y.m, tz=ax.p0.z.m)
    p_mat = np.array([p.as_array()])
    p_mat = reverse_transformation.transform(p_mat)
    p_mat = rotate_about_axis(p_mat, ax.get_vector(), angle)
    p_mat = forward_transformation.transform(p_mat)
    return Point3D.from_array(p_mat[0])
