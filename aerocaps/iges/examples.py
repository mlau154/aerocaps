import numpy as np

from aerocaps.iges.iges_generator import IGESGenerator
from aerocaps.iges.curves import LineIGES, BezierIGES, BoundaryCurveIGES, CircularArcIGES
from aerocaps.iges.surfaces import RuledSurfaceIGES, BoundedSurfaceIGES
from aerocaps.iges.transformation import TransformationMatrixIGES
from aerocaps.units.angle import Angle
from aerocaps.units.length import Length


def generate_planar_surface_iges(file_name: str):
    lower_line = LineIGES(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    upper_line = LineIGES(np.array([0.2, 1.0, 0.0]), np.array([0.8, 1.0, 0.0]))
    planar_surface = RuledSurfaceIGES(lower_line, upper_line)
    iges_generator = IGESGenerator(entities=[lower_line, upper_line, planar_surface], units="meters")
    iges_generator.generate(file_name=file_name)


def generate_bounded_surface_iges(file_name: str):
    lower_line = LineIGES(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    upper_line = LineIGES(np.array([0.2, 1.0, 0.0]), np.array([0.8, 1.0, 0.0]))
    planar_surface = RuledSurfaceIGES(lower_line, upper_line)
    bez_upper = BezierIGES(np.array([
        [0.3, 0.2, 0.0],
        [0.3, 0.25, 0.0],
        [0.35, 0.25, 0.0],
        [0.65, 0.24, 0.0],
        [0.7, 0.2, 0.0]
    ]))
    bez_lower = BezierIGES(np.array([
        [0.3, 0.2, 0.0],
        [0.3, 0.15, 0.0],
        [0.35, 0.15, 0.0],
        [0.65, 0.22, 0.0],
        [0.7, 0.2, 0.0]
    ]))
    outer_circle = CircularArcIGES(Length(m=0.3), start_angle=Angle(deg=0.0), end_angle=Angle(deg=0.0))
    transformation_matrix = TransformationMatrixIGES(tx=Length(m=0.5), ty=Length(m=0.35))
    outer_circle.transformation_matrix.value = transformation_matrix
    boundary_curve_inner = BoundaryCurveIGES(planar_surface, {bez_upper: [], bez_lower: []})
    boundary_curve_outer = BoundaryCurveIGES(planar_surface, {outer_circle: []})
    boundary_surface = BoundedSurfaceIGES(planar_surface, [boundary_curve_inner, boundary_curve_outer])
    entities = [
        lower_line,
        upper_line,
        planar_surface,
        bez_upper,
        bez_lower,
        outer_circle,
        transformation_matrix,
        boundary_curve_inner,
        boundary_curve_outer,
        boundary_surface
    ]
    iges_generator = IGESGenerator(entities=entities, units="meters")
    iges_generator.generate(file_name=file_name)
