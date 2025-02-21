import typing

import numpy as np

from aerocaps.geom import Surface


class STLGenerator:
    """
    Reference: https://www.loc.gov/preservation/digital/formats/fdd/fdd000506.shtml
    """
    def __init__(self, geoms: typing.List[Surface], Nu: int = 50, Nv: int = 50):
        self.geoms = geoms
        self.Nu = Nu
        self.Nv = Nv

    def generate(self, file_name: str):
        with open(file_name, "w") as stl_file:
            stl_file.write("solid aerocaps\n")

            for geom in self.geoms:
                point_array = geom.evaluate_grid(self.Nu, self.Nv)
                for i in range(point_array.shape[0] - 1):
                    for j in range(point_array.shape[1] - 1):
                        vertex_1 = point_array[i, j, :]
                        vertex_2 = point_array[i + 1, j, :]
                        vertex_3 = point_array[i, j + 1, :]
                        vertex_4 = point_array[i + 1, j + 1, :]
                        normal_1 = np.cross(vertex_2 - vertex_1, vertex_3 - vertex_1)
                        normal_2 = np.cross(vertex_3 - vertex_4, vertex_2 - vertex_4)
                        stl_file.write(f"facet normal {normal_1[0]} {normal_1[1]} {normal_1[2]}\n")
                        stl_file.write(f"    outer loop\n")
                        stl_file.write(f"        vertex {vertex_1[0]} {vertex_1[1]} {vertex_1[2]}\n")
                        stl_file.write(f"        vertex {vertex_2[0]} {vertex_2[1]} {vertex_2[2]}\n")
                        stl_file.write(f"        vertex {vertex_3[0]} {vertex_3[1]} {vertex_3[2]}\n")
                        stl_file.write(f"    endloop\n")
                        stl_file.write(f"endfacet\n")
                        stl_file.write(f"facet normal {normal_2[0]} {normal_2[1]} {normal_2[2]}\n")
                        stl_file.write(f"    outer loop\n")
                        stl_file.write(f"        vertex {vertex_4[0]} {vertex_4[1]} {vertex_4[2]}\n")
                        stl_file.write(f"        vertex {vertex_2[0]} {vertex_2[1]} {vertex_2[2]}\n")
                        stl_file.write(f"        vertex {vertex_3[0]} {vertex_3[1]} {vertex_3[2]}\n")
                        stl_file.write(f"    endloop\n")
                        stl_file.write(f"endfacet\n")

            stl_file.write("endsolid aerocaps\n")
