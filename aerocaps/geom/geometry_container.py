"""Storage container module"""
import time
import typing

import numpy as np
import pyvista as pv

from aerocaps.geom import Geometry
from aerocaps.iges.iges_generator import IGESGenerator

__all__ = [
    "GeometryContainer"
]


class GeometryContainer:
    """Storage container for geometric objects that adds convenience methods for plotting and export"""
    def __init__(self):
        r"""
        Storage container for geometric objects that adds convenience methods for plotting and export. The example
        code below shows how to add a curve and a surface to a new container, plots them in an interactive scene,
        exports them to IGES, and then removes both of them by varying identifiers:

        .. code-block:: python

            # Create the geometric objects
            curve = BezierCurve3D(np.array([
                [0.0, 0.0, 0.0],
                [0.3, 0.2, 0.1],
                [0.6, 0.1, 0.3],
                [1.0, -0.1, 0.2]
            ]), name='MyCurve')
            surf = BezierSurface(np.array([
                [
                    [0.0, 0.0, 0.0],
                    [0.3, 0.2, 0.1],
                    [0.6, 0.1, 0.3],
                    [1.0, -0.1, 0.2]
                ],
                [
                    [0.0, 0.0, 1.0],
                    [0.3, 0.4, 1.1],
                    [0.6, 0.2, 1.3],
                    [1.0, -0.3, 1.2]
                ]
            ]))

            # Instantiate a container
            container = GeometryContainer()

            # Add the geometries to the container
            container.add_geometry(point)
            container.add_geometry(curve)

            # List the geometries inside the container
            geom_names = container.geometry_name_list()
            print(f'{geom_names = }')

            # Plot the geometries in an interactive scene
            container.plot()

            # Export the geometries to an IGES file
            container.export_iges('curve_and_surf.igs', units='meters')

            # Remove the curve and surface by different methods
            container.remove_geometry('MyCurve')
            container.remove_geometry(surf)

            # Show that the container is now empty
            geom_names = container.geometry_name_list()
            print(f'{geom_names = }')
        """
        self._container = dict()

    def _get_max_index_associated_with_name(self, name: str) -> int:
        """
        Gets the maximum index associated with a given name in the geometry container. If the name is the container
        but the name does not have an index (denoted by ``-<index>``), ``0`` will be returned.

        Parameters
        ----------
        name: str
            Name of the geometry in the container

        Returns
        -------
        int
            Maximum index associated with the name
        """
        split_keys = [k.split("-") for k in self._container.keys()]
        indices = [int(split[-1]) for split in split_keys if split[0] == name and len(split) > 1]
        if not indices:
            return 0  # This means that "-1" will be appended to the name
        return max(indices)

    def add_geometry(self, geom: Geometry):
        """
        Adds a geometric object to the container, renaming the object with a higher index if necessary

        Parameters
        ----------
        geom: Geometry
            Geometric object to add
        """
        if geom.name in self._container.keys():
            max_index = self._get_max_index_associated_with_name(geom.name)
            geom._name = f"{geom.name.split()[0]}-{max_index + 1}"
        self._container[geom.name] = geom
        geom.container = self

    def remove_geometry(self, geom: str or Geometry) -> Geometry:
        """
        Removes a geometric object from the container

        Parameters
        ----------
        geom: str or Geometry
            The geometry to remove. If a :obj:`str`, this must be found in the list of geometry names that have
            already been added to the container or an exception will be thrown

        Returns
        -------
        Geometry
            The geometry removed
        """
        if isinstance(geom, str):
            if geom not in self._container.keys():
                raise ValueError(f"Could not find geometry {geom} in container")
            return self._container.pop(geom)
        if isinstance(geom, Geometry):
            if geom not in self._container.values():
                raise ValueError(f"Could not find geometry {geom} in container")
            return self._container.pop(geom.name)
        raise ValueError("'geom' must sub-class either 'str' or 'Geometry'")

    def geometry_by_name(self, name: str) -> Geometry or None:
        """
        Searches for a geometry in the container by name

        Parameters
        ----------
        name: str
            Name of the geometric object

        Returns
        -------
        Geometry or None
            If found, a geometric object is returned. Otherwise, ``None`` is returned
        """
        if name not in self._container.keys():
            return None
        return self._container[name]

    def geometry_name_list(self, geom_type: type = None) -> typing.List[str]:
        """
        Gets the list of geometries (by name) that have been added to the container

        Parameters
        ----------
        geom_type: type
            If specified, only geometries with the given type will be returned. Default: ``None``

        Returns
        -------
        typing.List[str]
            List of geometry names
        """
        if geom_type is None:
            return list(self._container.keys())
        return [k for k, v in self._container.items() if isinstance(v, geom_type)]

    def plot(self, show: bool = True):
        """
        Plots all the plottable objects in the container onto a :obj:`pyvista.Plotter` scene.
        Also adds a surface picker to dynamically show surface information on right-click.

        Parameters
        ----------
        show: bool
            Whether to show the plot. Default: ``True``
        """
        def selection_callback(mesh):

            if not hasattr(mesh, "aerocaps_surf"):
                return

            mesh.aerocaps_surf.plot_control_point_mesh_lines(
                plot,
                color="blue",
                name="selection_lines"
            )
            mesh.aerocaps_surf.plot_control_points(
                plot,
                render_points_as_spheres=True,
                color="black",
                point_size=16,
                name="selection_cps"
            )
            points = np.array([
                mesh.aerocaps_surf.evaluate(0.0, 0.5),
                mesh.aerocaps_surf.evaluate(1.0, 0.5),
                mesh.aerocaps_surf.evaluate(0.5, 0.0),
                mesh.aerocaps_surf.evaluate(0.5, 1.0)
            ])
            plot.add_point_labels(
                points=points,
                labels=["u0", "u1", "v0", "v1"],
                shape_color="white",
                always_visible=True,
                name="surf_edge_labels"
            )
            plot.add_text(
                text=str(mesh.aerocaps_surf),
                position="lower_left",
                name="surf_repr"
            )

        start_time = time.perf_counter()
        plot = pv.Plotter()
        for geom in self._container.values():
            if geom.construction:  # Skip the construction geometries
                continue
            if hasattr(geom, "plot_surface"):
                grid = geom.plot_surface(plot, 50, 50)
                grid.aerocaps_surf = geom
            if hasattr(geom, "plot"):
                geom.plot(plot, color="lime")

        plot.enable_mesh_picking(
            callback=selection_callback,
            style="surface",
            color="indianred",
            picker="hardware"
        )
        plot.add_axes()
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"\033[1;35mModel rendering time: {elapsed_time:.3f} seconds\033[0m")

        if show:
            plot.show()

    def export_iges(self, file_name: str, units: str = "meters"):
        """
        Exports all the exportable objects in the container to an IGES file

        Parameters
        ----------
        file_name: str
            Path to the IGES file
        units: str
            Physical length units used to export the geometries. See
            :obj:`aerocaps.iges.iges_generator.IGESGenerator.__init__` for more details. Default: ``"meters"``
        """
        geoms_to_export = [
            geom.to_iges() for geom in self._container.values() if not geom.construction and hasattr(geom, "to_iges")
        ]
        iges_generator = IGESGenerator(geoms_to_export, units)
        iges_generator.generate(file_name)
