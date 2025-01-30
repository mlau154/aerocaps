import numpy as np

import aerocaps as ac


def main():
    surf_1 = ac.BezierSurface(np.array([
        [
            [0.0, 0.0, 0.0],
            [0.3, 0.1, 0.0],
            [0.6, -0.1, 0.0],
            [1.0, 0.05, 0.0]
        ],
        [
            [0.0, 0.0, 0.5],
            [0.3, 0.1, 0.5],
            [0.6, -0.1, 0.5],
            [1.0, 0.05, 0.5]
        ],
        [
            [0.0, 0.0, 1.0],
            [0.3, 0.1, 1.0],
            [0.6, -0.1, 1.0],
            [1.0, 0.05, 1.0]
        ]
    ]))
    surf_2 = ac.BezierSurface(np.array([
        [
            [0.0, 0.0, 1.0],
            [0.3, 0.1, 1.0],
            [0.6, -0.1, 1.0],
            [1.0, 0.05, 1.0]
        ],
        [
            [0.0, 0.0, 1.5],
            [0.3, 0.1, 1.5],
            [0.6, -0.1, 1.5],
            [1.0, 0.05, 1.5]
        ],
        [
            [0.0, 0.0, 2.0],
            [0.3, 0.25, 2.0],
            [0.6, -0.35, 2.0],
            [1.1, 0.2, 2.0]
        ]
    ]))
    container = ac.GeometryContainer()
    container.add_geometry(surf_1)
    container.add_geometry(surf_2)
    # container.plot()
    container.export_iges("double_bez_surf_test.igs", units="meters")


if __name__ == "__main__":
    main()
