import os

import matplotlib.pyplot as plt

from aerocaps.examples.rational_bezier_curve import quarter_circle_rational_bezier
from aerocaps.utils.plotting import FONT_SERIF_14, FONT_SERIF_16, format_axis_basic


def elevate_quarter_circle(image_dir: str = None):
    r"""
    Plots quarter circle exactly represented both by a quadratic rational Bézier curve and a cubic
    rational Bézier curve

    Parameters
    ----------
    image_dir: str or None
        Where to store the output image. If no location is specified, the image file will not be saved, and the
        plot will be displayed on-screen instead. Default: ``None``
    """
    curve = quarter_circle_rational_bezier()
    elevated = curve.elevate_degree()

    # Generate the plots
    for c in [curve, elevated]:
        n = c.degree
        fig, ax = plt.subplots()
        format_axis_basic(ax, FONT_SERIF_14)
        ax.set_aspect("equal")
        c.plot(ax, projection="XY", color="steelblue")
        c.plot_control_points(ax, projection="XY", marker="o", mfc="black", mec="black", ls=":", color="black")
        ax.text(0.4, 0.4, f"{n = }", fontdict=FONT_SERIF_16)
        ax.set_xlabel(r"$x$", font=FONT_SERIF_16)
        ax.set_ylabel(r"$y$", font=FONT_SERIF_16)
        if image_dir:
            fig.savefig(os.path.join(image_dir, f"quarter_circle_n={n}.png"), bbox_inches="tight")
        else:
            plt.show()


def main(*args, **kwargs):
    elevate_quarter_circle(*args, **kwargs)


if __name__ == "__main__":
    main(image_dir=r"C:\Users\mlaue\Documents\aerocaps\docs\source\images")
