import os

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv


FONT_SERIF_10 = dict(family="serif", size=10)
FONT_SERIF_14 = dict(family="serif", size=14)
FONT_SERIF_16 = dict(family="serif", size=16)
FONT_SERIF_18 = dict(family="serif", size=18)


def format_axis_basic(ax: plt.Axes or Axes3D, font: dict):
    for tick in ax.get_xticklabels():
        tick.set_fontfamily(font["family"])
        tick.set_fontsize(font["size"])
    for tick in ax.get_yticklabels():
        tick.set_fontfamily(font["family"])
        tick.set_fontsize(font["size"])
    if not isinstance(ax, Axes3D):
        return
    for tick in ax.get_zticklabels():
        tick.set_fontfamily(font["family"])
        tick.set_fontsize(font["size"])


def format_axis_scientific(ax: plt.Axes, font: dict):
    format_axis_basic(ax, font)
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", left=True, bottom=True, top=True, right=True)
    ax.tick_params(which="major", direction="in", left=True, bottom=True, top=True, right=True)


def take_picture_3view_plus_iso(plot: pv.Plotter, save_dir: str, name_base: str, scale: int = 2,
                                position: tuple = (-1.0, -1.0, 1.3), focal_point: tuple = (0.5, 0.5, 0.0),
                                viewup: tuple = (0.0, 0.0, 1.0)):
    # Save top/front/right/iso views to png files
    plot.view_yx(negative=True)  # Top
    plot.screenshot(os.path.join(save_dir, f"{name_base}_top.png"), scale=scale)
    plot.view_yz(negative=True)  # Front
    plot.screenshot(os.path.join(save_dir, f"{name_base}_front.png"), scale=scale)
    plot.view_xz()  # Side
    plot.screenshot(os.path.join(save_dir, f"{name_base}_right.png"), scale=scale)
    plot.camera_position = pv.CameraPosition(position=position, focal_point=focal_point, viewup=viewup)
    plot.screenshot(os.path.join(save_dir, f"{name_base}_iso.png"), scale=scale)

