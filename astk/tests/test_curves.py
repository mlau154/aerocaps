import numpy as np
import shapely.geometry
import scienceplots
import matplotlib.pyplot as plt
from pymead.post.plot_formatters import show_save_fig

from tpai.geometry.curves import RationalBezierCurve3D, NURBSCurve3D
from tpai import FIGURE_DIR

plt.style.use(["science", "bright"])


def test_rational_bezier_circle_area():
    """
    Creates a circle from 4 quarter-circles represented exactly by rational Bézier curves and verifies that the area
    enclosed by the polygon of evaluated points along the arcs approaches the analytical area of a circle with the
    same radius
    """
    show_plot = False
    radius = 1.2
    t_vec = np.linspace(0.0, 1.0, 1001)
    arc_QI = RationalBezierCurve3D.generate_from_array(
        np.array([[radius, 0.0, 0.0], [radius, radius, 0.0], [0.0, radius, 0.0]]),
        weights=np.array([1.0, 1 / np.sqrt(2.0), 1.0])
    )
    arc_QII = RationalBezierCurve3D.generate_from_array(
        np.array([[0.0, radius, 0.0], [-radius, radius, 0.0], [-radius, 0.0, 0.0]]),
        weights=np.array([1.0, 1 / np.sqrt(2.0), 1.0])
    )
    arc_QIII = RationalBezierCurve3D.generate_from_array(
        np.array([[-radius, 0.0, 0.0], [-radius, -radius, 0.0], [0.0, -radius, 0.0]]),
        weights=np.array([1.0, 1 / np.sqrt(2.0), 1.0])
    )
    arc_QIV = RationalBezierCurve3D.generate_from_array(
        np.array([[0.0, -radius, 0.0], [radius, -radius, 0.0], [radius, 0.0, 0.0]]),
        weights=np.array([1.0, 1 / np.sqrt(2.0), 1.0])
    )
    circle_points = arc_QI.evaluate(t_vec)
    for arc in [arc_QII, arc_QIII, arc_QIV]:
        circle_points = np.row_stack((circle_points, arc.evaluate(t_vec)[1:, :]))
    poly = shapely.geometry.Polygon(circle_points)

    # Plot if desired
    if show_plot:
        plt.plot(circle_points[:, 0], circle_points[:, 1], color="steelblue")
        for arc in [arc_QI, arc_QII, arc_QIII, arc_QIV]:
            plt.plot(arc.get_control_point_array()[:, 0], arc.get_control_point_array()[:, 1],
                     ls="none", marker="s",
                     color="mediumaquamarine")
        plt.gca().set_aspect("equal")
        plt.show()

    analytical_area = np.pi * radius ** 2
    assert np.isclose(poly.area, analytical_area)


def test_nurbs_circle_area():
    """
    Creates a circle from a single NURBS curve and verifies that the area
    enclosed by the polygon of evaluated points along the arcs approaches the analytical area of a circle with the
    same radius
    """
    show_plot = False
    radius = 2.11
    t_vec = np.linspace(0.0, 1.0, 4001)
    arc = NURBSCurve3D(
        control_points=np.array([
            [radius, 0.0, 0.0],
            [radius, radius, 0.0],
            [0.0, radius, 0.0],
            [-radius, radius, 0.0],
            [-radius, 0.0, 0.0],
            [-radius, -radius, 0.0],
            [0.0, -radius, 0.0],
            [radius, -radius, 0.0],
            [radius, 0.0, 0.0]
        ]),
        weights=np.array([1.0, 1 / np.sqrt(2.0),
                          1.0, 1 / np.sqrt(2.0),
                          1.0, 1 / np.sqrt(2.0),
                          1.0, 1 / np.sqrt(2.0),
                          1.0]),
        knot_vector=np.array([0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0]),
        degree=2
    )
    circle_points_nurbs = arc.evaluate(t_vec)
    poly = shapely.geometry.Polygon(circle_points_nurbs)

    if show_plot:
        circle_points_trig = np.array([[
            radius * np.cos(t), radius * np.sin(t)] for t in np.linspace(0.0, np.pi * 2, len(t_vec))])
        plt.plot(circle_points_trig[:, 0], circle_points_trig[:, 1], color="steelblue", ls="-.", label="Trig")
        plt.plot(circle_points_nurbs[:, 0], circle_points_nurbs[:, 1], color="indianred", ls=":", label="NURBS")
        plt.gca().set_aspect("equal")
        plt.legend()
        plt.show()

    analytical_area = np.pi * radius ** 2
    assert np.isclose(poly.area, analytical_area)


def test_c0c1c2_rational_bezier_planar():
    """
    Creates 2 rational Bézier curves that are initially discontinuous in :math:`C^0`, :math:`C^1`, and :math:`C^2`,
    but then all 3 continuity levels are enforced between the end of the first and the start of the second curve by
    appropriately modifying the first three rows of control points on the second curve.
    """
    show_plot = True
    fig, ax = None, None
    rb1 = RationalBezierCurve3D.generate_from_array(
        np.array([
            [0.0, 0.0, 0.0],
            [0.3, 0.1, 0.0],
            [0.6, 0.5, 0.0],
            [0.8, 0.2, 0.0],
            [0.9, 0.25, 0.0]
        ]),
        weights=np.array([1.0, 0.7, 1.7, 0.8, 0.86])
    )
    rb2 = RationalBezierCurve3D.generate_from_array(
        np.array([
            [0.5, 0.1, 0.0],
            [0.7, 0.3, 0.0],
            [0.9, 0.4, 0.0],
            [1.7, 0.5, 0.0],
            [1.4, 0.35, 0.0]
        ]),
        weights=np.array([1.0, 0.7, 0.5, 0.8, 1.0])
    )

    if show_plot:
        fig, ax = plt.subplots(figsize=(4, 5))
        rb1.plot(ax, projection="XY", color="indianred", label="Curve a")
        rb1.plot_control_points(
            ax, projection="XY", color="indianred", ls=":", marker="o", mec="indianred", mfc="none",
            label="Curve a Control Points")
        rb2.plot(ax, projection="XY", color="mediumaquamarine", ls="-.", label="Curve b")
        rb2.plot_control_points(
            ax, projection="XY", color="mediumaquamarine", ls=":", marker="s", mec="mediumaquamarine", mfc="none",
            label="Curve b Control Points"
        )

    rb1.enforce_c0c1c2(rb2)

    if show_plot:
        rb2.plot(ax, projection="XY", color="steelblue", label="Curve b'")
        rb2.plot_control_points(
            ax, projection="XY", color="steelblue", ls=":", marker="o", mec="steelblue", mfc="none",
            label="Curve b' Control Points")
        # h, l = ax.get_legend_handles_labels()
        # reorder = lambda _l, nc: sum((_l[i::nc] for i in range(nc)), [])
        # fig.legend(reorder(h, 2), reorder(l, 2), ncol=2, bbox_to_anchor=[0.5, 0.89], loc="center")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        fig.tight_layout()
        show_save_fig(fig, FIGURE_DIR, "planar_rational_bezier_c2_continuity",
                      show=True, save=True, save_ext=(".pdf",))

    kappa_1a = rb1.compute_curvature_at_t1()
    kappa_0b = rb2.compute_curvature_at_t0()
    assert np.isclose(kappa_1a, kappa_0b)


def test_g0g1g2_rational_bezier_planar():
    """
    Creates 2 rational Bézier curves that are initially discontinuous in :math:`G^0`, :math:`G^1`, and :math:`G^2`,
    but then all 3 continuity levels are enforced between the end of the first and the start of the second curve by
    appropriately modifying the first three rows of control points on the second curve.
    """
    show_plot = True
    fig, ax = None, None
    rb1 = RationalBezierCurve3D.generate_from_array(
        np.array([
            [0.0, 0.0, 0.0],
            [0.3, 0.1, 0.0],
            [0.6, 0.5, 0.0],
            [0.8, 0.2, 0.0],
            [0.9, 0.25, 0.0]
        ]),
        weights=np.array([1.0, 0.7, 1.7, 0.8, 0.86])
    )
    rb2 = RationalBezierCurve3D.generate_from_array(
        np.array([
            [0.5, 0.1, 0.0],
            [0.7, 0.3, 0.0],
            [0.9, 0.4, 0.0],
            [1.7, 0.5, 0.0],
            [1.4, 0.35, 0.0]
        ]),
        weights=np.array([1.0, 0.7, 0.5, 0.8, 1.0])
    )

    if show_plot:
        fig, ax = plt.subplots(figsize=(4, 3))
        rb1.plot(ax, projection="XY", color="indianred", label="Curve a")
        rb1.plot_control_points(
            ax, projection="XY", color="indianred", ls=":", marker="o", mec="indianred", mfc="none",
            label="Curve a Control Points")
        rb2.plot(ax, projection="XY", color="mediumaquamarine", ls="-.", label="Curve b")
        rb2.plot_control_points(
            ax, projection="XY", color="mediumaquamarine", ls=":", marker="s", mec="mediumaquamarine", mfc="none",
            label="Curve b Control Points"
        )

    rb1.enforce_g0g1g2(rb2, f=0.5)

    if show_plot:
        rb2.plot(ax, projection="XY", color="steelblue", label="Curve b'")
        rb2.plot_control_points(
            ax, projection="XY", color="steelblue", ls=":", marker="o", mec="steelblue", mfc="none",
            label="Curve b' Control Points")
        h, l = ax.get_legend_handles_labels()
        reorder = lambda _l, nc: sum((_l[i::nc] for i in range(nc)), [])
        fig.legend(reorder(h, 2), reorder(l, 2), ncol=2, bbox_to_anchor=[0.5, 0.89], loc="center")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        fig.tight_layout()
        show_save_fig(fig, FIGURE_DIR, "planar_rational_bezier_g2_continuity",
                      show=True, save=True, save_ext=(".pdf",))

    kappa_1a = rb1.compute_curvature_at_t1()
    kappa_0b = rb2.compute_curvature_at_t0()
    assert np.isclose(kappa_1a, kappa_0b)


def test_g0g1g2_rational_bezier_nonplanar():
    """
    Creates 2 rational Bézier curves that are initially discontinuous in :math:`G^0`, :math:`G^1`, and :math:`G^2`,
    but then all 3 continuity levels are enforced between the end of the first and the start of the second curve by
    appropriately modifying the first three rows of control points on the second curve.
    """
    show_plot = True
    fig, ax = None, None
    rb1 = RationalBezierCurve3D.generate_from_array(
        np.array([
            [0.2, -0.1, -0.5],
            [0.3, 0.1, 0.2],
            [0.6, 0.5, 0.3],
            [0.8, 0.2, 0.2],
            [0.9, 0.25, 0.1]
        ]),
        weights=np.array([1.0, 0.7, 1.7, 0.8, 0.86])
    )
    rb2 = RationalBezierCurve3D.generate_from_array(
        np.array([
            [0.5, 0.1, 0.2],
            [0.7, 0.3, -0.3],
            [0.9, 0.4, 0.1],
            [1.4, 0.5, 0.6],
            [1.7, 0.35, 0.4]
        ]),
        weights=np.array([1.0, 0.7, 0.5, 0.8, 1.0])
    )

    if show_plot:
        fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(projection="3d"))
        rb1.plot(ax, color="indianred", label="Curve a")
        rb1.plot_control_points(
            ax, color="indianred", ls=":", marker="o", mec="indianred", mfc="none",
            label="Curve a Control Points")
        rb2.plot(ax, color="mediumaquamarine", ls="-.", label="Curve b")
        rb2.plot_control_points(
            ax, color="mediumaquamarine", ls=":", marker="s", mec="mediumaquamarine", mfc="none",
            label="Curve b Control Points"
        )

    rb1.enforce_g0g1g2(rb2, f=0.5)

    if show_plot:
        rb2.plot(ax, color="steelblue", label="Curve b'")
        rb2.plot_control_points(
            ax, color="steelblue", ls=":", marker="o", mec="steelblue", mfc="none",
            label="Curve b' Control Points")
        # h, l = ax.get_legend_handles_labels()
        # reorder = lambda _l, nc: sum((_l[i::nc] for i in range(nc)), [])
        # fig.legend(reorder(h, 2), reorder(l, 2), ncol=2, bbox_to_anchor=[0.5, 0.9], loc="center")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_aspect("equal")
        fig.subplots_adjust(left=-0.15)
        show_save_fig(fig, FIGURE_DIR, "nonplanar_rational_bezier_g2_continuity",
                      show=True, save=True, save_ext=(".pdf",))

    kappa_1a = rb1.compute_curvature_at_t1()
    kappa_0b = rb2.compute_curvature_at_t0()
    assert np.isclose(kappa_1a, kappa_0b)


def test_g0g1g2_quarter_circle_cubic_planar():
    """
    Creates 2 rational Bézier curves that are initially discontinuous in :math:`G^0`, :math:`G^1`, and :math:`G^2`,
    but then all 3 continuity levels are enforced between the end of the first and the start of the second curve by
    appropriately modifying the first three rows of control points on the second curve.
    """
    show_plot = True
    fig, ax = None, None
    rb1 = RationalBezierCurve3D.generate_from_array(
        np.array([
            [0.0, -0.5, 0.0],
            [0.0, -0.5, 0.5],
            [0.0, 0.0, 0.5],
        ]),
        weights=np.array([1.0, 1 / np.sqrt(2.0), 1.0])
    )
    rb2 = RationalBezierCurve3D.generate_from_array(
        np.array([
            [0.0, 0.1, 0.1],
            [0.0, 0.3, 0.3],
            [0.0, 0.4, 0.4],
            [0.0, 0.7, 0.2],
        ]),
        weights=np.array([1.0, 1.0, 1.0, 1.0])
    )

    if show_plot:
        fig, ax = plt.subplots(figsize=(4, 3.5))
        rb1.plot(ax, projection="YZ", color="indianred", label="Curve a")
        rb1.plot_control_points(
            ax, projection="YZ", color="indianred", ls=":", marker="o", mec="indianred", mfc="none",
            label="Curve a Control Points")
        rb2.plot(ax, projection="YZ", color="mediumaquamarine", ls="-.", label="Curve b")
        rb2.plot_control_points(
            ax, projection="YZ", color="mediumaquamarine", ls=":", marker="s", mec="mediumaquamarine", mfc="none",
            label="Curve b Control Points"
        )

    rb1.enforce_g0g1g2(rb2, f=2.0)

    if show_plot:
        rb2.plot(ax, projection="YZ", color="steelblue", label="Curve b'")
        rb2.plot_control_points(
            ax, projection="YZ", color="steelblue", ls=":", marker="o", mec="steelblue", mfc="none",
            label="Curve b' Control Points")
        # h, l = ax.get_legend_handles_labels()
        # reorder = lambda _l, nc: sum((_l[i::nc] for i in range(nc)), [])
        # fig.legend(reorder(h, 2), reorder(l, 2), ncol=2, bbox_to_anchor=[0.5, 0.85], loc="center")
        ax.set_xlabel("y")
        ax.set_ylabel("z")
        ax.set_aspect("equal")
        fig.tight_layout()
        show_save_fig(fig, FIGURE_DIR, "quarter_circle_bezier_g2_continuity",
                      show=True, save=True, save_ext=(".pdf",))

    kappa_1a = rb1.compute_curvature_at_t1()
    kappa_0b = rb2.compute_curvature_at_t0()
    assert np.isclose(kappa_1a, kappa_0b)


def test_rational_bezier_weights():
    show_plot = True
    middle_weights = [0.3, 0.5, 1 / np.sqrt(2.0), 1.0, 2.0]
    labels = [fr"$w_1={w:.4f}$" for w in middle_weights]
    rb = None
    fig, ax = plt.subplots(figsize=(3, 3))

    for w, label in zip(middle_weights, labels):
        rb = RationalBezierCurve3D.generate_from_array(
            np.array([
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ]),
            weights=np.array([1.0, w, 1.0])
        )
        rb.plot(ax, projection="XY", label=label)

    rb.plot_control_points(ax, projection="XY", color="black", marker="o", mec="black", mfc="none", ls=":",
                            label="Control Points")

    for idx, cp in enumerate(rb.get_control_point_array()):
        ax.text(cp[0] + 0.02, cp[1] + 0.02, fr"$P_{idx}$", color="black")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim([-0.05, 1.1])
    ax.set_ylim([-0.05, 1.1])
    ax.set_aspect("equal")
    ax.legend()
    fig.tight_layout()

    if not show_plot:
        return

    show_save_fig(fig, FIGURE_DIR, "rational_bezier_weights", show=True, save=True, save_ext=(".pdf", ".svg"))
