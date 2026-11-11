import math

import numpy as np

from common import (
    BUILD,
    COLORS,
    DEGREES,
    LINES,
    MARKERS,
    REMAINDER_RECT,
    SCALES,
    basis_values,
    check_diagnostics,
    coefficient_norm,
    combine,
    fields,
    h_axis,
    harmonic_basis,
    indices,
    jet,
    legend,
    line,
    math_number,
    operator_matrix,
    panel,
    plotting,
    projector,
    run_pollution,
    save,
    select,
    update,
    with_rates,
    write_table,
)


def leading_cocycle(p, op0, source, d, e):
    ell = (
        (
            d[0] * basis_values(p, 0.0, 0.0, 1, p - 2)
            + d[1] * basis_values(p, 0.0, 0.0, 0, p - 1)
        )
        @ op0
        @ source
    )
    P0, _ = projector(p, op0, (0.0, 0.0))
    inds = indices(p)
    generator = np.zeros(len(inds))
    generator[inds.index((1, p - 2))] = 1.0
    psi = P0 @ generator

    A11 = -op0[0, inds.index((2, 0))] / 2
    return np.outer(psi, e[0] * ell / (A11 * math.factorial(p - 2)))


def run_cocycle_and_branch():
    directions = []
    branches = []
    for p in DEGREES:
        field = combine(fields("tensor", p), 1.0)
        op = operator_matrix(p, field)

        hh = harmonic_basis(p, fields("tensor", p)[0])
        source = projector(p, op, (0.0, 0.0))[0] @ hh
        source = np.linalg.qr(source, mode="reduced")[0]
        for beta in (0.0, 0.001, 0.1, 0.3):
            for h in SCALES:
                d = h * np.array([0.4, -0.3])
                e = h * np.array([beta, 0.2])
                z = d + e
                py, uy = projector(p, op, d)
                pz, _ = projector(p, op, z)
                defect = (pz @ py - pz) @ source
                exact = -pz @ uy @ jet(p, d) @ op @ source
                update("factorisation", coefficient_norm(defect - exact))
                lead = leading_cocycle(p, op, source, d, e)
                remainder = coefficient_norm(defect - lead)
                if beta == 0.0:
                    update("tangential", coefficient_norm(defect))
                rank = np.linalg.matrix_rank(lead, tol=1e-11)
                assert rank <= 1

                assert np.linalg.matrix_rank(defect, tol=1e-11) <= 2 * p - 3
                directions.append(
                    dict(
                        p=p,
                        beta=beta,
                        h=h,
                        defect=coefficient_norm(defect),
                        remainder=remainder,
                        directional_scaled=remainder
                        / (
                            abs(e[0])
                            * np.linalg.norm(d)
                            * (np.linalg.norm(d) + np.linalg.norm(e))
                        )
                        if beta
                        else "",
                        leading_rank=rank,
                    )
                )
        for levels in (4, 8, 16):
            for h in SCALES:
                current = np.zeros(2)
                successive = source.copy()
                lead = np.zeros_like(source)
                s_path = 0.0
                for level in range(levels):
                    e = h / 4 * 2.0 ** (-level) * np.array([1.0, (-1.0) ** level])
                    if level:
                        lead += leading_cocycle(p, op, source, current, e)
                    current += e
                    s_path += np.linalg.norm(e)
                    successive = projector(p, op, current)[0] @ successive
                direct = projector(p, op, current)[0] @ source
                error = successive - direct
                branches.append(
                    dict(
                        p=p,
                        L=levels,
                        h=h,
                        s_path=s_path,
                        discrepancy=coefficient_norm(error),
                        remainder=coefficient_norm(error - lead),
                        scaled_remainder=coefficient_norm(error - lead)
                        / s_path**3,
                    )
                )
        print("directions and branches", p, flush=True)
    return (
        with_rates(directions, ("p", "beta"), ("defect", "remainder")),
        with_rates(branches, ("p", "L"), ("discrepancy", "remainder")),
    )


def orders(pollution, directions, branches):
    rows = []
    for p in (2, 3, 4, 5):
        tensor = select(pollution, p=p, family="tensor", geometry="offset", h=0.0125)
        sharp = select(pollution, p=p, family="sharp", geometry="uniform", h=0.0125)
        cancel = select(
            pollution, p=p, family="cancel_quadratic", geometry="uniform", h=0.0125
        )
        cocycle = select(directions, p=p, beta=0.3, h=0.0125)
        branch = select(branches, p=p, L=16, h=0.0125)
        rates = [
            tensor["theta_rate"],
            tensor["low_rate"],
            sharp["theta_rate"],
            cancel["theta_rate"],
            cocycle["remainder_rate"],
            branch["remainder_rate"],
        ]
        rows.append([p, *(f"{float(v):.3f}" for v in rates)])
    write_table(
        2,
        [
            "p",
            "tensor",
            "lower_modes",
            "sharpness",
            "perturbed_cancellation",
            "cocycle",
            "branch",
        ],
        rows,
    )


def draw_remainders(rows, branches):
    plt = plotting()
    from matplotlib.ticker import FuncFormatter

    hs = [0.2, 0.1, 0.05, 0.025, 0.0125]
    betas = [0.001, 0.1, 0.3]
    values = np.array(
        [
            [
                float(
                    next(
                        r["directional_scaled"]
                        for r in rows
                        if int(r["p"]) == 5
                        and float(r["beta"]) == b
                        and float(r["h"]) == h
                    )
                )
                for h in hs
            ]
            for b in betas
        ]
    )

    fig, ax = panel(plt, REMAINDER_RECT)
    remainder_bottom, remainder_height = REMAINDER_RECT[1], REMAINDER_RECT[3]
    im = ax.pcolormesh(
        np.arange(6) - 0.5,
        np.arange(4) - 0.5,
        values,
        cmap="cividis",
        vmin=200,
        vmax=260,
        shading="flat",
        edgecolors="white",
        linewidth=0.55,
    )
    ax.set(
        xlim=(-0.5, 4.5),
        ylim=(-0.5, 2.5),
        xticks=range(5),
        xticklabels=[math_number(h) for h in hs],
        yticks=range(3),
        yticklabels=[r"$10^{-3}$", r"$0.1$", r"$0.3$"],
        xlabel=r"$h$",
        ylabel=r"$\beta$",
    )
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)

    ax.yaxis.set_label_coords(-0.135, 0.5)
    for j in range(3):
        for i in range(5):
            color = "white" if values[j, i] < 220 else "#20242a"
            ax.text(
                i,
                j,
                "$" + format(values[j, i], ".1f") + "$",
                ha="center",
                va="center",
                color=color,
                fontsize=7.2,
            )
    cax = fig.add_axes((0.912, remainder_bottom, 0.014, remainder_height))
    cb = fig.colorbar(im, cax=cax, ticks=[200, 230, 260])
    cb.ax.tick_params(labelsize=6.3, length=1.2, pad=0.7)
    cax.set_title(r"$Q_{\rm dir}$", loc="right", fontsize=7.2, pad=2)
    save(plt, fig, "fig2a.pdf")

    rows = branches
    fig, ax = panel(plt, REMAINDER_RECT)
    for i, depth in enumerate((4, 8, 16)):
        z = sorted(
            (r for r in rows if int(r["p"]) == 5 and int(r["L"]) == depth),
            key=lambda r: float(r["h"]),
        )
        line(
            ax,
            [float(r["h"]) for r in z],
            [float(r["scaled_remainder"]) for r in z],
            i,
            rf"$L={depth}$",
            zorder=5 - i,
        )
    h_axis(ax)
    ax.set_ylabel(r"$Q_{\rm path}$")
    ax.set_ylim(17.5, 26)
    legend(ax, loc="lower left")
    zoom = ax.inset_axes([0.53, 0.73, 0.43, 0.22])
    for i, depth in enumerate((8, 16), 1):
        z = sorted(
            (r for r in rows if int(r["p"]) == 5 and int(r["L"]) == depth),
            key=lambda r: float(r["h"]),
        )
        zoom.plot(
            [float(r["h"]) for r in z],
            [float(r["scaled_remainder"]) for r in z],
            c=COLORS[i],
            marker=MARKERS[i],
            ls=LINES[i],
            lw=0.65,
            ms=2.7,
        )
    zoom.set(
        xlim=(0.011, 0.027),
        ylim=(23.1, 23.7),
        xticks=[0.0125, 0.025],
        yticks=[23.2, 23.6],
    )
    zoom.tick_params(direction="in", labelsize=5.8, length=1.5, pad=1)
    zoom.xaxis.set_major_formatter(FuncFormatter(math_number))
    ax.indicate_inset_zoom(zoom, edgecolor="#8b929a", linewidth=0.4, alpha=0.65)
    save(plt, fig, "fig2b.pdf")


def main():
    BUILD.mkdir(exist_ok=True)
    groups = (("tensor", "offset"), ("sharp", "uniform"), ("cancel_quadratic", "uniform"))
    rows, _ = run_pollution(groups, SCALES[-2:])
    directions, branches = run_cocycle_and_branch()
    check_diagnostics()
    orders(rows, directions, branches)
    draw_remainders(directions, branches)


if __name__ == "__main__":
    main()
