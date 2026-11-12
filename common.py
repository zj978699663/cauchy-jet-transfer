import csv
import math
import os
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import eigh


ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
REFERENCE = ROOT / "reference"


DEGREES = (2, 3, 4, 5)
SCALES = tuple(0.2 / 2**j for j in range(5))
GAMMA = 10.0
QUAD = 8


def indices(p):
    return [(i, n - i) for n in range(p + 1) for i in range(n + 1)]


def derivative(poly, axis):
    result = {}
    for ij, value in poly.items():
        if ij[axis]:
            dest = list(ij)
            dest[axis] -= 1
            result[tuple(dest)] = value * ij[axis]
    return result


def multiply(first, second):
    result = {}
    for (i, j), value in first.items():
        for (k, l), other in second.items():
            dest = (i + k, j + l)
            result[dest] = result.get(dest, 0.0) + value * other
    return result


def operator_matrix(p, field):
    inds = indices(p)
    lookup = {ij: k for k, ij in enumerate(inds)}
    result = np.zeros((len(inds), len(inds)))
    for col, ij in enumerate(inds):
        poly = {ij: 1.0}
        for i in range(2):
            for j in range(2):
                term = derivative(multiply(field[i][j], derivative(poly, j)), i)
                for kl, value in term.items():
                    result[lookup[kl], col] -= value
    return result


def basis_values(p, x, y, dx=0, dy=0):
    x, y = np.broadcast_arrays(x, y)
    values = []
    for i, j in indices(p):
        if i < dx or j < dy:
            values.append(np.zeros_like(x, dtype=float))
        else:
            factor = math.factorial(i) / math.factorial(i - dx)
            factor *= math.factorial(j) / math.factorial(j - dy)
            values.append(factor * x ** (i - dx) * y ** (j - dy))
    return np.stack(values, axis=-1)


def jet(p, c):
    return np.array([basis_values(p, *c, i, j) for i, j in indices(p - 2)])


def zero_basis(p, c):
    inds = indices(p)
    result = np.zeros((len(inds), p * (p - 1) // 2))
    for col, (i, j) in enumerate(ij for ij in inds if ij[0] >= 2):
        for a in range(i + 1):
            for b in range(j + 1):
                result[inds.index((a, b)), col] = (
                    math.comb(i, a)
                    * math.comb(j, b)
                    * (-c[0]) ** (i - a)
                    * (-c[1]) ** (j - b)
                )
    return result


DIAGNOSTICS = {
    "jet_residual": 0.0,
    "idempotence": 0.0,
    "cauchy": 0.0,
    "factorisation": 0.0,
    "schur_identity": 0.0,
    "quadrature_relative": 0.0,
    "tangential": 0.0,
}


def update(name, value):
    DIAGNOSTICS[name] = max(DIAGNOSTICS.get(name, 0.0), float(value))


def projector(p, op, c):
    J = jet(p, c) @ op
    Z = zero_basis(p, c)
    U = Z @ np.linalg.solve(J @ Z, np.eye(Z.shape[1]))
    P = np.eye(Z.shape[0]) - U @ J
    update(
        "jet_residual",
        np.linalg.norm(J @ P) / (1 + np.linalg.norm(J) * np.linalg.norm(P)),
    )
    update("idempotence", np.linalg.norm(P @ P - P) / (1 + np.linalg.norm(P)))

    nodes = np.linspace(-0.4, 0.4, p + 1)
    bb = np.vstack(
        (
            basis_values(p, c[0], c[1] + nodes),
            basis_values(p, c[0], c[1] + nodes, 1, 0),
        )
    )
    update(
        "cauchy",
        np.linalg.norm(bb @ (P - np.eye(len(P))))
        / (1 + np.linalg.norm(bb) * np.linalg.norm(P)),
    )
    return P, U


def fields(family, p):
    zero = [[{}, {}], [{}, {}]]
    if family == "tensor":
        a0 = [[{(0, 0): 2.0}, {(0, 0): 1 / 3}], [{(0, 0): 1 / 3}, {(0, 0): 1.0}]]
        cc = [
            [{(1, 0): 1.0, (0, 1): 2.0}, {(1, 0): 1 / 3, (0, 1): -1 / 4}],
            [{(1, 0): 1 / 3, (0, 1): -1 / 4}, {(1, 0): 2.0, (0, 1): -1.0}],
        ]
        dd = [
            [{(2, 0): 1.0, (0, 2): 1.0}, {(1, 1): 0.2}],
            [{(1, 1): 0.2}, {(2, 0): 1.0, (0, 2): 2.0}],
        ]
    else:
        a0 = [[{(0, 0): 1.0}, {}], [{}, {(0, 0): 1.0}]]
        dd = zero
        if family == "sharp":
            cc = [[{(1, 0): 0.4}, {}], [{}, {(1, 0): 0.4}]]
        elif family in ("cancel", "cancel_quadratic"):
            a, b = 0.02, 0.03
            cc = [
                [
                    {(1, 0): -(2 * p - 1) * b, (0, 1): -(2 * p + 1) * a},
                    {(1, 0): a, (0, 1): b},
                ],
                [
                    {(1, 0): a, (0, 1): b},
                    {(1, 0): -(2 * p + 1) * b, (0, 1): -(2 * p - 1) * a},
                ],
            ]
            if family == "cancel_quadratic":
                dd = [
                    [{(2, 0): 0.15, (0, 2): 0.15}, {}],
                    [{}, {(2, 0): 0.15, (0, 2): 0.15}],
                ]
        elif family == "geometric":
            cc = [[{(1, 0): 1.0}, {(0, 1): -1.0}], [{(0, 1): -1.0}, {(1, 0): 1.0}]]
        else:
            raise ValueError(family)
    return a0, cc, dd


def combine(parts, h):
    return [
        [
            {
                ij: sum(h**k * part[i][j].get(ij, 0.0) for k, part in enumerate(parts))
                for ij in set().union(*(part[i][j] for part in parts))
            }
            for j in range(2)
        ]
        for i in range(2)
    ]


def field_values(field, x, y):
    def entry(poly):
        return sum(
            value * x**i * y**j for (i, j), value in poly.items()
        ) + np.zeros_like(x)

    return [[entry(field[i][j]) for j in range(2)] for i in range(2)]


def harmonic_basis(p, a0):
    inds = indices(p)
    op = operator_matrix(p, a0)
    columns = []
    for n in range(p + 1):
        if n == 0:
            v = np.zeros(len(inds))
            v[0] = 1.0
            columns.append(v)
            continue
        for normal in (0, 1):
            v = np.zeros(len(inds))
            v[inds.index((normal, n - normal))] = 1.0
            for i in range(2, n + 1):
                row = inds.index((i - 2, n - i))
                col = inds.index((i, n - i))
                v[col] = -(op[row] @ v) / op[row, col]
            columns.append(v)
    hh = np.array(columns).T
    assert np.linalg.norm(op @ hh) < 1.0e-10
    return hh


def geometry(name):
    if name == "uniform":
        root = (-0.5, 0.5, -0.5, 0.5)
        cuts = (0.0, 0.0)
    elif name == "offset":
        root = (-0.35, 0.65, -0.45, 0.55)
        cuts = (0.1, 0.05)
    elif name == "line":
        return (
            (-0.5, 0.5, -0.5, 0.5),
            [(-0.5, 0.0, -0.5, 0.5), (0.0, 0.5, -0.5, 0.5)],
            [(0, 1, 0.0, -0.5, 0.5, 0)],
        )
    else:
        raise ValueError(name)
    l, r, b, t = root
    cx, cy = cuts
    children = [
        (u, v, s, w) for u, v in ((l, cx), (cx, r)) for s, w in ((b, cy), (cy, t))
    ]
    faces = [
        (0, 2, cx, b, cy, 0),
        (1, 3, cx, cy, t, 0),
        (0, 1, cy, l, cx, 1),
        (2, 3, cy, cx, r, 1),
    ]
    return root, children, faces


def quadrature(rect, n):
    l, r, b, t = rect
    v, w = leggauss(n)
    xx = (r + l) / 2 + (r - l) * v / 2
    yy = (t + b) / 2 + (t - b) * v / 2
    x, y = np.meshgrid(xx, yy, indexing="ij")
    weights = np.outer(w, w) * (r - l) * (t - b) / 4
    return x.ravel(), y.ravel(), weights.ravel()


def volume_gram(p, field, rect, n, mass=False):
    x, y, w = quadrature(rect, n)
    gx = basis_values(p, x, y, 1, 0)
    gy = basis_values(p, x, y, 0, 1)
    aa = field_values(field, x, y)
    mat = gx.T @ ((w * aa[0][0])[:, None] * gx)
    mat += gx.T @ ((w * aa[0][1])[:, None] * gy)
    mat += gy.T @ ((w * aa[1][0])[:, None] * gx)
    mat += gy.T @ ((w * aa[1][1])[:, None] * gy)
    if mass:
        vv = basis_values(p, x, y)
        mat += vv.T @ (w[:, None] * vv)
    return (mat + mat.T) / 2


def energy(p, field, child_maps, geom, n=QUAD):
    root, children, faces = geometry(geom)
    size = child_maps[0].shape[1]
    mat = np.zeros((size, size))
    for rect, cc in zip(children, child_maps):
        mat += cc.T @ volume_gram(p, field, rect, n) @ cc
    for i, j, fixed, left, right, axis in faces:
        v, w = leggauss(n)
        s = (right + left) / 2 + (right - left) * v / 2
        xx, yy = (
            (np.full_like(s, fixed), s) if axis == 0 else (s, np.full_like(s, fixed))
        )
        jump = basis_values(p, xx, yy) @ (child_maps[i] - child_maps[j])

        mat += GAMMA * p * p / 2 * jump.T @ (w[:, None] * jump)
    return (mat + mat.T) / 2


def maximum_quotient(numerator, denominator):
    values = eigh(
        (numerator + numerator.T) / 2,
        (denominator + denominator.T) / 2,
        eigvals_only=True,
    )
    if values[-1] < -1e-12:
        raise RuntimeError("Negative energy maximum")
    return math.sqrt(max(0.0, values[-1]))


def spectral_data(p, parts, geom):
    a0, cc, _ = parts
    op0 = operator_matrix(p, a0)
    op1 = operator_matrix(p, cc)
    hh = harmonic_basis(p, a0)
    top = hh.copy()
    top[:, :-2] = 0.0
    centers = [((r[0] + r[1]) / 2, (r[2] + r[3]) / 2) for r in geometry(geom)[1]]
    ww = [-projector(p, op0, c)[1] @ jet(p, c) @ op1 @ top for c in centers]
    gg = hh.T @ volume_gram(p, a0, geometry(geom)[0], QUAD, True) @ hh
    bb = energy(p, a0, ww, geom)
    eff = gg[-2:, -2:] - gg[-2:, :-2] @ np.linalg.solve(gg[:-2, :-2], gg[:-2, -2:])
    kappa = maximum_quotient(bb[-2:, -2:], eff)
    full = maximum_quotient(bb, gg)
    naive = maximum_quotient(bb[-2:, -2:], gg[-2:, -2:])
    update("schur_identity", abs(kappa - full) / (1 + kappa))
    rank = np.linalg.matrix_rank(np.vstack(ww), tol=1e-11)
    assert rank <= 2
    return hh, kappa, naive


def with_rates(rows, keys, metrics):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    for group in groups.values():
        previous = None
        for row in group:
            for metric in metrics:
                row[metric + "_rate"] = ""
                if (
                    previous is not None
                    and row[metric] > 1e-13
                    and previous[metric] > 1e-13
                ):
                    row[metric + "_rate"] = math.log(
                        previous[metric] / row[metric]
                    ) / math.log(previous["h"] / row["h"])
            previous = row
    return rows


POLLUTION_GROUPS = (
    ("tensor", "uniform"),
    ("tensor", "offset"),
    ("sharp", "uniform"),
    ("cancel", "uniform"),
    ("cancel_quadratic", "uniform"),
)


def run_pollution(groups=POLLUTION_GROUPS, scales=SCALES):
    rows = []
    spectra = []
    for p in DEGREES:
        for family, geom in groups:
            parts = fields(family, p)
            hh, kappa, naive = spectral_data(p, parts, geom)
            spectra.append(
                dict(
                    family=family,
                    p=p,
                    geometry=geom,
                    kappa=kappa,
                    kappa_without_schur=naive,
                )
            )
            root, children, _ = geometry(geom)
            centers = [((r[0] + r[1]) / 2, (r[2] + r[3]) / 2) for r in children]
            eye = np.eye(hh.shape[0])
            for h in scales:
                field = combine(parts, h)
                op = operator_matrix(p, field)
                qq = projector(p, op, (0.0, 0.0))[0] @ hh
                corrections = [
                    (projector(p, op, c)[0] - eye) @ qq for c in centers
                ]
                gg = (
                    qq.T @ volume_gram(p, field, root, QUAD, True) @ qq
                )
                num = energy(p, field, corrections, geom)
                theta = maximum_quotient(num, gg)
                low = maximum_quotient(num[:-2, :-2], gg[:-2, :-2])
                theta_high = maximum_quotient(
                    energy(p, field, corrections, geom, QUAD + 3),
                    qq.T
                    @ volume_gram(p, field, root, QUAD + 3, True)
                    @ qq,
                )
                update(
                    "quadrature_relative",
                    abs(theta_high - theta) / max(theta, 1e-30),
                )
                rows.append(
                    dict(
                        family=family,
                        p=p,
                        geometry=geom,
                        h=h,
                        theta=theta,
                        kappa=kappa,
                        low=low,
                    )
                )
            print("pollution", family, p, geom, flush=True)
    return rows, spectra


def coefficient_norm(matrix):
    return float(np.linalg.norm(matrix, 2))


def check_diagnostics():
    assert DIAGNOSTICS["quadrature_relative"] < 1e-7, DIAGNOSTICS
    assert (
        max(v for k, v in DIAGNOSTICS.items() if k != "quadrature_relative") < 1e-9
    ), DIAGNOSTICS


def value(cell):
    try:
        return Decimal(cell)
    except InvalidOperation:
        return cell


def write_table(n, header, rows):
    name = f"table{n}.csv"
    actual = [[str(cell) for cell in row] for row in [header, *rows]]
    with (REFERENCE / name).open(encoding="utf-8", newline="") as stream:
        expected = list(csv.reader(stream))
    if len(actual) != len(expected):
        raise ValueError(
            f"{name}: expected {len(expected) - 1} rows, got {len(actual) - 1}"
        )
    for i, (a, b) in enumerate(zip(actual, expected), 1):
        if list(map(value, a)) != list(map(value, b)):
            raise ValueError(f"{name}, line {i}: {a} differs from {b}")
    with (BUILD / name).open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(actual)
    print(f"{name}: {len(actual) - 1} rows checked")


def select(rows, **keys):
    (row,) = (r for r in rows if all(str(r[k]) == str(v) for k, v in keys.items()))
    return row


BLUE = "#176b91"
ORANGE = "#ce6a22"
TEAL = "#16856b"
PURPLE = "#845797"
INK = "#30343b"
COLORS = (BLUE, ORANGE, TEAL, PURPLE, "#a63d52")
MARKERS = ("o", "s", "^", "D", "v")
LINES = ("-", "--", "-.", (0, (4, 1, 1, 1)))

W = 0.48 * 390 / 72.27
H = 0.93 * W
PANEL_RECT = (0.17, 0.12, 0.80, 0.80 / 0.93)
REMAINDER_RECT = (0.145, 0.128, 0.76, 0.76 / 0.93)
STYLE = dict(
    axis_font=8.2,
    tick_font=7.2,
    legend_font=6.8,
    frame_width=0.55,
    line_width=0.80,
    marker_size=3.4,
    marker_edge=0.75,
    legend_frame=0.42,
    legend_alpha=0.80,
)


def plotting():
    cache = BUILD / "cache"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    tempfile.tempdir = str(cache)
    import matplotlib as mpl

    mpl.use("pgf")
    import matplotlib.pyplot as plt

    preamble = r"\usepackage{amsmath,bm}"
    mpl.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts": False,
            "pgf.preamble": preamble,
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 8.2,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.55,
            "lines.linewidth": STYLE["line_width"],
            "lines.markersize": STYLE["marker_size"],
            "lines.markeredgewidth": 0.75,
            "lines.markerfacecolor": "white",
            "axes.unicode_minus": False,
            "savefig.bbox": None,
            "savefig.pad_inches": 0,
        }
    )
    return plt


def panel(plt, rect=PANEL_RECT):
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes(rect)
    ax.tick_params(
        which="both",
        direction="in",
        top=True,
        right=True,
        length=2.5,
        width=0.55,
        pad=2,
    )
    ax.tick_params(which="minor", length=1.5, width=0.4)
    ax.xaxis.labelpad = 2
    ax.yaxis.labelpad = 2
    ax.set_xscale("linear")
    ax.set_yscale("linear")
    ax.set_box_aspect(1)
    ax.grid(False)
    return fig, ax


def save(plt, fig, name):
    box = fig.axes[0].get_position()
    assert box.width * box.height >= 0.62, name
    assert abs(box.width * W / (box.height * H) - 1) < 1.0e-10, name
    fig.savefig(BUILD / name)
    plt.close(fig)


def legend(ax, **kwargs):
    options = dict(
        frameon=True,
        facecolor="white",
        framealpha=STYLE["legend_alpha"],
        edgecolor="#c0c4c8",
        handlelength=1.45,
        labelspacing=0.22,
        borderpad=0.28,
        handletextpad=0.45,
    )
    options.update(kwargs)
    result = ax.legend(**options)
    result.get_frame().set_linewidth(STYLE["legend_frame"])
    return result


def line(ax, x, y, index, label, **kwargs):
    ax.plot(
        x,
        y,
        color=COLORS[index],
        marker=MARKERS[index],
        linestyle=LINES[index % len(LINES)],
        label=label,
        **kwargs,
    )


def math_number(value, _pos=None):
    return "$" + format(value, "g") + "$"


def h_axis(ax):
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    hs = [0.0125, 0.025, 0.05, 0.1, 0.2]
    ax.set_xscale("log", base=2)
    ax.set_xlim(0.0105, 0.238)
    ax.xaxis.set_major_locator(FixedLocator(hs))
    ax.xaxis.set_major_formatter(FuncFormatter(math_number))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"$h$")
