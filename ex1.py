import math
import mpmath as mp
import numpy as np
import sympy as sy
from scipy.linalg import eigh

from common import (
    BUILD,
    DEGREES,
    GAMMA,
    QUAD,
    SCALES,
    check_diagnostics,
    coefficient_norm,
    combine,
    energy,
    fields,
    geometry,
    h_axis,
    harmonic_basis,
    indices,
    legend,
    line,
    maximum_quotient,
    operator_matrix,
    panel,
    plotting,
    projector,
    run_pollution,
    save,
    select,
    update,
    volume_gram,
    write_table,
)

def run_certificates():
    certificates = []
    for p in DEGREES:
        parts = fields("tensor", p)
        h = 0.05
        field = combine(parts, h)
        op = operator_matrix(p, field)
        source = projector(p, op, (0.0, 0.0))[0] @ harmonic_basis(p, parts[0])
        for geom in ("uniform", "offset"):
            target_bases = []
            maps = []
            transfers = []
            for rect in geometry(geom)[1]:
                center = ((rect[0] + rect[1]) / 2, (rect[2] + rect[3]) / 2)
                P, _ = projector(p, op, center)
                basis = np.linalg.qr(P @ harmonic_basis(p, parts[0]), mode="reduced")[
                    0
                ]
                target_bases.append(basis)
                maps.append(P @ source)
                transfers.append(basis.T @ maps[-1])
            n = 2 * p + 1
            count = len(maps)
            embedded = []
            for i, basis in enumerate(target_bases):
                block = np.zeros((len(op), count * n))
                block[:, i * n : (i + 1) * n] = basis
                embedded.append(block)
            child = energy(p, field, embedded, geom)
            tt = np.vstack(transfers)
            gg = (
                source.T @ volume_gram(p, field, geometry(geom)[0], QUAD, True) @ source
            )
            pull = tt.T @ child @ tt
            direct = energy(p, field, maps, geom)
            assembly_error = coefficient_norm(pull - direct) / (
                1 + coefficient_norm(direct)
            )
            vals, vecs = eigh(pull, gg)
            v = vecs[:, -1]
            factor = math.sqrt(vals[-1])
            rayleigh = math.sqrt(v @ direct @ v / (v @ gg @ v))

            change = np.eye(n) + 0.1 * np.triu(np.ones((n, n)), 1)
            changed = maximum_quotient(change.T @ pull @ change, change.T @ gg @ change)
            update("matrix_assembly", assembly_error)
            update("gram_eigenvector", abs(factor - rayleigh))
            update("basis_invariance", abs(factor - changed))
            certificates.append(
                dict(
                    p=p,
                    geometry=geom,
                    h=h,
                    S=factor,
                    direct_rayleigh=rayleigh,
                    assembly_error=assembly_error,
                    basis_invariance_error=abs(factor - changed),
                )
            )
    return certificates


x, y = sy.symbols("x y")


def rat(value):
    return sy.Rational(float(value)).limit_denominator(10**8)


def independent_operator(p, field):
    matrix = sy.Matrix(
        [
            [
                sum(rat(c) * x**i * y**j for (i, j), c in field[a][b].items())
                for b in range(2)
            ]
            for a in range(2)
        ]
    )
    mons = [x**i * y**j for i, j in indices(p)]
    cols = []
    for v in mons:
        flux = matrix * sy.Matrix([sy.diff(v, x), sy.diff(v, y)])
        poly = sy.Poly(-sy.diff(flux[0], x) - sy.diff(flux[1], y), x, y)
        cols.append([poly.coeff_monomial(m) for m in mons])
    return sy.Matrix(cols).T


def symbolic_projector(p, field, c):
    op = independent_operator(p, field)
    inds = indices(p)
    mons = [x**i * y**j for i, j in inds]
    cx, cy = map(rat, c)
    J = (
        sy.Matrix(
            [
                [sy.diff(v, x, i, y, j).subs({x: cx, y: cy}) for v in mons]
                for i, j in indices(p - 2)
            ]
        )
        * op
    )
    zm = [sy.Poly((x - cx) ** i * (y - cy) ** j, x, y) for i, j in inds if i >= 2]
    Z = sy.Matrix([[v.coeff_monomial(m) for v in zm] for m in mons])
    P = sy.eye(len(inds)) - Z * (J * Z).inv() * J
    assert J * P == sy.zeros(*J.shape)
    assert P * P == P
    return P, op


def exact_integral_matrix(p, field, rect, mass=False):
    inds = indices(p)
    l, r, b, t = map(mp.mpf, map(str, rect))

    def moment(i, j):
        return (
            (r ** (i + 1) - l ** (i + 1))
            / (i + 1)
            * (t ** (j + 1) - b ** (j + 1))
            / (j + 1)
        )

    result = mp.matrix(len(inds))
    for row, ij in enumerate(inds):
        for col, kl in enumerate(inds):
            total = mp.mpf(0)
            for a in range(2):
                for bb in range(2):
                    if ij[a] == 0 or kl[bb] == 0:
                        continue
                    for (u, v), coeff in field[a][bb].items():
                        i = ij[0] + kl[0] + u - (a == 0) - (bb == 0)
                        j = ij[1] + kl[1] + v - (a == 1) - (bb == 1)
                        total += mp.mpf(str(coeff)) * ij[a] * kl[bb] * moment(i, j)
            if mass:
                total += moment(ij[0] + kl[0], ij[1] + kl[1])
            result[row, col] = total
    return result


def mp_matrix(mat):
    return mp.matrix([[mp.mpf(str(v)) for v in row] for row in mat])


def mp_jet(p, c):
    cx, cy = map(mp.mpf, map(str, c))
    inds = indices(p)
    return mp.matrix(
        [
            [
                mp.mpf(
                    math.factorial(i)
                    // math.factorial(i - a)
                    * math.factorial(j)
                    // math.factorial(j - b)
                )
                * cx ** (i - a)
                * cy ** (j - b)
                if i >= a and j >= b
                else 0
                for i, j in inds
            ]
            for a, b in indices(p - 2)
        ]
    )


def mp_projector(p, op, c):
    inds = indices(p)
    cx, cy = map(mp.mpf, map(str, c))
    zm = [(i, j) for i, j in inds if i >= 2]
    zz = mp.matrix(
        [
            [
                math.comb(i, a) * math.comb(j, b) * (-cx) ** (i - a) * (-cy) ** (j - b)
                if a <= i and b <= j
                else 0
                for i, j in zm
            ]
            for a, b in inds
        ]
    )
    jj = mp_jet(p, c) * op
    return mp.eye(len(inds)) - zz * (jj * zz) ** -1 * jj


def mp_energy(p, field, maps, geom):
    root, children, faces = geometry(geom)
    result = mp.matrix(maps[0].cols)
    inds = indices(p)
    for rect, cc in zip(children, maps):
        result += cc.T * exact_integral_matrix(p, field, rect) * cc
    for i, j, fixed, l, r, axis in faces:
        fixed, l, r = map(mp.mpf, map(str, (fixed, l, r)))
        face = mp.matrix(len(inds))
        for a, (u, v) in enumerate(inds):
            for b, (s, t) in enumerate(inds):
                fixedpow = u + s if axis == 0 else v + t
                activepow = v + t if axis == 0 else u + s
                face[a, b] = (
                    fixed**fixedpow
                    * (r ** (activepow + 1) - l ** (activepow + 1))
                    / (activepow + 1)
                )
        jump = maps[i] - maps[j]
        result += mp.mpf(str(GAMMA)) * p * p / (r - l) * jump.T * face * jump
    return result


def mp_maximum(num, den):
    ll = mp.cholesky(den)
    inv = ll**-1
    ww = inv * num * inv.T
    vals = mp.eigsy((ww + ww.T) / 2, eigvals_only=True)
    return mp.sqrt(max(mp.mpf(0), vals[vals.rows - 1]))


def ellipticity_bound(parts, h, geom):
    root = geometry(geom)[0]
    mx = max(abs(root[0]), abs(root[1]))
    my = max(abs(root[2]), abs(root[3]))
    a0 = np.array(
        [[parts[0][a][b].get((0, 0), 0.0) for b in range(2)] for a in range(2)]
    )
    bound = np.linalg.eigvalsh(a0)[0]
    for power, field in enumerate(parts[1:], 1):
        keys = set().union(*(field[a][b] for a in range(2) for b in range(2)))
        for i, j in keys:
            mat = np.array(
                [[field[a][b].get((i, j), 0.0) for b in range(2)] for a in range(2)]
            )
            bound -= h**power * mx**i * my**j * np.linalg.norm(mat, 2)
    return bound


def verify_pollution(data):
    for p in DEGREES:
        field = combine(fields("tensor", p), 0.025)
        P, op = symbolic_projector(p, field, (0.25, -0.25))
        numeric, _ = projector(p, operator_matrix(p, field), (0.25, -0.25))
        error = float(np.max(np.abs(np.array(P, float) - numeric)))
        assert error < 1e-11
    cross = []
    for p in DEGREES:
        for family in ("tensor", "sharp", "cancel", "cancel_quadratic"):
            geom = "offset" if family == "tensor" else "uniform"
            h = SCALES[-1]
            parts = fields(family, p)
            field = combine(parts, h)

            op = mp.matrix(independent_operator(p, field).tolist())
            hh = mp_matrix(harmonic_basis(p, parts[0]))
            qq = mp_projector(p, op, (0.0, 0.0)) * hh
            centers = [
                ((r[0] + r[1]) / 2, (r[2] + r[3]) / 2) for r in geometry(geom)[1]
            ]
            maps = [(mp_projector(p, op, c) - mp.eye(op.rows)) * qq for c in centers]
            den = qq.T * exact_integral_matrix(p, field, geometry(geom)[0], True) * qq
            theta = float(mp_maximum(mp_energy(p, field, maps, geom), den))
            low = float(
                mp_maximum(
                    mp_energy(p, field, [v[:, :-2] for v in maps], geom), den[:-2, :-2]
                )
            )
            row = next(
                row
                for row in data
                if int(row["p"]) == p
                and row["family"] == family
                and row["geometry"] == geom
                and float(row["h"]) == h
            )
            relative = abs(theta - float(row["theta"])) / theta
            lowrel = abs(low - float(row["low"])) / low
            cross.append(
                dict(
                    p=p,
                    family=family,
                    geometry=geom,
                    h=h,
                    theta_80=theta,
                    relative_difference=relative,
                    low_relative_difference=lowrel,
                )
            )
            assert max(relative, lowrel) < 0.01
            print("80-digit", p, family, relative, flush=True)
    minimum = min(
        ellipticity_bound(fields(fam, p), max(SCALES), geom)
        for p in DEGREES
        for fam in ("tensor", "sharp", "cancel", "cancel_quadratic")
        for geom in ("uniform", "offset")
    )
    assert minimum > 0
    maximum = max(
        max(v["relative_difference"], v["low_relative_difference"]) for v in cross
    )
    print(
        f"80-digit checks: {len(cross)} cases, maximum relative difference {maximum:.3g}"
    )


def spectral(spectra, certificates):
    rows = []
    for p in (2, 3, 4, 5):
        u = select(spectra, p=p, family="tensor", geometry="uniform")
        o = select(spectra, p=p, family="tensor", geometry="offset")
        f = select(certificates, p=p, geometry="offset", h=0.05)
        values = [u["kappa"], o["kappa"], o["kappa_without_schur"], f["S"]]
        rows.append([p, *(f"{float(v):.6f}" for v in values)])
    write_table(
        1,
        ["p", "kappa_uniform", "kappa_offset", "top_block_offset", "S_offset"],
        rows,
    )


def draw_pollution(rows):
    plt = plotting()
    from matplotlib.ticker import FixedLocator, FuncFormatter

    for name, family, geom, field, ylabel in [
        ("fig1a.pdf", "tensor", "offset", "linear", r"$\Theta(h)/(h\kappa)$"),
        ("fig1b.pdf", "cancel_quadratic", "uniform", "quadratic", r"$\Theta(h)/h^2$"),
    ]:
        fig, ax = panel(plt)
        for i, p in enumerate((2, 3, 4, 5)):
            selected = sorted(
                (
                    r
                    for r in rows
                    if r["family"] == family
                    and r["geometry"] == geom
                    and int(r["p"]) == p
                ),
                key=lambda r: float(r["h"]),
            )
            x = np.array([float(r["h"]) for r in selected])
            denominator = (
                x * np.array([float(r["kappa"]) for r in selected])
                if field == "linear"
                else x * x
            )
            y = np.array([float(r["theta"]) for r in selected]) / denominator
            line(ax, x, y, i, rf"$p={p}$")
        h_axis(ax)
        ax.set_ylabel(ylabel)
        if field == "linear":
            ax.axhline(1, color="#555b63", ls=":", lw=0.7, zorder=0)
            legend(ax, loc="lower left", ncol=2, columnspacing=0.65)
        else:
            ax.set_ylim(0.052, 0.087)
            ax.yaxis.set_major_locator(FixedLocator([0.055, 0.065, 0.075, 0.085]))
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda value, pos: "$" + format(value, ".3f") + "$")
            )
            legend(ax, loc="center right", ncol=2, columnspacing=0.65)
        save(plt, fig, name)


def main():
    BUILD.mkdir(exist_ok=True)
    rows, spectra = run_pollution()
    certificates = run_certificates()
    check_diagnostics()
    with mp.workdps(80):
        verify_pollution(rows)
    spectral(spectra, certificates)
    draw_pollution(rows)


if __name__ == "__main__":
    main()
