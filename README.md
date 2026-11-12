# Cauchy-jet transfer experiments

## Scope

This package reproduces Tables I–II and Figures 1–2 of *Cauchy Data Preservation and Pollution Asymptotics for Variable-Coefficient Quasi-Trefftz Discontinuous Galerkin Transfer*.

## Run

Use Python 3.13.9 and pdfLaTeX (TeX Live 2025) with Computer Modern, `amsmath` and `bm`. Run from this directory.

```text
python -m pip install numpy==2.3.5 scipy==1.16.3 sympy==1.14.0 mpmath==1.3.0 matplotlib==3.10.6
python -B ex1.py
python -B ex2.py
```

## Tables, figures and checks

Each command can run on its own. Both generate their data and write the results to `build/`.

| Paper | Command | Calculation and checks | Output |
|---|---|---|---|
| §11.1 Worst-case pollution and cancellation | `python -B ex1.py` | Compute the constants and transfer factors for degrees 2–5. Check the projectors by symbolic differentiation and the smallest-scale pollution cases at 80 digits. | `table1.csv`, `fig1a.pdf`, `fig1b.pdf` |
| §11.2 Directional cocycles and accumulated branch terms | `python -B ex2.py` | Recompute the required pollution rates and the directional and branch remainders. Use 0.025 and 0.0125 for the observed orders. | `table2.csv`, `fig2a.pdf`, `fig2b.pdf` |

Both scripts use the fields, scales, and eight-point Gauss rule in `common.py`. Both commands check their CSV values against `reference/`.

Compare `fig1a.pdf`, `fig1b.pdf`, `fig2a.pdf`, and `fig2b.pdf` in `build/` with the files of the same name in `reference/`. Check the page size, curves, ticks, legends, and edges. For a pixel comparison, render each pair at 120 dpi with Poppler's `pdftoppm -r 120 -singlefile input.pdf output` and compare the resulting PPM images.
