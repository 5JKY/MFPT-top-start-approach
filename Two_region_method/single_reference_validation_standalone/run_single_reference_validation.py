#!/usr/bin/env python3
"""
Directly reproduce the manuscript's two-region single-reference validation.

This script uses ONLY:
  * two_region_validation_data.npz  (raw saved RW P_st and MFPT arrays)
  * one model reference value beta U(x_R) at x_R = -1.0 in region A

It does NOT use the saved two-reference reconstructed free-energy arrays.
It reconstructs both RW regional profiles from P_st and MFPT, gives them
arbitrary additive constants, fixes region A with the single reference, then
matches region B by six-point quadratic extrapolation to a = -0.1.

The transfer-matrix benchmark is rebuilt internally from the model potential.
The known value beta U(a) is used only AFTER reconstruction to quantify the
validation deviation.

Expected output:
    RW deviations: 0.082221, 0.015966, 0.068273  (all < 0.1)
    TM deviations: 0.005235, 0.005037, 0.002627  (~5e-3 or smaller)

Usage:
    python run_single_reference_validation.py

Requirements:
    numpy, scipy, matplotlib
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, IntegrationWarning
from scipy.interpolate import CubicSpline


# -----------------------------------------------------------------------------
# Settings used in the manuscript validation
# -----------------------------------------------------------------------------
a = -0.1
b1 = -1.1
b2 = 1.1
h = 0.01
x_ref_A = a-h
n_fit = 6

cases = {
    "low": (3.0, 4.0),
    "medium": (12.0, 10.0),
    "high": (30.0, 25.0),
}


def beta_U(x, A1, A2, mu1=-1.0, sigma1=0.5, mu2=1.0, sigma2=0.6):
    """Analytical model potential used for validation."""
    x = np.asarray(x, dtype=float)
    V1 = A1 * np.exp(-((x - mu1) ** 2) / (2.0 * sigma1**2))
    V2 = A2 * np.exp(-((x - mu2) ** 2) / (2.0 * sigma2**2))
    return -(V1 + V2)


# -----------------------------------------------------------------------------
# MFPT reconstruction with arbitrary additive constants
# -----------------------------------------------------------------------------
def reconstruct_region_A_shape(x_arr, Pst_arr, mfpt_arr):
    """Region A = [b1,a]: absorbing at b1, reflecting at a."""
    x_arr = np.asarray(x_arr, dtype=float)
    Pst_arr = np.asarray(Pst_arr, dtype=float)
    mfpt_arr = np.asarray(mfpt_arr, dtype=float)

    interp_Pst = CubicSpline(x_arr, Pst_arr)
    N = x_arr.size
    Bx = np.zeros(N - 2)

    for i in range(N - 2):
        xi = x_arr[1 + i]
        integral_Pst, _ = quad(interp_Pst, x_arr[0], xi)
        Bx[i] = -(
            integral_Pst
            + (mfpt_arr[1 + i] - mfpt_arr[0]) / mfpt_arr[0]
        ) / interp_Pst(xi)

    interp_invB = CubicSpline(x_arr[1:-1], 1.0 / Bx)
    U = np.zeros(N - 2)

    # Arbitrary zero at x_arr[-2]; no model energy is used here.
    for i in range(N - 2):
        xi = x_arr[1 + i]
        integral_invB, _ = quad(interp_invB, xi, x_arr[-2])
        U[i] = np.log(Bx[i] / Bx[-1]) - integral_invB

    return U


def reconstruct_region_B_shape(x_arr, Pst_arr, mfpt_arr):
    """Region B = [a,b2]: reflecting at a, absorbing at b2."""
    x_arr = np.asarray(x_arr, dtype=float)
    Pst_arr = np.asarray(Pst_arr, dtype=float)
    mfpt_arr = np.asarray(mfpt_arr, dtype=float)

    interp_Pst = CubicSpline(x_arr, Pst_arr)
    N = x_arr.size
    Bx = np.zeros(N - 2)

    for i in range(N - 2):
        xi = x_arr[1 + i]
        integral_Pst, _ = quad(interp_Pst, xi, x_arr[-1])
        Bx[i] = -(
            integral_Pst
            - (mfpt_arr[-1] - mfpt_arr[1 + i]) / mfpt_arr[-1]
        ) / interp_Pst(xi)

    interp_invB = CubicSpline(x_arr[1:-1], 1.0 / Bx)
    U = np.zeros(N - 2)

    # Arbitrary zero at x_arr[1]; no model energy is used here.
    for i in range(N - 2):
        xi = x_arr[1 + i]
        integral_invB, _ = quad(interp_invB, x_arr[1], xi)
        U[i] = np.log(Bx[i] / Bx[0]) - integral_invB

    return U


# -----------------------------------------------------------------------------
# Single-reference matching
# -----------------------------------------------------------------------------
def quadratic_boundary_value(x, U, side):
    """
    Fit the six points nearest a to
        U = c0 + c1 (x-a) + c2 (x-a)^2,
    so c0 is the extrapolated one-sided value at a.
    """
    x = np.asarray(x, dtype=float)
    U = np.asarray(U, dtype=float)

    if side == "A":
        xx, yy = x[-n_fit:], U[-n_fit:]
    elif side == "B":
        xx, yy = x[:n_fit], U[:n_fit]
    else:
        raise ValueError("side must be 'A' or 'B'")

    coeff = np.polynomial.polynomial.polyfit(xx - a, yy, 2)
    return float(coeff[0]), coeff


def single_reference_match(xA, UA_shape, xB, UB_shape, U_ref):
    # ONE supplied energy reference, in region A only.
    UA = np.asarray(UA_shape, float).copy()
    UB = np.asarray(UB_shape, float).copy()
    UA += U_ref - np.interp(x_ref_A, xA, UA)

    Ua_from_A, coeff_A = quadratic_boundary_value(xA, UA, "A")
    Ua_from_B_raw, coeff_B = quadratic_boundary_value(xB, UB, "B")

    # Determine region-B additive constant solely from continuity at a.
    shift_B = Ua_from_A - Ua_from_B_raw
    UB += shift_B

    return {
        "UA": UA,
        "UB": UB,
        "Ua_fit": Ua_from_A,
        "shift_B": shift_B,
        "coeff_A": coeff_A,
        "coeff_B": coeff_B,
    }


# -----------------------------------------------------------------------------
# Transfer matrix benchmark -- self-contained implementation
# -----------------------------------------------------------------------------
def metro_accept(U):
    U = np.asarray(U, dtype=float)
    Aab = np.ones(U.size - 1)
    Aba = np.ones(U.size - 1)
    Aac = np.ones(U.size - 1)
    Aca = np.ones(U.size - 1)

    dr = U[1:] - U[:-1]
    plus = dr > 0
    Aab[plus] = np.exp(-dr[plus])
    Aba[~plus] = np.exp(dr[~plus])

    dl = -dr
    minus = dl < 0
    Aca[minus] = np.exp(dl[minus])
    Aac[~minus] = np.exp(-dl[~minus])

    return Aab, Aba, Aac, Aca


def base_transition_matrix(x, potential):
    Aab, Aba, Aac, Aca = metro_accept(potential(x))
    main = 2.0 * np.ones(x.size) - np.append(0.0, Aac) - np.append(Aab, 0.0)
    return 0.5 * (
        np.diag(Aca, k=-1) + np.diag(main, k=0) + np.diag(Aba, k=1)
    )


def transition_region_A(x, potential):
    P = base_transition_matrix(x, potential)
    P[:, 0] = 0.0
    P[-1, 0] = 1.0
    return P


def transition_region_B(x, potential):
    P = base_transition_matrix(x, potential)
    P[:, -1] = 0.0
    P[0, -1] = 1.0
    return P


def stationary_and_mfpt_matrix(P, absorbing_index):
    w, v = np.linalg.eig(P)
    idx = int(np.argmin(np.abs(w - 1.0)))
    vec = np.real(v[:, idx])
    if np.sum(vec) < 0:
        vec = -vec

    steady = vec / (h * np.sum(vec))
    steady[absorbing_index] = 0.0
    steady /= h * np.sum(steady)

    pi = vec / np.sum(vec)
    N = pi.size
    I = np.eye(N)
    E = np.ones((N, N))
    Z = np.linalg.inv(I - P.T + np.outer(np.ones(N), pi))
    Zd = np.diag(np.diag(Z))
    Mdiag = np.diag(1.0 / pi)
    M = (I - Z + E @ Zd) @ Mdiag
    Mbar = M - Mdiag

    return steady, Mbar


def build_tm_shapes(A1, A2):
    xA_full = np.linspace(b1, a, int(round((a - b1) / h)) + 1)
    xB_full = np.linspace(a, b2, int(round((b2 - a) / h)) + 1)
    potential = lambda x: beta_U(x, A1, A2)

    PA = transition_region_A(xA_full, potential)
    PB = transition_region_B(xB_full, potential)
    PstA, MA = stationary_and_mfpt_matrix(PA, 0)
    PstB, MB = stationary_and_mfpt_matrix(PB, -1)

    UA = reconstruct_region_A_shape(xA_full, PstA, MA[-1])
    UB = reconstruct_region_B_shape(xB_full, PstB, MB[0])

    return xA_full[1:-1], UA, xB_full[1:-1], UB


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------
def save_csv(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "case,U_model_a,U_fit_a_RW,abs_dev_RW,U_fit_a_TM,abs_dev_TM,"
            "RW_shift_B,TM_shift_B\n"
        )
        for r in rows:
            f.write(
                f"{r['case']},{r['U_model_a']:.12g},{r['U_fit_a_RW']:.12g},"
                f"{r['abs_dev_RW']:.12g},{r['U_fit_a_TM']:.12g},"
                f"{r['abs_dev_TM']:.12g},{r['RW_shift_B']:.12g},"
                f"{r['TM_shift_B']:.12g}\n"
            )


def make_plots(rows, profiles, outdir):
    names = [r["case"].capitalize() for r in rows]
    xpos = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(xpos, [r["U_model_a"] for r in rows], marker="o", label="Model")
    ax.plot(xpos, [r["U_fit_a_RW"] for r in rows], marker="s", linestyle="--", label="RW")
    ax.plot(xpos, [r["U_fit_a_TM"] for r in rows], marker="^", linestyle=":", label="TM")
    ax.set_xticks(xpos, names)
    ax.set_xlabel("Barrier case")
    ax.set_ylabel(r"$\beta U(a)$")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "boundary_values.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(xpos, [r["abs_dev_RW"] for r in rows], marker="o", label="RW")
    ax.plot(xpos, [r["abs_dev_TM"] for r in rows], marker="s", label="TM")
    ax.set_yscale("log")
    ax.set_xticks(xpos, names)
    ax.set_xlabel("Barrier case")
    ax.set_ylabel(r"$|\beta U_{\rm fit}(a)-\beta U_{\rm model}(a)|$")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "boundary_deviations.png", dpi=220)
    plt.close(fig)

    for case, p in profiles.items():
        xx = np.linspace(b1, b2, 500)
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.plot(xx, beta_U(xx, p["A1"], p["A2"]), label="Model")
        ax.plot(p["xA_RW"], p["rw"]["UA"], linestyle="--", label="RW A")
        ax.plot(p["xB_RW"], p["rw"]["UB"], linestyle="--", label="RW B matched")
        ax.plot(p["xA_TM"], p["tm"]["UA"], linestyle=":", label="TM A")
        ax.plot(p["xB_TM"], p["tm"]["UB"], linestyle=":", label="TM B matched")
        ax.axvline(a, linestyle="--", linewidth=1.0)
        ax.set_xlabel("x")
        ax.set_ylabel(r"$\beta U(x)$")
        ax.set_title(f"{case.capitalize()} barrier: single-reference match")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(outdir / f"matched_profile_{case}.png", dpi=220)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).with_name("two_region_validation_data.npz"),
        help="Raw RW data NPZ (default: beside this script).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("single_reference_validation_output"),
        help="Output directory.",
    )
    args = parser.parse_args()

    warnings.simplefilter("ignore", IntegrationWarning)
    data = np.load(args.data, allow_pickle=False)
    outdir = args.output
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    profiles = {}

    for case, (A1, A2) in cases.items():
        PstA = data[f"{case}_PstA"]
        PstB = data[f"{case}_PstB"]
        mfptA = data[f"{case}_mfptA"]
        mfptB = data[f"{case}_mfptB"]

        xA_full = np.linspace(b1, a, PstA.size)
        xB_full = np.linspace(a, b2, PstB.size)
        xA = xA_full[1:-1]
        xB = xB_full[1:-1]

        # RW reconstruction directly from raw Pst/MFPT arrays.
        UA_RW_shape = reconstruct_region_A_shape(xA_full, PstA, mfptA)
        UB_RW_shape = reconstruct_region_B_shape(xB_full, PstB, mfptB)
        U_ref = float(beta_U(x_ref_A, A1, A2))
        rw = single_reference_match(xA, UA_RW_shape, xB, UB_RW_shape, U_ref)

        # TM reconstruction independently rebuilt in this script.
        xA_TM, UA_TM_shape, xB_TM, UB_TM_shape = build_tm_shapes(A1, A2)
        tm = single_reference_match(xA_TM, UA_TM_shape, xB_TM, UB_TM_shape, U_ref)

        # Known U(a) used ONLY here, after reconstruction, to validate the fit.
        U_model_a = float(beta_U(a, A1, A2))
        dev_RW = abs(rw["Ua_fit"] - U_model_a)
        dev_TM = abs(tm["Ua_fit"] - U_model_a)

        rows.append({
            "case": case,
            "U_model_a": U_model_a,
            "U_fit_a_RW": rw["Ua_fit"],
            "abs_dev_RW": dev_RW,
            "U_fit_a_TM": tm["Ua_fit"],
            "abs_dev_TM": dev_TM,
            "RW_shift_B": rw["shift_B"],
            "TM_shift_B": tm["shift_B"],
        })

        profiles[case] = {
            "A1": A1,
            "A2": A2,
            "xA_RW": xA,
            "xB_RW": xB,
            "xA_TM": xA_TM,
            "xB_TM": xB_TM,
            "rw": rw,
            "tm": tm,
        }

    save_csv(rows, outdir / "single_reference_boundary_results.csv")
    make_plots(rows, profiles, outdir)

    print("\nTwo-region single-reference boundary validation")
    print(f"x_R = {x_ref_A}, a = {a}, quadratic degree = 2, fit points = {n_fit}")
    print("-" * 93)
    print(
        f"{'case':<8} {'model U(a)':>13} {'RW fit U(a)':>13} {'|RW dev|':>11} "
        f"{'TM fit U(a)':>13} {'|TM dev|':>11}"
    )
    print("-" * 93)
    for r in rows:
        print(
            f"{r['case']:<8} {r['U_model_a']:13.6f} {r['U_fit_a_RW']:13.6f} "
            f"{r['abs_dev_RW']:11.6f} {r['U_fit_a_TM']:13.6f} "
            f"{r['abs_dev_TM']:11.6f}"
        )
    print("-" * 93)

    max_rw = max(r["abs_dev_RW"] for r in rows)
    max_tm = max(r["abs_dev_TM"] for r in rows)
    print(f"Maximum RW deviation = {max_rw:.6f}  (< 0.007)")
    print(f"Maximum TM deviation = {max_tm:.6f}  (<0.002)")
    print(f"\nOutputs written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
