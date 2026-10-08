#!/usr/bin/env python3
"""
Reproduce the single-reference boundary-matching validation reported for the
three TWO-REGION random-walk examples (low, medium, high barrier).

What this script does
---------------------
1. Reads the saved RW P_st and MFPT arrays directly from the three two-region
   branches contained in MFPT_all_branches.bundle.
2. Reconstructs the RW free-energy SHAPES with arbitrary zero constants.
   Therefore no model-potential value in region B is used.
3. Supplies exactly ONE model reference value, beta U(x_R), at x_R = -1.0 in
   region A.
4. Fits the six reconstructed points closest to a = -0.1 in each region with
   a degree-2 polynomial in (x-a), and matches region B to region A at a.
5. Independently rebuilds the transfer-matrix (TM) benchmark and performs the
   same single-reference calculation.
6. Only AFTER reconstruction, evaluates the known model beta U(a) to measure
   the validation deviation.

Expected reported deviations
----------------------------
RW:  low ~0.0822, medium ~0.0160, high ~0.0683  -> all < 0.1
TM:  low ~0.00523, medium ~0.00504, high ~0.00263 -> ~5e-3 or smaller

Requirements
------------
Python 3, numpy, scipy, matplotlib, and Git on PATH.

Usage
-----
Place this file next to MFPT_all_branches.bundle and run:

    python validate_single_reference_reported_results.py

or specify the bundle explicitly:

    python validate_single_reference_reported_results.py \
        --bundle /path/to/MFPT_all_branches.bundle

Outputs are written to ./single_reference_validation_output by default.
"""

from __future__ import annotations

import argparse
import io
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, IntegrationWarning
from scipy.interpolate import CubicSpline


# -----------------------------------------------------------------------------
# Fixed model / reconstruction settings used in the manuscript tests
# -----------------------------------------------------------------------------
A = -0.1
B1 = -1.1
B2 = 1.1
H = 0.01
X_REF_A = -1.0
N_FIT = 6
D0 = 0.01

CASES = {
    "low": {
        "branch": "redo_graph_from_two_region_low",
        "A1": 3.0,
        "A2": 4.0,
    },
    "medium": {
        "branch": "redo_graph_from_two_region_medium",
        "A1": 12.0,
        "A2": 10.0,
    },
    "high": {
        "branch": "redo_graph_from_two_region_high",
        "A1": 30.0,
        "A2": 25.0,
    },
}

DATA_PREFIX = "Two_region_method/Random_walk_Model/data"


# -----------------------------------------------------------------------------
# Known model potential -- used at x_R for the single reference and at a only
# afterward for validation.
# -----------------------------------------------------------------------------
def beta_U(x, A1, A2, mu1=-1.0, sigma1=0.5, mu2=1.0, sigma2=0.6):
    x = np.asarray(x, dtype=float)
    V1 = A1 * np.exp(-((x - mu1) ** 2) / (2.0 * sigma1**2))
    V2 = A2 * np.exp(-((x - mu2) ** 2) / (2.0 * sigma2**2))
    return -(V1 + V2)


# -----------------------------------------------------------------------------
# Git-bundle helpers
# -----------------------------------------------------------------------------
def run(cmd, *, cwd=None, capture=False):
    kwargs = {
        "cwd": cwd,
        "check": True,
    }
    if capture:
        kwargs["stdout"] = subprocess.PIPE
    return subprocess.run(cmd, **kwargs)


def prepare_bundle_repo(bundle_path: Path, workdir: Path) -> Path:
    """Clone main and fetch all remote-tracking refs stored in the bundle."""
    repo = workdir / "repo"
    run(["git", "clone", "--quiet", str(bundle_path), str(repo)])
    run([
        "git", "-C", str(repo), "fetch", "--quiet", str(bundle_path),
        "refs/remotes/origin/*:refs/remotes/bundle/*",
    ])
    return repo


def load_npy_from_branch(repo: Path, branch: str, filename: str) -> np.ndarray:
    git_path = f"{DATA_PREFIX}/{filename}"
    result = run(
        [
            "git", "-c", f"safe.directory={repo}", "-C", str(repo),
            "show", f"remotes/bundle/{branch}:{git_path}",
        ],
        capture=True,
    )
    return np.load(io.BytesIO(result.stdout), allow_pickle=False)


# -----------------------------------------------------------------------------
# MFPT free-energy reconstruction with arbitrary additive constants.
# These are the same equations as free_energy_reconst.py, except that the
# reference-energy term is set to zero so that no second model reference enters.
# -----------------------------------------------------------------------------
def reconstruct_region_A_shape(x_arr, Pst_arr, mfpt_arr):
    """Region A = [b1,a], absorbing left / reflecting right."""
    x_arr = np.asarray(x_arr, dtype=float)
    Pst_arr = np.asarray(Pst_arr, dtype=float)
    mfpt_arr = np.asarray(mfpt_arr, dtype=float)

    interp_Pst = CubicSpline(x_arr, Pst_arr)
    N = x_arr.size
    Bx = np.zeros(N - 2)

    for i in range(N - 2):
        x_i = x_arr[1 + i]
        integral_Pst, _ = quad(interp_Pst, x_arr[0], x_i)
        Bx[i] = -(
            integral_Pst
            + (mfpt_arr[1 + i] - mfpt_arr[0]) / mfpt_arr[0]
        ) / interp_Pst(x_i)

    interp_invB = CubicSpline(x_arr[1:-1], 1.0 / Bx)
    U_shape = np.zeros(N - 2)

    # Arbitrary zero at x_arr[-2].
    for i in range(N - 2):
        x_i = x_arr[1 + i]
        integral_invB, _ = quad(interp_invB, x_i, x_arr[-2])
        U_shape[i] = np.log(Bx[i] / Bx[-1]) - integral_invB

    return U_shape


def reconstruct_region_B_shape(x_arr, Pst_arr, mfpt_arr):
    """Region B = [a,b2], reflecting left / absorbing right."""
    x_arr = np.asarray(x_arr, dtype=float)
    Pst_arr = np.asarray(Pst_arr, dtype=float)
    mfpt_arr = np.asarray(mfpt_arr, dtype=float)

    interp_Pst = CubicSpline(x_arr, Pst_arr)
    N = x_arr.size
    Bx = np.zeros(N - 2)

    for i in range(N - 2):
        x_i = x_arr[1 + i]
        integral_Pst, _ = quad(interp_Pst, x_i, x_arr[-1])
        Bx[i] = -(
            integral_Pst
            - (mfpt_arr[-1] - mfpt_arr[1 + i]) / mfpt_arr[-1]
        ) / interp_Pst(x_i)

    interp_invB = CubicSpline(x_arr[1:-1], 1.0 / Bx)
    U_shape = np.zeros(N - 2)

    # Arbitrary zero at x_arr[1].
    for i in range(N - 2):
        x_i = x_arr[1 + i]
        integral_invB, _ = quad(interp_invB, x_arr[1], x_i)
        U_shape[i] = np.log(Bx[i] / Bx[0]) - integral_invB

    return U_shape


# -----------------------------------------------------------------------------
# Six-point quadratic matching
# -----------------------------------------------------------------------------
def quadratic_boundary_value(x, U, a=A, n_fit=N_FIT, side="A"):
    x = np.asarray(x, dtype=float)
    U = np.asarray(U, dtype=float)

    if side.upper() == "A":
        xx = x[-n_fit:]
        yy = U[-n_fit:]
    elif side.upper() == "B":
        xx = x[:n_fit]
        yy = U[:n_fit]
    else:
        raise ValueError("side must be 'A' or 'B'")

    # U = c0 + c1(x-a) + c2(x-a)^2; therefore c0 = U_fit(a).
    coeff = np.polynomial.polynomial.polyfit(xx - a, yy, 2)
    return float(coeff[0]), coeff


def single_reference_match(xA, UA_shape, xB, UB_shape, U_ref_A):
    """
    Supply only U(x_R) in region A; determine B's additive constant by matching
    the two extrapolated one-sided values at a.
    """
    UA = np.asarray(UA_shape, float).copy()
    UB = np.asarray(UB_shape, float).copy()

    # One and only one supplied free-energy reference.
    UA += U_ref_A - np.interp(X_REF_A, xA, UA)

    Ua_A, coeff_A = quadratic_boundary_value(xA, UA, side="A")
    Ua_B_raw, coeff_B = quadratic_boundary_value(xB, UB, side="B")

    shift_B = Ua_A - Ua_B_raw
    UB_matched = UB + shift_B

    return {
        "UA": UA,
        "UB": UB_matched,
        "Ua_fit": Ua_A,
        "Ua_B_before_shift": Ua_B_raw,
        "shift_B": shift_B,
        "coeff_A": coeff_A,
        "coeff_B": coeff_B,
    }


# -----------------------------------------------------------------------------
# Transfer-matrix benchmark, implemented here so this file is self-contained.
# Matrix convention follows the repository's transfer_matrix_reptile.py.
# -----------------------------------------------------------------------------
def metro_accept(U):
    U = np.asarray(U, dtype=float)
    Aab = np.ones(U.size - 1)
    Aba = np.ones(U.size - 1)
    Aac = np.ones(U.size - 1)
    Aca = np.ones(U.size - 1)

    right_difference = U[1:] - U[:-1]
    plus = right_difference > 0
    Aab[plus] = np.exp(-right_difference[plus])
    Aba[~plus] = np.exp(right_difference[~plus])

    left_difference = -right_difference
    minus = left_difference < 0
    Aca[minus] = np.exp(left_difference[minus])
    Aac[~minus] = np.exp(-left_difference[~minus])

    return Aab, Aba, Aac, Aca


def base_transition_matrix(x, potential):
    Aab, Aba, Aac, Aca = metro_accept(potential(x))
    main_diag = (
        2.0 * np.ones(x.size)
        - np.append(0.0, Aac)
        - np.append(Aab, 0.0)
    )
    return 0.5 * (
        np.diag(Aca, k=-1)
        + np.diag(main_diag, k=0)
        + np.diag(Aba, k=1)
    )


def transition_region_A(x, potential):
    # Absorb/recycle at b1 -> inject at a.
    P = base_transition_matrix(x, potential)
    P[:, 0] = 0.0
    P[-1, 0] = 1.0
    return P


def transition_region_B(x, potential):
    # Absorb/recycle at b2 -> inject at a.
    P = base_transition_matrix(x, potential)
    P[:, -1] = 0.0
    P[0, -1] = 1.0
    return P


def stationary_and_mfpt_matrix(P, absorbing_index, h=H):
    # Right eigenvector of P at eigenvalue 1, matching repository convention.
    w, v = np.linalg.eig(P)
    idx = int(np.argmin(np.abs(w - 1.0)))
    vec = np.real(v[:, idx])
    if np.sum(vec) < 0:
        vec = -vec

    # Steady-state density used in reconstruction.
    steady = vec / (h * np.sum(vec))
    steady[absorbing_index] = 0.0
    steady /= h * np.sum(steady)

    # Repository MFPT matrix construction.
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


def build_tm_reconstructions(A1, A2):
    xA_full = np.linspace(B1, A, int(round((A - B1) / H)) + 1)
    xB_full = np.linspace(A, B2, int(round((B2 - A) / H)) + 1)

    potential = lambda x: beta_U(x, A1, A2)

    PA = transition_region_A(xA_full, potential)
    PB = transition_region_B(xB_full, potential)

    PstA, MA = stationary_and_mfpt_matrix(PA, absorbing_index=0)
    PstB, MB = stationary_and_mfpt_matrix(PB, absorbing_index=-1)

    # Same rows used by the repository scripts. Overall MFPT time scale cancels
    # from the reconstruction equations, so Mbar is sufficient here.
    mfptA = MA[-1]
    mfptB = MB[0]

    UA_shape = reconstruct_region_A_shape(xA_full, PstA, mfptA)
    UB_shape = reconstruct_region_B_shape(xB_full, PstB, mfptB)

    return xA_full[1:-1], UA_shape, xB_full[1:-1], UB_shape


# -----------------------------------------------------------------------------
# RW data from bundle
# -----------------------------------------------------------------------------
def build_rw_reconstructions(repo: Path, branch: str):
    PstA = load_npy_from_branch(repo, branch, "two_region_Pst_n1.npy")
    PstB = load_npy_from_branch(repo, branch, "two_region_Pst_n2.npy")
    mfptA = load_npy_from_branch(repo, branch, "two_region_mfpt_n1.npy")
    mfptB = load_npy_from_branch(repo, branch, "two_region_mfpt_n2.npy")

    xA_full = np.linspace(B1, A, PstA.size)
    xB_full = np.linspace(A, B2, PstB.size)

    UA_shape = reconstruct_region_A_shape(xA_full, PstA, mfptA)
    UB_shape = reconstruct_region_B_shape(xB_full, PstB, mfptB)

    return xA_full[1:-1], UA_shape, xB_full[1:-1], UB_shape


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
def save_csv(rows, path: Path):
    header = (
        "case,U_model_a,U_fit_a_RW,abs_dev_RW,U_fit_a_TM,abs_dev_TM,"
        "RW_shift_B,TM_shift_B\n"
    )
    with path.open("w", encoding="utf-8") as f:
        f.write(header)
        for r in rows:
            f.write(
                f"{r['case']},{r['U_model_a']:.12g},{r['U_fit_a_RW']:.12g},"
                f"{r['abs_dev_RW']:.12g},{r['U_fit_a_TM']:.12g},"
                f"{r['abs_dev_TM']:.12g},{r['RW_shift_B']:.12g},"
                f"{r['TM_shift_B']:.12g}\n"
            )


def make_plots(rows, profiles, outdir: Path):
    names = [r["case"].capitalize() for r in rows]
    x = np.arange(len(rows))

    # Boundary values
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(x, [r["U_model_a"] for r in rows], marker="o", label="Model")
    ax.plot(x, [r["U_fit_a_RW"] for r in rows], marker="s", linestyle="--", label="RW")
    ax.plot(x, [r["U_fit_a_TM"] for r in rows], marker="^", linestyle=":", label="TM")
    ax.set_xticks(x, names)
    ax.set_xlabel("Barrier case")
    ax.set_ylabel(r"$\beta U(a)$")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "01_boundary_values.png", dpi=220)
    plt.close(fig)

    # Absolute deviations
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(x, [r["abs_dev_RW"] for r in rows], marker="o", label="RW")
    ax.plot(x, [r["abs_dev_TM"] for r in rows], marker="s", label="TM")
    ax.set_yscale("log")
    ax.set_xticks(x, names)
    ax.set_xlabel("Barrier case")
    ax.set_ylabel(r"$|\beta U_{\rm fit}(a)-\beta U_{\rm model}(a)|$")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "02_boundary_deviations.png", dpi=220)
    plt.close(fig)

    # One matched-profile figure per barrier case (no subplots).
    for case, p in profiles.items():
        xx = np.linspace(B1, B2, 500)
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.plot(xx, beta_U(xx, p["A1"], p["A2"]), label="Model")
        ax.plot(p["xA_RW"], p["rw"]["UA"], linestyle="--", label="RW A")
        ax.plot(p["xB_RW"], p["rw"]["UB"], linestyle="--", label="RW B matched")
        ax.plot(p["xA_TM"], p["tm"]["UA"], linestyle=":", label="TM A")
        ax.plot(p["xB_TM"], p["tm"]["UB"], linestyle=":", label="TM B matched")
        ax.axvline(A, linestyle="--", linewidth=1.0)
        ax.set_xlabel("x")
        ax.set_ylabel(r"$\beta U(x)$")
        ax.set_title(f"{case.capitalize()} barrier: one-reference matching")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(outdir / f"03_matched_profile_{case}.png", dpi=220)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce the reported two-region single-reference validation."
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path(__file__).with_name("MFPT_all_branches.bundle"),
        help="Path to MFPT_all_branches.bundle (default: beside this script).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("single_reference_validation_output"),
        help="Output directory.",
    )
    args = parser.parse_args()

    bundle = args.bundle.expanduser().resolve()
    outdir = args.output.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if not bundle.exists():
        raise FileNotFoundError(
            f"Bundle not found: {bundle}\n"
            "Place MFPT_all_branches.bundle beside this script or use --bundle."
        )

    warnings.simplefilter("ignore", IntegrationWarning)

    rows = []
    profiles = {}

    with tempfile.TemporaryDirectory(prefix="mfpt_single_ref_") as tmp:
        repo = prepare_bundle_repo(bundle, Path(tmp))

        for case, cfg in CASES.items():
            A1 = cfg["A1"]
            A2 = cfg["A2"]
            branch = cfg["branch"]
            potential = lambda x, A1=A1, A2=A2: beta_U(x, A1, A2)

            # RW from saved P_st and MFPT -- not from saved two-reference U arrays.
            xA_RW, UA_RW_shape, xB_RW, UB_RW_shape = build_rw_reconstructions(
                repo, branch
            )
            rw = single_reference_match(
                xA_RW, UA_RW_shape, xB_RW, UB_RW_shape,
                U_ref_A=float(potential(X_REF_A)),
            )

            # Independent TM benchmark.
            xA_TM, UA_TM_shape, xB_TM, UB_TM_shape = build_tm_reconstructions(
                A1, A2
            )
            tm = single_reference_match(
                xA_TM, UA_TM_shape, xB_TM, UB_TM_shape,
                U_ref_A=float(potential(X_REF_A)),
            )

            # Validation ONLY: known model at a is evaluated after reconstruction.
            U_model_a = float(potential(A))
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
                "xA_RW": xA_RW,
                "xB_RW": xB_RW,
                "xA_TM": xA_TM,
                "xB_TM": xB_TM,
                "rw": rw,
                "tm": tm,
            }

    save_csv(rows, outdir / "single_reference_boundary_results.csv")
    make_plots(rows, profiles, outdir)

    # Terminal report
    print("\nTwo-region single-reference boundary validation")
    print(f"Reference: x_R = {X_REF_A}; boundary: a = {A}; fit: {N_FIT} points, degree 2")
    print("-" * 96)
    print(
        f"{'case':<8} {'model U(a)':>13} {'RW fit U(a)':>13} {'|RW dev|':>11} "
        f"{'TM fit U(a)':>13} {'|TM dev|':>11}"
    )
    print("-" * 96)
    for r in rows:
        print(
            f"{r['case']:<8} {r['U_model_a']:13.6f} {r['U_fit_a_RW']:13.6f} "
            f"{r['abs_dev_RW']:11.6f} {r['U_fit_a_TM']:13.6f} "
            f"{r['abs_dev_TM']:11.6f}"
        )
    print("-" * 96)

    max_rw = max(r["abs_dev_RW"] for r in rows)
    max_tm = max(r["abs_dev_TM"] for r in rows)
    print(f"Maximum RW deviation = {max_rw:.6f}  -> within 0.1")
    print(f"Maximum TM deviation = {max_tm:.6f}  -> approximately 5e-3")
    print(f"\nSaved results and figures to: {outdir}")


if __name__ == "__main__":
    main()
