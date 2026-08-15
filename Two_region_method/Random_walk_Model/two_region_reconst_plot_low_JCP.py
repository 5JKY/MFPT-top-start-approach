import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.sparse.linalg import spsolve
from scipy.integrate import quad
from scipy.interpolate import interp1d, PchipInterpolator, CubicSpline

# ============================================================
# JCP publication style
# ============================================================
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 16,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "lines.linewidth": 1.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def format_axes(ax):
    """Apply consistent publication-style formatting."""
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        length=5,
        width=1
    )
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


# ============================================================
# Low-barrier double-well potential
# ============================================================
def double_gaussian_potential(
    x,
    A1=3,
    mu1=-1,
    sigma1=0.5,
    A2=4,
    mu2=1,
    sigma2=0.6
):
    V1 = A1 * np.exp(-((x - mu1)**2) / (2 * sigma1**2))
    V2 = A2 * np.exp(-((x - mu2)**2) / (2 * sigma2**2))
    return -(V1 + V2)


beta_U = double_gaussian_potential

D0 = 0.01


def D(x):
    return D0 * x**0


# ============================================================
# Figure 1: Double-well potential and dividing point
# ============================================================
x = np.linspace(-1.1, 1.1, 400)

fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)

ax.plot(
    x,
    beta_U(x),
    color="black",
    linewidth=2.0
)

# Dividing point between regions A and B
a = -0.1

ax.scatter(
    a,
    beta_U(a),
    color="black",
    s=40,
    zorder=5
)

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\beta U(x)$")

format_axes(ax)

fig.savefig(
    "two_region_double_well_potential.pdf",
    bbox_inches="tight"
)

plt.show()
plt.close(fig)


# ============================================================
# Two-region transfer-matrix calculation
# ============================================================
b1 = -1.1
b2 = 1.1
h = 0.01

N1 = int((a - b1) / h + 1)
N2 = int((b2 - a) / h + 1)

x1_arr = np.linspace(b1, a, N1)
x2_arr = np.linspace(a, b2, N2)

num_particles = 2000
init_position = a

hx = h
ht = hx**2 / (2 * D0)

n1_arr = np.arange(b1, a + h/2, h)
n1_arr = np.round(n1_arr, decimals=5)

n2_arr = np.arange(a, b2 + h/2, h)
n2_arr = np.round(n2_arr, decimals=5)

from transfer_matrix_reptile import (
    TransferMatrix_InReAb,
    TransferMatrix_AbReIn
)

ari1_trans = TransferMatrix_AbReIn(
    h,
    x1_arr,
    beta_U,
    0
)

ira2_trans = TransferMatrix_InReAb(
    h,
    x2_arr,
    beta_U,
    0
)

ari1_trans.steady_state[0] = 0
ira2_trans.steady_state[-1] = 0

ari1_trans.steady_state = (
    ari1_trans.steady_state /
    (h * np.sum(ari1_trans.steady_state))
)

ira2_trans.steady_state = (
    ira2_trans.steady_state /
    (h * np.sum(ira2_trans.steady_state))
)

from mfpt_matrix_calc import mfpt_matrix

m1_bar = mfpt_matrix(ari1_trans)
m2_bar = mfpt_matrix(ira2_trans)

delt_t = h**2 / (2 * D0)

trans_mfpt_n1 = delt_t * m1_bar[-1]
trans_mfpt_n2 = delt_t * m2_bar[0]


# ============================================================
# Load random-walk data
# ============================================================
two_Pst_n1 = np.load("data/two_region_Pst_n1.npy")
two_Pst_n2 = np.load("data/two_region_Pst_n2.npy")

two_mfpt_n1 = np.load("data/two_region_mfpt_n1.npy")
two_mfpt_n2 = np.load("data/two_region_mfpt_n2.npy")


# ============================================================
# Figure 2: Steady-state distribution
# ============================================================
fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)

ax.plot(
    x1_arr,
    ari1_trans.steady_state,
    label="Transfer matrix A",
    color="darkgreen"
)

ax.plot(
    x2_arr,
    ira2_trans.steady_state,
    label="Transfer matrix B",
    color="darkorange"
)

ax.plot(
    n1_arr,
    two_Pst_n1,
    "--",
    label="RW A",
    color="red",
    marker="o",
    markersize=4,
    markevery=15
)

ax.plot(
    n2_arr,
    two_Pst_n2,
    "--",
    label="RW B",
    color="blue",
    marker="o",
    markersize=4,
    markevery=15
)

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$P_{\mathrm{st}}(x)$")

format_axes(ax)

ax.legend(
    loc="best",
    frameon=False
)

fig.savefig(
    "two_Pst.pdf",
    bbox_inches="tight"
)

plt.show()
plt.close(fig)


# ============================================================
# Figure 3: -ln[Pst(x)]
# ============================================================
# Exclude absorbing-boundary points where Pst = 0.
valid_tm_A = ari1_trans.steady_state > 0
valid_tm_B = ira2_trans.steady_state > 0
valid_rw_A = two_Pst_n1 > 0
valid_rw_B = two_Pst_n2 > 0

fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)

ax.plot(
    x1_arr[valid_tm_A],
    -np.log(ari1_trans.steady_state[valid_tm_A]),
    label="Transfer matrix A",
    color="darkgreen"
)

ax.plot(
    x2_arr[valid_tm_B],
    -np.log(ira2_trans.steady_state[valid_tm_B]),
    label="Transfer matrix B",
    color="darkorange"
)

ax.plot(
    n1_arr[valid_rw_A],
    -np.log(two_Pst_n1[valid_rw_A]),
    "--",
    label="RW A",
    color="red",
    marker="o",
    markersize=4,
    markevery=15
)

ax.plot(
    n2_arr[valid_rw_B],
    -np.log(two_Pst_n2[valid_rw_B]),
    "--",
    label="RW B",
    color="blue",
    marker="o",
    markersize=4,
    markevery=15
)

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$-\ln[P_{\mathrm{st}}(x)]$")

format_axes(ax)

ax.legend(
    loc="best",
    frameon=False
)

fig.savefig(
    "two_lnPst.pdf",
    bbox_inches="tight"
)

plt.show()
plt.close(fig)


# ============================================================
# Figure 4: Mean first-passage time
# ============================================================
fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)

ax.plot(
    x1_arr,
    trans_mfpt_n1,
    label="MFPT matrix A",
    color="darkgreen"
)

ax.plot(
    x2_arr,
    trans_mfpt_n2,
    label="MFPT matrix B",
    color="darkorange"
)

ax.plot(
    n1_arr,
    two_mfpt_n1,
    "--",
    label="RW A",
    color="red",
    marker="o",
    markersize=4,
    markevery=15
)

ax.plot(
    n2_arr,
    two_mfpt_n2,
    "--",
    label="RW B",
    color="blue",
    marker="o",
    markersize=4,
    markevery=15
)

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\tau(x)$")

format_axes(ax)

ax.legend(
    loc="best",
    frameon=False
)

fig.savefig(
    "two_MFPT.pdf",
    bbox_inches="tight"
)

plt.show()
plt.close(fig)


# ============================================================
# Free-energy reconstruction
# ============================================================
from free_energy_reconst import (
    reconstruct_energy_ar,
    reconstruct_energy_ra
)

trans_reconst_n1 = reconstruct_energy_ar(
    x1_arr,
    beta_U=beta_U,
    Pst_arr=ari1_trans.steady_state,
    mfpt_arr=m1_bar[-1]
)

trans_reconst_n2 = reconstruct_energy_ra(
    x2_arr,
    beta_U=beta_U,
    Pst_arr=ira2_trans.steady_state,
    mfpt_arr=m2_bar[0]
)

two_reconst_n1 = np.load(
    "data/two_region_reconst_n1.npy"
)

two_reconst_n2 = np.load(
    "data/two_region_reconst_n2.npy"
)


# ============================================================
# Figure 5: Two-region free-energy reconstruction
# ============================================================
fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)

x_full = np.arange(b1, b2, h)

ax.plot(
    x_full,
    beta_U(x_full),
    label="Original",
    color="black",
    linewidth=2.0
)

ax.plot(
    x1_arr[1:-1],
    trans_reconst_n1,
    ":",
    label="MFPT matrix A",
    color="darkgreen",
    marker="^",
    markersize=4,
    markevery=20
)

ax.plot(
    x2_arr[1:-1],
    trans_reconst_n2,
    ":",
    label="MFPT matrix B",
    color="darkorange",
    marker="^",
    markersize=4,
    markevery=20
)

ax.plot(
    n1_arr[1:-1],
    two_reconst_n1,
    "--",
    label="RW A",
    color="red",
    marker="o",
    markersize=4,
    markevery=15
)

ax.plot(
    n2_arr[1:-1],
    two_reconst_n2,
    "--",
    label="RW B",
    color="blue",
    marker="o",
    markersize=4,
    markevery=15
)

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\beta U(x)$")

format_axes(ax)

ax.legend(
    loc="best",
    frameon=False
)

fig.savefig(
    "two_reconst.pdf",
    bbox_inches="tight"
)

plt.show()
plt.close(fig)
