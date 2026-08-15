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
    ax.tick_params(axis="both", which="both",
                   direction="in", top=True, right=True,
                   length=5, width=1)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

# ============================================================
# Medium-barrier double-well potential
# ============================================================
def double_gaussian_potential(x, A1=12, mu1=-1, sigma1=0.5,
                              A2=10, mu2=1, sigma2=0.6):
    V1 = A1 * np.exp(-((x - mu1)**2) / (2 * sigma1**2))
    V2 = A2 * np.exp(-((x - mu2)**2) / (2 * sigma2**2))
    return -(V1 + V2)

beta_U = double_gaussian_potential

D0 = 0.01
def D(x):
    return D0*x**0

# ============================================================
# Figure 1: Double-well potential
# ============================================================
x = np.linspace(-1.1, 1.1, 400)

fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
ax.plot(x, beta_U(x), color="black", linewidth=2.0)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\beta U(x)$")
format_axes(ax)
fig.savefig("double_well_potential.pdf", bbox_inches="tight")
plt.show()
plt.close(fig)

# ============================================================
# Transfer-matrix calculation
# ============================================================
a = -1.1
b2 = 1.1
h = 0.01
N2 = int((b2-a)/h+1)
x2_arr = np.linspace(a, b2, N2)
num_particles = 2000
init_position = a
hx = h
ht = hx**2/(2*D0)
n2_arr = np.arange(a, b2+h/2, h)
n2_arr = np.round(n2_arr, decimals=5)

from transfer_matrix_reptile import TransferMatrix_InReAb
ira2_trans = TransferMatrix_InReAb(h, x2_arr, beta_U, 0)
ira2_trans.steady_state[-1] = 0
ira2_trans.steady_state = ira2_trans.steady_state/(h*np.sum(ira2_trans.steady_state))

from mfpt_matrix_calc import mfpt_matrix_stable_ira
m2_bar = mfpt_matrix_stable_ira(ira2_trans)
delt_t = h**2/(2*D0)

# ============================================================
# Figure 2: Steady-state distribution
# ============================================================
Pst_n2 = np.load("data/reguera_Pst_n2.npy")

fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
ax.plot(x2_arr, ira2_trans.steady_state,
        label="Transfer matrix", color="darkorange")
ax.plot(n2_arr, Pst_n2, "--",
        label="RW", color="blue",
        marker="o", markersize=4, markevery=15)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$P_{\mathrm{st}}(x)$")
format_axes(ax)
ax.legend(loc="best", frameon=False)
fig.savefig("reguera_Pst.pdf", bbox_inches="tight")
plt.show()
plt.close(fig)

# ============================================================
# Figure 3: -ln[Pst(x)]
# ============================================================
valid_tm = ira2_trans.steady_state > 0
valid_rw = Pst_n2 > 0

fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
ax.plot(x2_arr[valid_tm],
        -np.log(ira2_trans.steady_state[valid_tm]),
        label="Transfer matrix", color="darkorange")
ax.plot(n2_arr[valid_rw],
        -np.log(Pst_n2[valid_rw]),
        "--", label="RW", color="blue",
        marker="o", markersize=4, markevery=15)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$-\ln[P_{\mathrm{st}}(x)]$")
format_axes(ax)
ax.legend(loc="best", frameon=False)
fig.savefig("reguera_lnPst.pdf", bbox_inches="tight")
plt.show()
plt.close(fig)

# ============================================================
# Figure 4: Mean first-passage time
# ============================================================
mfpt2_simu_arr = np.load("data/reguera_mfpt_n2.npy")

fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
ax.plot(x2_arr, delt_t*m2_bar[0],
        label="Transfer matrix", color="darkorange")
ax.plot(n2_arr, mfpt2_simu_arr, "--",
        label="RW", color="blue",
        marker="o", markersize=4, markevery=15)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\tau(x)$")
format_axes(ax)
ax.legend(loc="best", frameon=False)
fig.savefig("reguera_MFPT.pdf", bbox_inches="tight")
plt.show()
plt.close(fig)

# ============================================================
# Free-energy reconstruction
# ============================================================
from free_energy_reconst import reconstruct_energy_ra

trans_beta_Grec2_arr2 = reconstruct_energy_ra(
    x2_arr, beta_U=beta_U,
    Pst_arr=ira2_trans.steady_state,
    mfpt_arr=m2_bar[0])

trans_beta_GrecM_arr2 = np.log(
    (1-m2_bar[0]/m2_bar[0,-1])/ira2_trans.steady_state)

const_trans = beta_U(x2_arr)[2] - trans_beta_GrecM_arr2[2]
trans_beta_GrecM_arr2 += const_trans

simu_beta_Grec2_arr2 = np.load("data/reguera_reconst_n2.npy")

simu_beta_GrecM_arr2 = np.log(
    (1-mfpt2_simu_arr/mfpt2_simu_arr[-1])/Pst_n2)

const_simu = beta_U(x2_arr)[2] - simu_beta_GrecM_arr2[2]

# Preserved from your original code:
simu_beta_GrecM_arr2 += const_simu

# ============================================================
# Figure 5: Free-energy reconstruction
# ============================================================
fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)

ax.plot(x2_arr[1:-1], beta_U(x2_arr[1:-1]),
        label="Original", color="black", linewidth=2.0)

ax.plot(x2_arr[1:-1], trans_beta_Grec2_arr2, ":",
        label="TM-Reguera", color="darkorange")

ax.plot(x2_arr, trans_beta_GrecM_arr2, "-.",
        label="TM-simplified", color="red")

ax.plot(n2_arr[1:-1], simu_beta_Grec2_arr2, "--",
        label="RW-Reguera", color="blue",
        marker="o", markersize=4, markevery=15)

ax.plot(n2_arr, simu_beta_GrecM_arr2, "--",
        label="RW-simplified")

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\beta U(x)$")
format_axes(ax)
ax.legend(loc="best", frameon=False)

fig.savefig("reguera_reconst.pdf", bbox_inches="tight")
plt.show()
plt.close(fig)