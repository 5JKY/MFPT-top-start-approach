import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ============================================================
# JCP style for the 3x3 Top-Start figure set
# Matches the final two-region convention
# ============================================================
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "lines.linewidth": 1.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def format_axes(ax):
    ax.minorticks_on()
    ax.tick_params(axis="both", which="major", direction="in",
                   top=True, right=True, length=5.5, width=1.1)
    ax.tick_params(axis="both", which="minor", direction="in",
                   top=True, right=True, length=3.0, width=0.9)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)


def add_legend(ax, loc="best"):
    legend = ax.legend(
        loc=loc,
        frameon=True,
        framealpha=0.85,
        fancybox=True,
        fontsize=13.5,
        handlelength=1.4,
        handletextpad=0.3,
        borderpad=0.2,
        labelspacing=0.15
    )
    legend.get_frame().set_linewidth(0.8)
    return legend


def add_panel_label(ax, label, position="upper-left"):
    positions = {
        "upper-left":  (0.03, 0.97, "left",  "top"),
        "upper-right": (0.97, 0.97, "right", "top"),
        "lower-left":  (0.03, 0.03, "left",  "bottom"),
        "lower-right": (0.97, 0.03, "right", "bottom"),
    }

    x, y, ha, va = positions[position]

    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=16,
        ha=ha,
        va=va
    )


# ============================================================
# High-barrier double-well potential
# ============================================================
def double_gaussian_potential(x, A1=30, mu1=-1, sigma1=0.5,
                              A2=25, mu2=1, sigma2=0.6):
    V1 = A1 * np.exp(-((x - mu1)**2) / (2 * sigma1**2))
    V2 = A2 * np.exp(-((x - mu2)**2) / (2 * sigma2**2))
    return -(V1 + V2)


beta_U = double_gaussian_potential
D0 = 0.01

a = -0.1
b1 = -1.1
b2 = 1.1
h = 0.01

N1 = int((a - b1) / h + 1)
N2 = int((b2 - a) / h + 1)

x1_arr = np.linspace(b1, a, N1)
x2_arr = np.linspace(a, b2, N2)

n1_arr = np.round(np.arange(b1, a + h/2, h), decimals=5)
n2_arr = np.round(np.arange(a, b2 + h/2, h), decimals=5)


# ============================================================
# Transfer-matrix calculation
# ============================================================
from transfer_matrix_reptile import TransferMatrix_InReAb, TransferMatrix_AbReIn

ari1_trans = TransferMatrix_AbReIn(h, x1_arr, beta_U, 0)
ira2_trans = TransferMatrix_InReAb(h, x2_arr, beta_U, 0)

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
# Load Top-Start random-walk data
# ============================================================
top_Pst_n1 = np.load("data/top_start_Pst_n1.npy")
top_Pst_n2 = np.load("data/top_start_Pst_n2.npy")

top_mfpt_n1 = np.load("data/top_start_mfpt_n1.npy")
top_mfpt_n2 = np.load("data/top_start_mfpt_n2.npy")


# ============================================================
# (g) Steady-state distribution
# ============================================================
fig, ax = plt.subplots(figsize=(3.3, 2.75), constrained_layout=True)

ax.plot(
    x1_arr,
    ari1_trans.steady_state,
    label="TM A",
    color="darkgreen"
)

ax.plot(
    x2_arr,
    ira2_trans.steady_state,
    label="TM B",
    color="darkorange"
)

ax.plot(
    n1_arr,
    top_Pst_n1,
    label="RW A",
    color="red",
    linestyle="--",
    linewidth=1.6,
    marker="o",
    markersize=4.5,
    markevery=15
)

ax.plot(
    n2_arr,
    top_Pst_n2,
    label="RW B",
    color="blue",
    linestyle="--",
    linewidth=1.6,
    marker="o",
    markersize=4.5,
    markevery=15
)

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$P_{\mathrm{st}}(x)$")

format_axes(ax)
add_legend(ax)
add_panel_label(ax, "(g)")

fig.savefig("top_Pst.pdf", bbox_inches="tight")
plt.close(fig)


# ============================================================
# (h) Mean first-passage time
# ============================================================
fig, ax = plt.subplots(figsize=(3.3, 2.75), constrained_layout=True)

ax.plot(
    x1_arr,
    trans_mfpt_n1,
    label="TM A",
    color="darkgreen"
)

ax.plot(
    x2_arr,
    trans_mfpt_n2,
    label="TM B",
    color="darkorange"
)

ax.plot(
    n1_arr,
    top_mfpt_n1,
    label="RW A",
    color="red",
    linestyle="--",
    linewidth=1.6,
    marker="o",
    markersize=4.5,
    markevery=15
)

ax.plot(
    n2_arr,
    top_mfpt_n2,
    label="RW B",
    color="blue",
    linestyle="--",
    linewidth=1.6,
    marker="o",
    markersize=4.5,
    markevery=15
)

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\tau(x)$")

format_axes(ax)
add_legend(ax)
add_panel_label(ax, "(h)")

fig.savefig("top_MFPT.pdf", bbox_inches="tight")
plt.close(fig)


# ============================================================
# Load Top-Start free-energy reconstruction
# ============================================================
top_reconst_n1 = np.load("data/top_start_reconst_n1.npy")
top_reconst_n2 = np.load("data/top_start_reconst_n2.npy")


# ============================================================
# (i) Top-Start free-energy reconstruction
# ============================================================
fig, ax = plt.subplots(figsize=(3.3, 2.75), constrained_layout=True)

x_full = np.arange(b1, b2, h)

ax.plot(
    x_full,
    beta_U(x_full),
    label="Model",
    color="black",
    zorder=2
)

ax.plot(
    n1_arr[1:-1],
    top_reconst_n1,
    label="RW A",
    color="red",
    linestyle="--",
    linewidth=1.6,
    marker="o",
    markersize=4.5,
    markevery=15,
    zorder=4
)

ax.plot(
    n2_arr[1:-1],
    top_reconst_n2,
    label="RW B",
    color="blue",
    linestyle="--",
    linewidth=1.6,
    marker="o",
    markersize=4.5,
    markevery=15,
    zorder=4
)

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\beta U(x)$")

format_axes(ax)
add_legend(ax)
add_panel_label(ax, "(i)")

fig.savefig("top_reconst.pdf", bbox_inches="tight")
plt.close(fig)

print("Generated: top_Pst.pdf, top_MFPT.pdf, top_reconst.pdf")
