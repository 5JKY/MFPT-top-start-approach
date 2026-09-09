import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 17,
    "axes.labelsize": 17,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "lines.linewidth": 2.0,
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

def add_legend(ax):
    legend = ax.legend(
        loc="best",
        frameon=True,
        framealpha=0.85,
        fancybox=True,
        fontsize=16,
        handlelength=1.2,
        handletextpad=0.3,
        borderpad=0.25,
        labelspacing=0.2)

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
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=18, ha=ha, va=va)

def double_gaussian_potential(x, A1=30, mu1=-1, sigma1=0.5,
                              A2=25, mu2=1, sigma2=0.6):
    V1 = A1 * np.exp(-((x - mu1)**2) / (2 * sigma1**2))
    V2 = A2 * np.exp(-((x - mu2)**2) / (2 * sigma2**2))
    return -(V1 + V2)

beta_U = double_gaussian_potential
D0 = 0.01

a = -1.1
b2 = 1.1
h = 0.01
N2 = int((b2 - a) / h + 1)
x2_arr = np.linspace(a, b2, N2)
n2_arr = np.round(np.arange(a, b2 + h/2, h), decimals=5)

from transfer_matrix_reptile import TransferMatrix_InReAb
ira2_trans = TransferMatrix_InReAb(h, x2_arr, beta_U, 0)
ira2_trans.steady_state[-1] = 0
ira2_trans.steady_state = (
    ira2_trans.steady_state /
    (h * np.sum(ira2_trans.steady_state))
)

from mfpt_matrix_calc import mfpt_matrix_stable_ira
m2_bar = mfpt_matrix_stable_ira(ira2_trans)
delt_t = h**2 / (2 * D0)

# Pst_n2 = np.load("data/reguera_Pst_n2.npy")
# mfpt2_simu_arr = np.load("data/reguera_mfpt_n2.npy")

# (a) Pst
fig, ax = plt.subplots(figsize=(3.3, 2.75), constrained_layout=True)
ax.plot(x2_arr, ira2_trans.steady_state, label="TM", color="darkorange")
# ax.plot(n2_arr, Pst_n2, "--", label="RW", color="blue",
#         marker="o", markersize=5.0, markevery=15)
# ax.plot(
#     n2_arr, Pst_n2,
#     label="RW",
#     color="blue",
#     linestyle='--',
#     linewidth=1.6,
#     marker="o",
#     markersize=4.5,
#     markevery=15
# )
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$P_{\mathrm{st}}(x)$")
format_axes(ax)
add_legend(ax)
add_panel_label(ax, "(e)", position="lower-left")
fig.savefig("reguera_Pst_high.pdf", bbox_inches="tight")
plt.show()
plt.close(fig)

# (b) -ln Pst
valid_TM = ira2_trans.steady_state > 0
# valid_RW = Pst_n2 > 0
fig, ax = plt.subplots(figsize=(3.3, 2.75), constrained_layout=True)
ax.plot(x2_arr[valid_TM], -np.log(ira2_trans.steady_state[valid_TM]),
        label="TM", color="darkorange")
# ax.plot(n2_arr[valid_RW], -np.log(Pst_n2[valid_RW]), "--",
#         label="RW", color="blue", marker="o", markersize=5.0, markevery=15)
# ax.plot(
#     n2_arr[valid_RW], -np.log(Pst_n2[valid_RW]),
#     label="RW",
#     color="blue",
#     linestyle='--',
#     linewidth=1.6,
#     marker="o",
#     markersize=4.5,
#     markevery=15
# )
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$-\ln[P_{\mathrm{st}}(x)]$")
format_axes(ax)
add_legend(ax)
add_panel_label(ax, "(f)")
fig.savefig("reguera_lnPst_high.pdf", bbox_inches="tight")
plt.show()
plt.close(fig)

# (c) MFPT
fig, ax = plt.subplots(figsize=(3.3, 2.75), constrained_layout=True)
ax.plot(x2_arr, delt_t * m2_bar[0] / 1e9, label="TM", color="darkorange")
# ax.plot(n2_arr, mfpt2_simu_arr, "--", label="RW", color="blue",
#         marker="o", markersize=5.0, markevery=15)
# ax.plot(
#     n2_arr,
#     mfpt2_simu_arr / 1e4,
#     label="RW",
#     color="blue",
#     linestyle='--',
#     linewidth=1.6,
#     marker="o",
#     markersize=4.5,
#     markevery=15
# )
ax.set_xlabel(r"$x$")
# ax.set_ylabel(r"$\tau(x)\times 10^9$")
ax.set_ylabel(r"$\tau(x) / 10^9$")
# ax.set_ylabel(r"$\tau(x)$")
# ax.ticklabel_format(axis="y", style="sci", scilimits=(4, 4), useMathText=True)
format_axes(ax)
add_legend(ax)
add_panel_label(ax, "(g)")
fig.savefig("reguera_MFPT_high.pdf", bbox_inches="tight")
plt.show()
plt.close(fig)

# Reguera reconstruction only; simplified reconstruction removed
from free_energy_reconst import reconstruct_energy_ra
trans_beta_Grec_arr = reconstruct_energy_ra(
    x2_arr,
    beta_U=beta_U,
    Pst_arr=ira2_trans.steady_state,
    mfpt_arr=m2_bar[0]
)
# simu_beta_Grec_arr = np.load("data/reguera_reconst_n2.npy")


# Save numerical data for diagnostics
tau_steps = np.asarray(m2_bar[0], dtype=float)
tau_time = delt_t * tau_steps
Pst = np.asarray(ira2_trans.steady_state, dtype=float)
model_U = beta_U(x2_arr)
reconst_U = np.asarray(trans_beta_Grec_arr, dtype=float)


# (d) Reconstructed free energy
fig, ax = plt.subplots(figsize=(3.3, 2.75), constrained_layout=True)
ax.plot(x2_arr[1:-1], beta_U(x2_arr[1:-1]),
        label="Model", color="black")
ax.plot(x2_arr[1:-1], trans_beta_Grec_arr, ":",
        label="TM", color="darkorange")
# ax.plot(n2_arr[1:-1], simu_beta_Grec_arr, "--",
#         label="RW", color="blue", marker="o", markersize=5.0, markevery=15)
# ax.plot(
#     n2_arr[1:-1],
#     simu_beta_Grec_arr,
#     label="RW",
#     color="blue",
#     linestyle='--',
#     linewidth=1.6,
#     marker="o",
#     markersize=4.5,
#     markevery=15
# )
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\beta U(x)$")
format_axes(ax)
add_legend(ax)
add_panel_label(ax, "(h)")
fig.savefig("reguera_reconst_high.pdf", bbox_inches="tight")
plt.show()
plt.close(fig)
