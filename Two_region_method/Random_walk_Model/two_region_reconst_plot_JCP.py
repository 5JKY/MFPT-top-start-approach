import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.sparse.linalg import spsolve
from scipy.integrate import quad
from scipy.interpolate import interp1d, PchipInterpolator, CubicSpline

# JCP publication style
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "lines.linewidth": 1.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def finish_axes():
    ax = plt.gca()
    ax.tick_params(axis="both", which="both", direction="in",
                   top=True, right=True, length=5, width=1)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

def add_legend(loc="best"):
    legend = plt.legend(loc=loc, frameon=True, framealpha=0.8,
                        fancybox=True)
    legend.get_frame().set_linewidth(0.8)
    return legend


def add_panel_label(label):
    ax = plt.gca()
    ax.text(
        0.03,
        0.97,
        label,
        transform=ax.transAxes,
        fontsize=10,
        ha="left",
        va="top"
    )

# Define the double-well potential using two Gaussian functions
# def double_gaussian_potential(x, A1=3, mu1=-1, sigma1=0.5, A2=4, mu2=1, sigma2=0.6):
def double_gaussian_potential(x, A1=12, mu1=-1, sigma1=0.5, A2=10, mu2=1, sigma2=0.6):
# def double_gaussian_potential(x, A1=30, mu1=-1, sigma1=0.5, A2=25, mu2=1, sigma2=0.6):
    V1 = A1 * np.exp(-((x - mu1)**2) / (2 * sigma1**2))
    V2 = A2 * np.exp(-((x - mu2)**2) / (2 * sigma2**2))
    return -(V1 + V2)
beta_U = double_gaussian_potential

D0 = 0.01
def D(x):
    # return D0*x**(2/3)
    return D0*x**0
x = np.linspace(-1.1, 1.1, 400)

# plt.figure(figsize=(8, 6))
plt.figure(figsize=(3.3, 2.75))
plt.plot(x, beta_U(x), color='black', linewidth=2.0)
# No title: describe the panel in the manuscript caption


plt.xlabel(r'$x$')
plt.ylabel(r'$\beta U(x)$')
# plt.axhline(0, color='black',linewidth=0.5)
# plt.axvline(1.1, color='red',linewidth=1)
# plt.axvline(-1.1, color='red',linewidth=1)
plt.scatter(-0.1, beta_U(-0.1), color='black', s=40, zorder=5)
finish_axes()
plt.tight_layout()
plt.savefig("two_region_double_well_potential.pdf", bbox_inches="tight")
plt.show()

a = -0.1    # location of reflecting boundary (will be used twice)
b1 = -1.1  # location of lower absorbing boundary
b2 = 1.1  # location of upper absorbing boundary
h = 0.01
N1 = int((a-b1)/h+1)
N2 = int((b2-a)/h+1)
x1_arr = np.linspace(b1, a, N1)
x2_arr = np.linspace(a, b2, N2)
num_particles = 2000
init_position = a
hx = h
ht = hx**2/(2*D0)
n1_arr = np.arange(b1, a+h/2, h)
n1_arr = np.round(n1_arr, decimals=5)
n2_arr = np.arange(a, b2+h/2, h)
n2_arr = np.round(n2_arr, decimals=5)
from transfer_matrix_reptile import TransferMatrix_InReAb, TransferMatrix_AbReIn
ari1_trans = TransferMatrix_AbReIn(h, x1_arr, beta_U, 0)
ira2_trans = TransferMatrix_InReAb(h, x2_arr, beta_U, 0)
ari1_trans.steady_state[0] = 0
ira2_trans.steady_state[-1] = 0
ari1_trans.steady_state = ari1_trans.steady_state/(h*np.sum(ari1_trans.steady_state))
ira2_trans.steady_state = ira2_trans.steady_state/(h*np.sum(ira2_trans.steady_state))
from mfpt_matrix_calc import mfpt_matrix
m1_bar = mfpt_matrix(ari1_trans)
m2_bar = mfpt_matrix(ira2_trans)
delt_t = h**2/(2*D0)
trans_mfpt_n1=delt_t*m1_bar[-1]
trans_mfpt_n2=delt_t*m2_bar[0]

plt.figure(figsize=(3.3, 2.75))
plt.plot(x1_arr, ari1_trans.steady_state, label="TM A", color='darkgreen')
plt.plot(x2_arr, ira2_trans.steady_state, label="TM B", color='darkorange')
two_Pst_n1 = np.load("data/two_region_Pst_n1.npy")
two_Pst_n2 = np.load("data/two_region_Pst_n2.npy")
plt.plot(n1_arr, two_Pst_n1, '--', label="RW A", color='red', marker='o', markersize=3.2, markevery=15)
plt.plot(n2_arr, two_Pst_n2, '--', label="RW B", color='blue', marker='o', markersize=3.2, markevery=15)


plt.xlabel(r'$x$')
plt.ylabel(r"$P_{\mathrm{st}}(x)$")
# plt.title('Steady-State Distribution', fontsize=16, fontweight='bold')
# plt.legend(fontsize=13)
# Add legend here (after plotting, before saving/showing)
finish_axes()
legend = add_legend(loc="upper right")
add_panel_label("(a)")
plt.tight_layout()
plt.savefig("two_Pst.pdf", bbox_inches="tight")  # high-quality PNG
plt.show()


plt.figure(figsize=(3.3, 2.75))
valid_tm_A = ari1_trans.steady_state > 0
valid_tm_B = ira2_trans.steady_state > 0
valid_rw_A = two_Pst_n1 > 0
valid_rw_B = two_Pst_n2 > 0
plt.plot(x1_arr[valid_tm_A], -np.log(ari1_trans.steady_state[valid_tm_A]), label="TM A", color='darkgreen')
plt.plot(x2_arr[valid_tm_B], -np.log(ira2_trans.steady_state[valid_tm_B]), label="TM B", color='darkorange')
plt.plot(n1_arr[valid_rw_A], -np.log(two_Pst_n1[valid_rw_A]), '--', label="RW A", color='red', marker='o', markersize=3.2, markevery=15)
plt.plot(n2_arr[valid_rw_B], -np.log(two_Pst_n2[valid_rw_B]), '--', label="RW B", color='blue', marker='o', markersize=3.2, markevery=15)


plt.xlabel(r'$x$')
plt.ylabel(r"$-\ln[P_{\mathrm{st}}(x)]$")
# plt.title('-ln(Steady-State Distribution)', fontsize=16, fontweight='bold')
# plt.legend(fontsize=12)
# Add legend here (after plotting, before saving/showing)
finish_axes()
legend = add_legend(loc="upper center")
add_panel_label("(b)")
plt.tight_layout()
plt.savefig("two_lnPst.pdf", bbox_inches="tight")  # high-quality PNG
plt.show()


plt.figure(figsize=(3.3, 2.75))
plt.plot(x1_arr, trans_mfpt_n1, '-', label="TM A", color='darkgreen')
plt.plot(x2_arr, trans_mfpt_n2, '-', label="TM B", color='darkorange')
two_mfpt_n1 = np.load("data/two_region_mfpt_n1.npy")
two_mfpt_n2 = np.load("data/two_region_mfpt_n2.npy")
plt.plot(n1_arr, two_mfpt_n1, '--', label="RW A", color='red', marker='o', markersize=3.2, markevery=15)
plt.plot(n2_arr, two_mfpt_n2, '--', label="RW B", color='blue', marker='o', markersize=3.2, markevery=15)
plt.xlabel(r'$x$')
plt.ylabel(r"$\tau(x)$")


# plt.title('Mean First-Passage Time', fontsize=16, fontweight='bold')
# plt.legend(fontsize=12)
# Add legend here (after plotting, before saving/showing)
finish_axes()
legend = add_legend(loc="upper center")
add_panel_label("(c)")
plt.tight_layout()
plt.savefig("two_MFPT.pdf", bbox_inches="tight")  # high-quality PNG
plt.show()


from free_energy_reconst import reconstruct_energy_ar, reconstruct_energy_ra
trans_reconst_n1 = reconstruct_energy_ar(x1_arr, beta_U=beta_U, Pst_arr=ari1_trans.steady_state, mfpt_arr=m1_bar[-1])
trans_reconst_n2 = reconstruct_energy_ra(x2_arr, beta_U=beta_U, Pst_arr=ira2_trans.steady_state, mfpt_arr=m2_bar[0])
two_reconst_n1 = np.load("data/two_region_reconst_n1.npy")
two_reconst_n2 = np.load("data/two_region_reconst_n2.npy")
plt.figure(figsize=(3.3, 2.75))
plt.plot(np.arange(b1,b2,h), beta_U(np.arange(b1,b2,h)), label="Original", color="black")
plt.plot(x1_arr[1:-1], trans_reconst_n1, ':', label="TM A", color='darkgreen', marker='^', markersize=3.2, markevery=20)
plt.plot(x2_arr[1:-1], trans_reconst_n2, ':', label="TM B", color='darkorange', marker='^', markersize=3.2, markevery=20)
plt.plot(n1_arr[1:-1], two_reconst_n1, '--', label="RW A", color='red', marker='o', markersize=3.2, markevery=15)
# plt.plot(n1_arr[1:-1], two_reconst_n1, label="walkers left (two)")
plt.plot(n2_arr[1:-1], two_reconst_n2, '--', label="RW B", color='blue', marker='o', markersize=3.2, markevery=15)
# plt.plot(n2_arr[1:-1], two_reconst_n2, label="walkers right (two)")
# plt.plot(x1_arr[1:-1], beta_U(x1_arr[1:-1]), ':', label="exact-left")
# plt.plot(x2_arr[1:-1], beta_U(x2_arr[1:-1]), ':', label="exact-right")
# Plot formatting
plt.xlabel(r'$x$')
plt.ylabel(r'$\beta U(x)$')


# plt.title('Two-Region Free Energy Reconstruction', fontsize=16, fontweight='bold')
# plt.legend(fontsize=12)
# Add legend here (after plotting, before saving/showing)
finish_axes()
legend = add_legend(loc="best")
add_panel_label("(d)")
plt.tight_layout()
plt.savefig("two_reconst.pdf", bbox_inches="tight")  # high-quality PNG
plt.show()