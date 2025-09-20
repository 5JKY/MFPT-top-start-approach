import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.integrate import quad
from scipy.interpolate import interp1d, PchipInterpolator, CubicSpline

# Define the double-well potential using two Gaussian functions
def double_gaussian_potential(x, A1=3, mu1=-1, sigma1=0.5, A2=4, mu2=1, sigma2=0.6):
# def double_gaussian_potential(x, A1=12, mu1=-1, sigma1=0.5, A2=10, mu2=1, sigma2=0.6):
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
plt.plot(x, beta_U(x), label='Potential Energy', color='black')
plt.title('Double-Well Potential Energy', fontsize=16, fontweight='bold')
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.xlabel('$x$', fontsize=20, fontweight='bold')
plt.ylabel(r'$\beta U(x)$', fontsize=20, fontweight='bold')
# plt.axhline(0, color='black',linewidth=0.5)
# plt.axvline(1.1, color='red',linewidth=1)
# plt.axvline(-1.1, color='red',linewidth=1)
# plt.scatter(-1.1, beta_U(-1.1), color='black')
# plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
# plt.legend()
plt.savefig("double_well_potential.png", dpi=600, bbox_inches='tight')  # high-quality PNG
plt.show()


a = -1.1    # location of reflecting boundary (will be used twice)
b2 = 1.1  # location of upper absorbing boundary
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

plt.plot(x2_arr, ira2_trans.steady_state, label="transfer matrix", color='darkorange')
Pst_n2 = np.load("data/reguera_Pst_n2.npy")
plt.plot(n2_arr, Pst_n2, '--', label="RW", color='blue', marker='o', markevery=15)
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.xlabel('$x$', fontsize=20, fontweight='bold')
plt.ylabel("$P_{st}(x)$", fontsize=20, fontweight='bold')
plt.title('Steady-State Distribution', fontsize=16, fontweight='bold')
# plt.legend(fontsize=12)
# Add legend here (after plotting, before saving/showing)
legend = plt.legend(fontsize=13, loc='best', frameon=True)
# Optional: Bold legend text and thicken frame
for text in legend.get_texts():
    text.set_fontweight('bold')
    text.set_color('black')
plt.savefig("reguera_Pst.png", dpi=600, bbox_inches='tight')  # high-quality PNG
plt.show()


plt.plot(x2_arr, -np.log(ira2_trans.steady_state), label="transfer matrix", color='darkorange')
plt.plot(x2_arr, -np.log(Pst_n2), '--', label='RW', color='blue', marker='o', markevery=15)
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.xlabel('$x$', fontsize=20, fontweight='bold')
plt.ylabel("$-ln[P_{st}(x)]$", fontsize=20)
plt.title('-ln(Steady-State Distribution)', fontsize=16, fontweight='bold')
# plt.legend(fontsize=12)
# Add legend here (after plotting, before saving/showing)
legend = plt.legend(fontsize=13, loc='best', frameon=True)
# Optional: Bold legend text and thicken frame
for text in legend.get_texts():
    text.set_fontweight('bold')
    text.set_color('black')
plt.savefig("reguera_lnPst.png", dpi=600, bbox_inches='tight')  # high-quality PNG
plt.show()


plt.plot(x2_arr, delt_t*m2_bar[0], label="transfer matrix", color='darkorange')
mfpt2_simu_arr = np.load("data/reguera_mfpt_n2.npy")
plt.plot(n2_arr, mfpt2_simu_arr, '--', label="RW", color='blue', marker='o', markevery=15)
plt.xticks(fontsize=13, fontweight='bold')
plt.yticks(fontsize=13, fontweight='bold')
plt.xlabel('$x$', fontsize=20, fontweight='bold')
plt.ylabel(r"$\tau (x)$", fontsize=20, fontweight='bold')
plt.title('Mean First-Passage Time', fontsize=16, fontweight='bold')
# plt.legend(fontsize=12)
# Add legend here (after plotting, before saving/showing)
legend = plt.legend(fontsize=13, loc='best', frameon=True)
# Optional: Bold legend text and thicken frame
for text in legend.get_texts():
    text.set_fontweight('bold')
    text.set_color('black')
plt.savefig("reguera_MFPT.png", dpi=600, bbox_inches='tight')  # high-quality PNG
plt.show()

from free_energy_reconst import reconstruct_energy_ra
trans_beta_Grec2_arr2 = reconstruct_energy_ra(x2_arr, beta_U=beta_U, Pst_arr=ira2_trans.steady_state, mfpt_arr=m2_bar[0])
trans_beta_GrecM_arr2 = np.log((1-m2_bar[0]/m2_bar[0,-1])/ira2_trans.steady_state)
const_trans = beta_U(x2_arr)[2] - trans_beta_GrecM_arr2[2]
trans_beta_GrecM_arr2 += const_trans
simu_beta_Grec2_arr2 = np.load("data/reguera_reconst_n2.npy")
simu_beta_GrecM_arr2 = np.log((1-mfpt2_simu_arr/mfpt2_simu_arr[-1])/Pst_n2)
const_simu = beta_U(x2_arr)[2] - simu_beta_GrecM_arr2[2]
simu_beta_GrecM_arr2 += const_trans
plt.plot(x2_arr[1:-1], beta_U(x2_arr[1:-1]), label="original", color="black")
plt.plot(x2_arr[1:-1], trans_beta_Grec2_arr2, ':', label="TM-Reguera", color='darkorange')
plt.plot(x2_arr, trans_beta_GrecM_arr2, '-.', label="TM-simplified", color='red')
plt.plot(n2_arr[1:-1], simu_beta_Grec2_arr2, '--', label="RW-Reguera", color='blue', marker='o', markevery=15)
plt.plot(n2_arr, simu_beta_GrecM_arr2, '--', label="RW-simplified")
# Plot formatting
plt.xlabel('$x$', fontsize=20,  fontweight='bold')
plt.ylabel('$ \\beta U(x) $', fontsize=20, fontweight='bold')
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.title('Reguera Free Energy Reconstruction', fontsize=16, fontweight='bold')
# plt.legend(fontsize=12)
# Add legend here (after plotting, before saving/showing)
legend = plt.legend(fontsize=13, loc='best', frameon=True)
# Optional: Bold legend text and thicken frame
for text in legend.get_texts():
    text.set_fontweight('bold')
    text.set_color('black')
plt.savefig("reguera_reconst.png", dpi=600, bbox_inches='tight')  # high-quality PNG
plt.show()
