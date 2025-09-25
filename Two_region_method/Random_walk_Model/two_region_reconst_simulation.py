# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.integrate import quad
from scipy.interpolate import interp1d, PchipInterpolator, CubicSpline

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
x = np.linspace(-1.5, 1.5, 400)

# Plot the potential
plt.figure(figsize=(8, 6))
plt.plot(x, beta_U(x), label='Potential Energy')
plt.title('Double-Well Potential with Two Gaussian Functions')
plt.xlabel('$x$')
plt.ylabel(r'$\beta U(x)$')
# plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(1.1, color='red',linewidth=1)
plt.axvline(-1.1, color='red',linewidth=1)
plt.scatter(-0.1, beta_U(-0.1), color='black')
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.legend()
plt.savefig('graphs/two_region_double_well_potential.png', dpi=300)
plt.close()

# %%
a = -0.1    # location of reflecting boundary (will be used twice)
b1 = -1.1  # location of lower absorbing boundary
b2 = 1.1  # location of upper absorbing boundary
h = 0.01
N1 = int((a-b1)/h+1)
N2 = int((b2-a)/h+1)
x1_arr = np.linspace(b1, a, N1)
x2_arr = np.linspace(a, b2, N2)

# %%

from mfpt_Pst_RW_simulate import simulate_AbRe, simulate_ReAb

num_particles = 2000
init_position = a
hx = h
ht = hx**2/(2*D0)
n1_arr = np.arange(b1, a+h/2, h)
n1_arr = np.round(n1_arr, decimals=5)
count_n1, ti_n1 = simulate_AbRe(init_point=init_position, num_particles=num_particles, beta_U=beta_U, n_arr=n1_arr, a=a, b=b1, hx=hx)

n2_arr = np.arange(a, b2+h/2, h)
n2_arr = np.round(n2_arr, decimals=5)
count_n2, ti_n2 = simulate_ReAb(init_point=init_position, num_particles=num_particles, beta_U=beta_U, n_arr=n2_arr, a=a, b=b2, hx=hx)

# %%
count_n1[0] = 0 
count_n2[-1] = 0 
Pst_n1 = count_n1/(h*np.sum(count_n1))
Pst_n2 = count_n2/(h*np.sum(count_n2))
np.save('data/two_region_Pst_n1', Pst_n1)
np.save('data/two_region_Pst_n2', Pst_n2)
plt.plot(n1_arr, Pst_n1, label="Metro walkers (AbRe)")
plt.plot(n2_arr, Pst_n2, label="Metro walkers (ReAb)")
from transfer_matrix_reptile import TransferMatrix_InReAb, TransferMatrix_AbReIn
ari1_trans = TransferMatrix_AbReIn(h, x1_arr, beta_U, 0)
ira2_trans = TransferMatrix_InReAb(h, x2_arr, beta_U, 0)

ari1_trans.steady_state[0] = 0
ira2_trans.steady_state[-1] = 0
ari1_trans.steady_state = ari1_trans.steady_state/(h*np.sum(ari1_trans.steady_state))
ira2_trans.steady_state = ira2_trans.steady_state/(h*np.sum(ira2_trans.steady_state))
plt.plot(x1_arr, ari1_trans.steady_state, '--', label="ARI-Pst")
plt.plot(x2_arr, ira2_trans.steady_state, '--', label="IRA-Pst")


plt.xlabel('x')
plt.ylabel("$P_{st}(x)$")
plt.legend()
plt.grid()
plt.savefig('graphs/two_region_Pst_simu.png', dpi=300)
plt.close()


# %%
plt.plot(x1_arr, -np.log(Pst_n1), label='Metro walkers (AbRe)')
plt.plot(x2_arr, -np.log(Pst_n2), label='Metro walkers (ReAb)')
plt.plot(x1_arr, -np.log(ari1_trans.steady_state),'--', label="ARI-Pst")
plt.plot(x2_arr, -np.log(ira2_trans.steady_state), '--', label="IRA-Pst")
plt.xlabel('x')
plt.ylabel("$-ln[Pst(x)]$")
plt.grid()
plt.legend()
plt.savefig('graphs/two_region_Pst_simu_ln.png', dpi=300)
plt.close()

# %%
from mfpt_matrix_calc import mfpt_matrix

m1_bar = mfpt_matrix(ari1_trans)
m2_bar = mfpt_matrix(ira2_trans)
delt_t = h**2/(2*D0)
plt.plot(x1_arr, delt_t*m1_bar[-1], '--', label="ARI-MFPT")
plt.plot(x2_arr, delt_t*m2_bar[0], '--', label="IRA-MFPT")

mfpt1_simu_arr = ht*np.mean(ti_n1, axis=0)
mfpt2_simu_arr = ht*np.mean(ti_n2, axis=0)
mfpt1_simu_arr[-1] = 0
mfpt2_simu_arr[0] = 0
np.save('data/two_region_mfpt_n1', mfpt1_simu_arr)
np.save('data/two_region_mfpt_n2', mfpt2_simu_arr)

plt.plot(n1_arr, mfpt1_simu_arr, label="Metro walkers (AbRe)")
plt.plot(n2_arr, mfpt2_simu_arr, label="Metro walkers (ReAb)")
plt.xlabel('x')
plt.ylabel(r"$\tau_{MFPT} (x)$")
plt.title('MFPT vs. Position of Absorbing Boundary')
plt.legend()
plt.grid()
plt.savefig('graphs/two_region_MFPT_simu.png', dpi=300)
plt.close()

# %% [markdown]
# #### Steady-State Flux and Probability Distribution Function - Numerically Nest integrate the expression (referred to be exact) 

# %%
# Define the inner function to integrate as a function of y
def inner_integrand(y):
    return 1.0/D(y)*np.exp(beta_U(y))

# Define the inner integral as a function of x
def inner_integral(x):
    y_lower = b1
    y_upper = x
    result, error = quad(inner_integrand, y_lower, y_upper)
    return result

# Define the outer integral
x_lower = b1
x_upper = a

# Define the outer function to integrate (also as a function of x)
def outer_integrand(x):
    return -np.exp(-beta_U(x))*inner_integral(x)

# Perform the outer integration
invert_st1_flux, error = quad(outer_integrand, x_lower, x_upper)
st1_flux = 1.0/invert_st1_flux

# Define the inner function to integrate as a function of y
def inner_integrand(y):
    return 1.0/D(y)*np.exp(beta_U(y))

# Define the inner integral as a function of x
def inner_integral(x):
    y_lower = x
    y_upper = b2
    result, error = quad(inner_integrand, y_lower, y_upper)
    return result

# Define the outer integral
x_lower = a
x_upper = b2

# Define the outer function to integrate (also as a function of x)
def outer_integrand(x):
    return np.exp(-beta_U(x))*inner_integral(x)

invert_st2_flux, error = quad(outer_integrand, x_lower, x_upper)
st2_flux = 1.0/invert_st2_flux
print(st1_flux, st2_flux)

# %%
def st1_P_func(x):
    def integrand(y):
        return 1.0/D(y)*np.exp(beta_U(y))
    # Perform the integration
    y_lower = b1
    y_upper = x
    result, error = quad(integrand, y_lower, y_upper)
    result *= -st1_flux*np.exp(-beta_U(x))
    return result

st1_P_arr = np.zeros(x1_arr.size)
for i in np.arange(x1_arr.size):
    st1_P_arr[i] = st1_P_func(x1_arr[i])

def st2_P_func(x):
    def integrand(y):
        return 1.0/D(y)*np.exp(beta_U(y))
    # Perform the integration
    y_lower = x
    y_upper = b2
    result, error = quad(integrand, y_lower, y_upper)
    result *= st2_flux*np.exp(-beta_U(x))
    return result

st2_P_arr = np.zeros(x2_arr.size)
for i in np.arange(x2_arr.size):
    st2_P_arr[i] = st2_P_func(x2_arr[i])

# %%
plt.plot(x1_arr, ari1_trans.steady_state, label="ARI-Pst")
plt.plot(x2_arr, ira2_trans.steady_state, label="IRA-Pst")
plt.plot(x1_arr, st1_P_arr, '--', label='num_integrate1')
plt.plot(x2_arr, st2_P_arr, '--', label='num_integrate2')
plt.title('Steady-state Probability Distribution')
plt.xlabel('x')
plt.ylabel("$P_{st}(x)$")
plt.grid()
plt.legend()
plt.savefig('graphs/two_region_Pst_exact.png', dpi=300)
plt.close()

# %%
plt.plot(x1_arr, -np.log(ari1_trans.steady_state), label="ARI-Pst")
plt.plot(x2_arr, -np.log(ira2_trans.steady_state), label="IRA-Pst")
plt.plot(x1_arr, -np.log(st1_P_arr), '--', label='num_integrate1')
plt.plot(x2_arr, -np.log(st2_P_arr), '--', label='num_integrate2')
plt.xlabel('x')
plt.ylabel("$-ln[Pst(x)]$")
plt.grid()
plt.legend()
plt.savefig('graphs/two_region_Pst_exact_ln.png', dpi=300)
plt.close()
# %% [markdown]
# #### Numerically Nest integrate for MFPT - referred to be exact

# %%
x0 = a   # Regura's method, overlap the starting point and reflecting boundary
# Define the inner function to integrate as a function of z
def inner_integrand(z):
    return np.exp(-beta_U(z))

# Define the inner integral as a function of y
def inner_integral(y):
    z_lower = a
    z_upper = y
    result, error = quad(inner_integrand, z_lower, z_upper)
    return result

# Define the outer integral2
y_lower = x0
y_upper = b2
# Define the outer function to integrate (also as a function of y)
def outer_integrand(y):
    return np.exp(beta_U(y))*inner_integral(y)/D(y)
mfpt2_arr = np.zeros(x2_arr.size)
for i in np.arange(x2_arr.size):
    mfpt2_arr[i], _ = quad(outer_integrand, y_lower, x2_arr[i])

# Define the outer integral1
y_lower = x0
y_upper = b1
# Define the outer function to integrate (also as a function of y)
def outer_integrand(y):
    return np.exp(beta_U(y))*inner_integral(y)/D(y)
mfpt1_arr = np.zeros(x1_arr.size)
for i in np.arange(x1_arr.size):
    mfpt1_arr[i], _ = quad(outer_integrand, y_lower, x1_arr[i])

plt.xlabel('x')
plt.ylabel(r"$\tau_{MFPT} (x)$")
plt.plot(x1_arr, delt_t*m1_bar[-1], label="ARI-MFPT")
plt.plot(x2_arr, delt_t*m2_bar[0], label="IRA-MFPT")
plt.plot(x1_arr, mfpt1_arr,':', label='num_integrate1')
plt.plot(x2_arr, mfpt2_arr, ':',  label='num_integrate2')
plt.title('MFPT vs. Position of Absorbing Boundary')
plt.legend()
plt.grid()
plt.savefig('graphs/two_region_MFPT_exact.png', dpi=300)
plt.close()

# %% [markdown]
# #### Test Reconstruction Function with Exact Data

# %%
from free_energy_reconst import reconstruct_energy_ar, reconstruct_energy_ra
exact_beta_Grec2_arr1 = reconstruct_energy_ar(x1_arr, beta_U=beta_U, Pst_arr=st1_P_arr, mfpt_arr=mfpt1_arr)
exact_beta_Grec2_arr2 = reconstruct_energy_ra(x2_arr, beta_U=beta_U, Pst_arr=st2_P_arr, mfpt_arr=mfpt2_arr)

plt.plot(x1_arr[1:-1], exact_beta_Grec2_arr1, label="exact reconst")
plt.plot(x2_arr[1:-1], exact_beta_Grec2_arr2, label="exact reconst")

plt.plot(x1_arr[1:-1], beta_U(x1_arr[1:-1]), ':', label="exact")
plt.plot(x2_arr[1:-1], beta_U(x2_arr[1:-1]), ':', label="exact")

# Plot formatting
plt.xlabel('n')
plt.ylabel('$ \\beta \Delta G(n) $')
plt.title('free energy reconstruction')
plt.legend()
plt.grid()
plt.savefig('graphs/two_region_reconst_exact.png', dpi=300)
plt.close()

# %% [markdown]
# ### Reconstruction Using data [mfpt matrix (mat_Mbar[i]), Pst (ria_trans.steady_state)] calculated by Transfer Matrix

# %%
trans_beta_Grec2_arr1 = reconstruct_energy_ar(x1_arr, beta_U=beta_U, Pst_arr=ari1_trans.steady_state, mfpt_arr=m1_bar[-1])
trans_beta_Grec2_arr2 = reconstruct_energy_ra(x2_arr, beta_U=beta_U, Pst_arr=ira2_trans.steady_state, mfpt_arr=m2_bar[0])

plt.plot(x1_arr[1:-1], trans_beta_Grec2_arr1, label="transfer matrix reconst")
plt.plot(x2_arr[1:-1], trans_beta_Grec2_arr2, label="transfer matrix reconst")
plt.plot(x1_arr[1:-1], beta_U(x1_arr[1:-1]), ':', label="exact")
plt.plot(x2_arr[1:-1], beta_U(x2_arr[1:-1]), ':', label="exact")

# Plot formatting
plt.xlabel('n')
plt.ylabel('$ \\beta \Delta G(n) $')
plt.title('free energy reconstruction')
plt.legend()
plt.grid()
plt.savefig('graphs/two_region_reconst_transfer.png', dpi=300)
plt.close()

# %% [markdown]
# ### Reconstruction Using data [mfpt (mfpt1_simu_arr), Pst (Pst_n)] extracted from simulation

# %%
simu_beta_Grec2_arr1 = reconstruct_energy_ar(n1_arr, beta_U=beta_U, Pst_arr=Pst_n1, mfpt_arr=mfpt1_simu_arr)
simu_beta_Grec2_arr2 = reconstruct_energy_ra(n2_arr, beta_U=beta_U, Pst_arr=Pst_n2, mfpt_arr=mfpt2_simu_arr)
np.save('data/two_region_reconst_n1', simu_beta_Grec2_arr1)
np.save('data/two_region_reconst_n2', simu_beta_Grec2_arr2)

plt.plot(n1_arr[1:-1], simu_beta_Grec2_arr1, label="AbRe_simu reconst")
plt.plot(n2_arr[1:-1], simu_beta_Grec2_arr2, label="ReAb_simu reconst")

plt.plot(x1_arr[1:-1], beta_U(x1_arr[1:-1]), ':', label="exact")
plt.plot(x2_arr[1:-1], beta_U(x2_arr[1:-1]), ':', label="exact")

# Plot formatting
plt.xlabel('n')
plt.ylabel('$ \\beta \Delta G(n) $')
plt.title('free energy reconstruction')
plt.legend()
plt.grid()
plt.savefig('graphs/two_reconst_simu.png', dpi=300)
plt.close()
# %%



