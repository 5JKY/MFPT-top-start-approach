# MFPT Top-Start Approach

This repository contains the random-walk simulations, transition-matrix calculations, numerical data,
analysis scripts, and plottings scripts used in our study of mean first-passage time (MFPT) free-energy reconstruction.

The work investigates three reconstruction approaches:

1. Conventional Reguera MFPT reconstruction
2. Two-region reconstruction
3. Top-start reconstruction

The methods are tested using one-dimensional double-well model potentials with
low, medium, and high free-energy barriers.

## Repository structure

The main calculations are organized into three directories:

```text
Reguera_method/Random_walk_Model/
Two_region_method/Random_walk_Model/
Top_start_method/Random_walk_Model/
```

Each directory contains the code and data associated with the corresponding
MFPT reconstruction procedure.

## Model system

The model free-energy profile is defined as

$$
\beta U(x) = -A_1 \exp\left[-\frac{(x-\mu_1)^2}{2\sigma_1^2}\right] -A_2 \exp\left[-\frac{(x-\mu_2)^2}{2\sigma_2^2}\right].
$$

The parameters are

$$
\mu_1=-1,\qquad
\mu_2=1,\qquad
\sigma_1=0.5,\qquad
\sigma_2=0.6.
$$

The amplitudes are

| Case | $A_1$ | $A_2$ | Approximate barrier height |
|---|---:|---:|---:|
| Low barrier | 3 | 4 | $2 \  k_{\mathrm B}T$ |
| Medium barrier | 12 | 10 | $8 \  k_{\mathrm B}T$ |
| High barrier | 30 | 25 | $20 \  k_{\mathrm B}T$ |

The random-walk simulations use

$$
\Delta x = 0.01,\qquad
D_0 = 0.01,
$$

corresponding to a time step

$$
\Delta t = \frac{(\Delta x)^2}{2D_0}=0.005.
$$

The simulations reported in the manuscript use 2000 independent random walkers.

## Reconstruction methods

### Reguera reconstruction

The conventional Reguera reconstruction uses a reflecting boundary at
$x=-1.1$ and an absorbing boundary at $x=1.1$.

The low- and medium-barrier cases are used to demonstrate the increasing
sampling difficulty associated with full barrier-crossing trajectories.

### Two-region reconstruction

For the two-region method, a reflecting boundary is placed at

$$
a=-0.1,
$$

with absorbing boundaries

$$
b_1=-1.1,\qquad b_2=1.1.
$$

Separate region-specific steady-state distributions and MFPTs are obtained for
regions A and B and used in the corresponding reconstruction equations.

### Top-start reconstruction


For the top-start method, each walker is initialized at

$$
a=-0.1,
$$ 

and evolves with recycling at the absorbing boundaries until both $b_1=-1.1$ and $b_2=1.1$ have been visited.

The resulting trajectories are decomposed to obtain the region-specific
steady-state distributions and MFPTs required for free-energy reconstruction.

## Transition-matrix calculations

Transition-matrix (TM) calculations are used as numerical benchmarks for the
random-walk (RW) simulations.

The TM calculations use the same spatial discretization and transition
probabilities as the RW simulations.

## Data and figures

The numerical data supporting the results of the manuscript, together with the
scripts used for analysis and figure generation, are available in the main
branch of this repository. Polished versions of the figures are available in
eight separate branches.
