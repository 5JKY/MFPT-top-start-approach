import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ============================================================
# JCP publication style
# ============================================================
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


# ============================================================
# Model potential
# ============================================================
def double_gaussian_potential(
    x,
    A1,
    A2,
    mu1=-1.0,
    sigma1=0.5,
    mu2=1.0,
    sigma2=0.6
):
    V1 = A1 * np.exp(
        -((x - mu1)**2) / (2 * sigma1**2)
    )

    V2 = A2 * np.exp(
        -((x - mu2)**2) / (2 * sigma2**2)
    )

    return -(V1 + V2)


# ============================================================
# x range
# ============================================================
x = np.linspace(-1.1, 1.1, 500)


# ============================================================
# Three model potentials
# ============================================================
U_low = double_gaussian_potential(
    x,
    A1=3,
    A2=4
)

U_medium = double_gaussian_potential(
    x,
    A1=12,
    A2=10
)

U_high = double_gaussian_potential(
    x,
    A1=30,
    A2=25
)


# ============================================================
# Plot
# ============================================================
fig, ax = plt.subplots(
    figsize=(3.3, 2.75),
    constrained_layout=True
)

# Low barrier
ax.plot(
    x,
    U_low,
    "-",
    label="Low barrier"
)

# Medium barrier
ax.plot(
    x,
    U_medium,
    "--",
    label="Medium barrier"
)

# High barrier
ax.plot(
    x,
    U_high,
    "-.",
    label="High barrier"
)


# ============================================================
# Axes
# ============================================================
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\beta U(x)$")

# Minor ticks
ax.minorticks_on()

# Major ticks
ax.tick_params(
    axis="both",
    which="major",
    direction="in",
    top=True,
    right=True,
    length=5,
    width=1.0
)

# Minor ticks
ax.tick_params(
    axis="both",
    which="minor",
    direction="in",
    top=True,
    right=True,
    length=2.5,
    width=0.8
)

# Spine thickness
for spine in ax.spines.values():
    spine.set_linewidth(1.0)


# ============================================================
# Legend
# ============================================================
legend = ax.legend(
    loc="best",
    frameon=True,
    framealpha=0.8,
    fancybox=True
)

legend.get_frame().set_linewidth(0.8)


# ============================================================
# Save
# ============================================================
fig.savefig(
    "model_double_well_potentials.pdf",
    bbox_inches="tight"
)

plt.show()