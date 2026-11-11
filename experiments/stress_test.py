import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..'
        )
    )
)

import numpy as np
import matplotlib.pyplot as plt

from src.mpc_controller import mpc_step
from src.utils import set_seed

# ======================================================
# REPRODUCIBILITY
# ======================================================

set_seed(0)

# ======================================================
# PARAMETERS
# ======================================================

n = 30
m = 3
T = 200

A = np.eye(n) * 1.002

B = np.random.randn(n, m) * 0.05

G = np.eye(n)

Q = np.eye(n)
R = np.eye(m)

noise_std = 0.2
sigma = 1.0 - noise_std

u_bounds = (-5.0, 5.0)

threshold = 50.0

# ======================================================
# INITIAL STATES
# ======================================================

x_nc = np.ones(n) * 0.1
x_ht = np.ones(n) * 0.1
x_mpc = np.ones(n) * 0.1

# ======================================================
# STORAGE
# ======================================================

exp_nc = []
exp_ht = []
exp_mpc = []

# ======================================================
# SIMULATION
# ======================================================

for t in range(T):

    # persistent adversarial injection
    w = np.random.randn(n) * 0.04 + 0.02

    # --------------------------------------------------
    # NC
    # --------------------------------------------------

    u_nc = np.zeros(m)

    x_nc = (
        A @ x_nc
        + B @ u_nc
        + G @ w
    )

    exp_nc.append(
        np.sum(x_nc)
    )

    # --------------------------------------------------
    # HT
    # --------------------------------------------------

    if np.sum(x_ht) > threshold:
        u_ht = -0.1 * np.ones(m)
    else:
        u_ht = np.zeros(m)

    x_ht = (
        A @ x_ht
        + B @ u_ht
        + G @ w
    )

    exp_ht.append(
        np.sum(x_ht)
    )

    # --------------------------------------------------
    # RT-MPC
    # --------------------------------------------------

    measurement_noise = (
        np.random.randn(n)
        * (1.0 - sigma)
        * 0.2
    )

    x_hat = x_mpc + measurement_noise

    u_mpc = mpc_step(
        x_hat,
        A,
        B,
        Q,
        R,
        u_bounds,
        sigma
    )

    x_mpc = (
        A @ x_mpc
        + B @ u_mpc
        + G @ w
    )

    exp_mpc.append(
        np.sum(x_mpc)
    )

# ======================================================
# ARRAYS
# ======================================================

exp_nc = np.array(exp_nc)
exp_ht = np.array(exp_ht)
exp_mpc = np.array(exp_mpc)

# ======================================================
# PLOT
# ======================================================

plt.figure(figsize=(8, 5))

plt.plot(
    exp_nc,
    label="NC (No Control)",
    linewidth=2
)

plt.plot(
    exp_ht,
    label="HT (Heuristic Threshold)",
    linestyle="--"
)

plt.plot(
    exp_mpc,
    label="RT-MPC (Proposed)",
    linewidth=2
)

plt.xlabel("Time Step")
plt.ylabel("Exposure Level")

plt.title(
    "Exposure Dynamics Under Persistent Adversarial Pressure"
)

# inset

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

ax = plt.gca()

axins = inset_axes(
    ax,
    width="40%",
    height="30%",
    loc="lower right"
)

axins.plot(exp_nc[:50])
axins.plot(exp_ht[:50])
axins.plot(exp_mpc[:50])

axins.set_title(
    "Early-Phase Behavior"
)

axins.grid(True)

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "Figure_Adversarial_StressTest.png",
    dpi=300,
    bbox_inches="tight"
)

print(
    "✅ Figure_Adversarial_StressTest.png generated"
)