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

# ==========================================================
# PARAMETERS
# ==========================================================

n = 30
m = 3
T = 200

set_seed(0)

A = np.eye(n) * 0.98
B = np.random.randn(n, m) * 0.05
G = np.eye(n)

Q = np.eye(n)
R = np.eye(m)

u_bounds = (-0.5, 0.5)
threshold = 5.0

# detector confidence range
sigma_values = np.linspace(0, 1, 20)

exp_nc = []
exp_ht = []
exp_mpc = []

# ==========================================================
# LOOP OVER CONFIDENCE LEVELS
# ==========================================================

for sigma in sigma_values:

    set_seed(0)

    x_nc = np.ones(n) * 0.1
    x_ht = np.ones(n) * 0.1
    x_mpc = np.ones(n) * 0.1

    # store histories for steady-state averaging
    hist_nc = []
    hist_ht = []
    hist_mpc = []

    # ======================================================
    # RUN SIMULATION
    # ======================================================

    for t in range(T):

        # common disturbance
        w = np.random.randn(n) * 0.02

        # --------------------------------------------------
        # NC
        # --------------------------------------------------

        u_nc = np.zeros(m)

        x_nc = A @ x_nc + B @ u_nc + G @ w

        hist_nc.append(np.sum(x_nc))

        # --------------------------------------------------
        # HT
        # --------------------------------------------------

        if np.sum(x_ht) > threshold:
            u_ht = -0.3 * np.ones(m)
        else:
            u_ht = np.zeros(m)

        x_ht = A @ x_ht + B @ u_ht + G @ w

        hist_ht.append(np.sum(x_ht))

        # --------------------------------------------------
        # RT-MPC
        # --------------------------------------------------

        # lower confidence -> larger observation error
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

        x_mpc = A @ x_mpc + B @ u_mpc + G @ w

        hist_mpc.append(np.sum(x_mpc))

    # ======================================================
    # STORE STEADY-STATE VALUES
    # ======================================================

    exp_nc.append(
        np.mean(hist_nc[-20:])
    )

    exp_ht.append(
        np.mean(hist_ht[-20:])
    )

    exp_mpc.append(
        np.mean(hist_mpc[-20:])
    )

# ==========================================================
# CONVERT
# ==========================================================

exp_nc = np.array(exp_nc)
exp_ht = np.array(exp_ht)
exp_mpc = np.array(exp_mpc)

# ==========================================================
# PLOT
# ==========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    sigma_values,
    exp_nc,
    linestyle="--",
    label="NC (No Control)"
)

plt.plot(
    sigma_values,
    exp_ht,
    linestyle=":",
    label="HT (Heuristic Threshold)"
)

plt.plot(
    sigma_values,
    exp_mpc,
    linewidth=2,
    label="RT-MPC (Proposed)"
)

plt.xlabel("Detector confidence $\\sigma$")
plt.ylabel("Steady-state exposure")
plt.title("Robustness to Detector Uncertainty")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "Figure_Uncertainty_Robustness.png",
    dpi=300
)

print("✅ Figure 3 generated successfully")