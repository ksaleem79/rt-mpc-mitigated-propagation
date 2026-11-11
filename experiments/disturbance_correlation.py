import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.mpc_controller import mpc_step
from src.utils import set_seed

# =====================================================
# REPRODUCIBILITY
# =====================================================

set_seed(0)

# =====================================================
# SETTINGS
# =====================================================

n = 30
m = 3

samples = 60

# Monte Carlo repetitions per disturbance level
repetitions = 10

# propagation horizon
T = 30

A = np.eye(n) * 1.002
B = np.random.randn(n, m) * 0.05
G = np.eye(n)

Q = np.eye(n)
R = np.eye(m)

u_bounds = (-5.0, 5.0)

sigma = 0.8

threshold = 50.0

# =====================================================
# STORAGE
# =====================================================

disturbances = []

resp_nc = []
resp_ht = []
resp_mpc = []

# =====================================================
# MAIN LOOP
# =====================================================

for disturbance_level in np.linspace(1.5, 5.5, samples):

    disturbances.append(disturbance_level)

    nc_runs = []
    ht_runs = []
    mpc_runs = []

    # -----------------------------------------------
    # Monte Carlo repetitions
    # -----------------------------------------------

    for r in range(repetitions):

        set_seed(r)

        x_nc = np.ones(n) * 0.1
        x_ht = np.ones(n) * 0.1
        x_mpc = np.ones(n) * 0.1

        # propagate for T steps
        for t in range(T):

            w = (
                np.random.randn(n) * 0.05
                + disturbance_level * 0.02
            )

            # ---------------------------------------
            # NC
            # ---------------------------------------

            u_nc = np.zeros(m)

            x_nc = (
                A @ x_nc
                + G @ w
            )

            # ---------------------------------------
            # HT
            # ---------------------------------------

            if np.sum(x_ht) > threshold:
                u_ht = -0.1 * np.ones(m)
            else:
                u_ht = np.zeros(m)

            x_ht = (
                A @ x_ht
                + B @ u_ht
                + G @ w
            )

            # ---------------------------------------
            # RT-MPC
            # ---------------------------------------

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

        nc_runs.append(np.mean(x_nc))
        ht_runs.append(np.mean(x_ht))
        mpc_runs.append(np.mean(x_mpc))

    # average over realizations

    resp_nc.append(np.mean(nc_runs))
    resp_ht.append(np.mean(ht_runs))
    resp_mpc.append(np.mean(mpc_runs))

# =====================================================
# SAVE DATA
# =====================================================

df = pd.DataFrame(
    {
        "disturbance": disturbances,
        "NC": resp_nc,
        "HT": resp_ht,
        "RT_MPC": resp_mpc
    }
)

os.makedirs(
    "results",
    exist_ok=True
)

df.to_csv(
    "results/disturbance_response.csv",
    index=False
)

print(
    "✅ Data saved to results/disturbance_response.csv"
)

# =====================================================
# PLOT FIGURE
# =====================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    disturbances,
    resp_nc,
    alpha=0.6,
    label="NC (No Control)"
)

plt.scatter(
    disturbances,
    resp_ht,
    alpha=0.6,
    label="HT (Heuristic Threshold)"
)

plt.scatter(
    disturbances,
    resp_mpc,
    alpha=0.6,
    label="RT-MPC (Proposed)"
)

# trend lines

for y, color in zip(
    [resp_nc, resp_ht, resp_mpc],
    ["tab:blue", "tab:orange", "tab:green"]
):
    coeff = np.polyfit(
        disturbances,
        y,
        1
    )

    trend = np.poly1d(coeff)

    plt.plot(
        disturbances,
        trend(disturbances),
        color=color,
        linewidth=2
    )

plt.xlabel(
    "Adversarial Disturbance Magnitude"
)

plt.ylabel(
    "Steady-State Exposure"
)

plt.title(
    "Disturbance–Response Relationship Across Mitigation Strategies"
)

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "Figure_Disturbance_Response_Correlation.png",
    dpi=300,
    bbox_inches="tight"
)

print(
    "✅ Figure_Disturbance_Response_Correlation.png generated"
)
