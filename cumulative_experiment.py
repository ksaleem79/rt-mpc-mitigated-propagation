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
import csv

from src.mpc_controller import mpc_step
from src.utils import set_seed

# ==========================================================
# PARAMETERS
# ==========================================================

set_seed(0)

n = 30
m = 3
T = 60

A = np.eye(n) * 1.002
B = np.random.randn(n, m) * 0.05
G = np.eye(n)

Q = np.eye(n)
R = np.eye(m)

sigma = 0.8

u_bounds = (-5.0, 5.0)

# reviewer-consistent HT baseline
threshold = 50.0
ht_intervention = -0.1

# ==========================================================
# INITIAL STATES
# ==========================================================

x_nc = np.ones(n) * 0.1
x_ht = np.ones(n) * 0.1
x_mpc = np.ones(n) * 0.1

cum_nc = []
cum_ht = []
cum_mpc = []

total_nc = 0.0
total_ht = 0.0
total_mpc = 0.0

# ==========================================================
# SIMULATION
# ==========================================================

for t in range(T):

    # adversarial disturbance
    w = np.random.randn(n) * 0.04 + 0.02

    # ------------------------------------------------------
    # NC
    # ------------------------------------------------------

    u_nc = np.zeros(m)

    x_nc = A @ x_nc + B @ u_nc + G @ w

    total_nc += np.sum(x_nc)

    cum_nc.append(total_nc)

    # ------------------------------------------------------
    # HT
    # ------------------------------------------------------

    if np.sum(x_ht) > threshold:
        u_ht = ht_intervention * np.ones(m)
    else:
        u_ht = np.zeros(m)

    x_ht = A @ x_ht + B @ u_ht + G @ w

    total_ht += np.sum(x_ht)

    cum_ht.append(total_ht)

    # ------------------------------------------------------
    # RT-MPC
    # ------------------------------------------------------

    u_mpc = mpc_step(
        x_mpc,
        A,
        B,
        Q,
        R,
        u_bounds,
        sigma
    )

    x_mpc = A @ x_mpc + B @ u_mpc + G @ w

    total_mpc += np.sum(x_mpc)

    cum_mpc.append(total_mpc)

# ==========================================================
# SAVE RESULTS
# ==========================================================

os.makedirs("results", exist_ok=True)

np.save(
    "results/exposure_runs.npy",
    np.array([
        cum_nc,
        cum_ht,
        cum_mpc
    ])
)

with open(
    "results/cumulative_exposure.csv",
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "time",
            "NC",
            "HT",
            "RT_MPC"
        ]
    )

    for t in range(T):

        writer.writerow(
            [
                t,
                cum_nc[t],
                cum_ht[t],
                cum_mpc[t]
            ]
        )

print("✅ Results saved to results/")

# ==========================================================
# PLOT
# ==========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    cum_nc,
    label="NC (No Control)",
    linewidth=2,
    linestyle="--"
)

plt.plot(
    cum_ht,
    label="HT (Heuristic Threshold)",
    linewidth=2,
    linestyle=":"
)

plt.plot(
    cum_mpc,
    label="RT-MPC (Proposed)",
    linewidth=2
)

plt.xlabel("Time Step")
plt.ylabel("Cumulative Exposure")

plt.title(
    "Cumulative Exposure under Persistent Adversarial Pressure"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "Figure_Cumulative_Exposure.png",
    dpi=300,
    bbox_inches="tight"
)

print("✅ Figure_Cumulative_Exposure.png generated")
