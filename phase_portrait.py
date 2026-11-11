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

# =====================================================
# REPRODUCIBILITY
# =====================================================

set_seed(0)

# =====================================================
# PARAMETERS
# =====================================================

n = 30
m = 3
T = 200

A = np.eye(n) * 0.995
B = np.random.randn(n, m) * 0.05
G = np.eye(n)

Q = np.eye(n)
R = np.eye(m)

sigma = 0.8

u_bounds = (-5.0, 5.0)

threshold = 50.0

# =====================================================
# INITIAL STATES
# =====================================================

x_nc = np.ones(n) * 3.0
x_ht = np.ones(n) * 3.0
x_mpc = np.ones(n) * 3.0

# =====================================================
# STORAGE
# =====================================================

exp_nc = []
u_nc_vals = []

exp_ht = []
u_ht_vals = []

exp_mpc = []
u_mpc_vals = []

# =====================================================
# SIMULATION LOOP
# =====================================================

for t in range(T):

    w = np.random.randn(n) * 0.03

    # -------------------------------------------------
    # NC (No Control)
    # -------------------------------------------------

    u_nc = np.zeros(m)

    x_nc = (
        A @ x_nc
        + G @ w
    )

    exp_nc.append(
        np.sum(x_nc)
    )

    u_nc_vals.append(
        0.0
    )

    # -------------------------------------------------
    # HT (Heuristic Threshold)
    # -------------------------------------------------

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

    # actual intervention magnitude
    u_ht_vals.append(
        np.linalg.norm(u_ht)
    )

    # -------------------------------------------------
    # RT-MPC (Proposed)
    # -------------------------------------------------

    u_mpc = mpc_step(
        x_mpc,
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

    # actual intervention magnitude
    u_mpc_vals.append(
        np.linalg.norm(u_mpc)
    )

# =====================================================
# ARRAYS
# =====================================================

exp_nc = np.array(exp_nc)
u_nc_vals = np.array(u_nc_vals)

exp_ht = np.array(exp_ht)
u_ht_vals = np.array(u_ht_vals)

exp_mpc = np.array(exp_mpc)
u_mpc_vals = np.array(u_mpc_vals)

# =====================================================
# PHASE PORTRAIT
# =====================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    exp_nc,
    u_nc_vals,
    alpha=0.6,
    label="NC (No Control)"
)

plt.scatter(
    exp_ht,
    u_ht_vals,
    alpha=0.6,
    label="HT (Heuristic Threshold)"
)

plt.scatter(
    exp_mpc,
    u_mpc_vals,
    alpha=0.6,
    label="RT-MPC (Proposed)"
)

plt.xlabel("Exposure Level")
plt.ylabel("Intervention Magnitude")

plt.title(
    "Phase Portrait: Exposure–Intervention Relationship"
)

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "Figure_Phase_Portrait.png",
    dpi=300,
    bbox_inches="tight"
)

print("✅ Figure_Phase_Portrait.png generated")