import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

from src.mpc_controller import mpc_step
from src.utils import set_seed

# PARAMETERS
n = 30
m = 3
T = 200

A = np.eye(n) * 0.98
B = np.random.randn(n, m) * 0.05
G = np.eye(n)

Q = np.eye(n)
R = np.eye(m)

u_bounds = (-0.5, 0.5)

threshold = 5.0

# Confidence range
sigma_values = np.linspace(0, 1, 20)

exp_nc = []
exp_ht = []
exp_mpc = []

# LOOP OVER CONFIDENCE LEVELS
for sigma in sigma_values:

    set_seed(0)

    x_nc = np.ones(n) * 0.1
    x_ht = np.ones(n) * 0.1
    x_mpc = np.ones(n) * 0.1

    # RUN SIMULATION
    for t in range(T):

        w = np.random.randn(n) * 0.02

        # --- NC ---
        u_nc = np.zeros(m)
        x_nc = A @ x_nc + B @ u_nc + G @ w

        # --- HT ---
        if np.sum(x_ht) > threshold:
            u_ht = -0.3 * np.ones(m)
        else:
            u_ht = np.zeros(m)

        x_ht = A @ x_ht + B @ u_ht + G @ w

        # --- RT-MPC ---
        u_mpc = mpc_step(x_mpc, A, B, Q, R, u_bounds, sigma)
        x_mpc = A @ x_mpc + B @ u_mpc + G @ w

    # Store steady-state (average last 20 steps)
    exp_nc.append(np.mean([np.sum(x_nc)]))
    exp_ht.append(np.mean([np.sum(x_ht)]))
    exp_mpc.append(np.mean([np.sum(x_mpc)]))


# CONVERT TO ARRAYS
exp_nc = np.array(exp_nc)
exp_ht = np.array(exp_ht)
exp_mpc = np.array(exp_mpc)

# PLOT
plt.figure(figsize=(8,5))

plt.plot(sigma_values, exp_nc, label="NC (No Control)", linestyle="--")
plt.plot(sigma_values, exp_ht, label="HT (Heuristic Threshold)", linestyle=":")
plt.plot(sigma_values, exp_mpc, label="RT-MPC (Proposed)", linewidth=2)

plt.xlabel("Detector confidence $\sigma$")
plt.ylabel("Steady-state exposure")

plt.title("Robustness to Detector Uncertainty")

plt.legend()
plt.grid(alpha=0.3)

plt.savefig("Figure_Uncertainty_Robustness.png", dpi=300)

print("✅ Figure 3 generated successfully")
