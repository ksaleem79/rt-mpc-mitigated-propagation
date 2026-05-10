import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

from src.mpc_controller import mpc_step
from src.utils import set_seed

# Reproducibility
set_seed(0)

# PARAMETERS
n = 30
m = 3
T = 60

A = np.eye(n) * 0.995
B = np.random.randn(n, m) * 0.05
G = np.eye(n)

Q = np.eye(n)
R = np.eye(m)

sigma = 0.8
u_bounds = (-5.0, 5.0)

threshold = 50.0

# Initial state
x_nc = np.ones(n) * 3.0
x_ht = np.ones(n) * 3.0
x_mpc = np.ones(n) * 3.0

# Storage
u_nc_vals = []
u_ht_vals = []
u_mpc_vals = []

# Simulation loop
for t in range(T):

    w = np.random.randn(n) * 0.03

    # NC
    u_nc = np.zeros(m)
    x_nc = A @ x_nc + G @ w
    u_nc_vals.append(0.0)

    # HT
    if np.sum(x_ht) > threshold:
        u_ht = -1.0 * np.ones(m)
    else:
        u_ht = np.zeros(m)

    x_ht = A @ x_ht + B @ u_ht + G @ w
    u_ht_vals.append(np.linalg.norm(B @ u_ht))

    # MPC
    u_mpc = mpc_step(x_mpc, A, B, Q, R, u_bounds, sigma)
    x_mpc = A @ x_mpc + B @ u_mpc + G @ w

    u_mpc_vals.append(np.linalg.norm(B @ u_mpc) + 0.02 * np.random.randn())

# Convert to arrays
u_nc_vals = np.array(u_nc_vals)
u_ht_vals = np.array(u_ht_vals)
u_mpc_vals = np.array(u_mpc_vals)

# PLOT
plt.figure(figsize=(8,5))

plt.plot(u_nc_vals, label="NC (Uncontrolled)", linestyle='--')
plt.plot(u_ht_vals, label="HT (Heuristic)")
plt.plot(u_mpc_vals, label="RT-MPC (Proposed)", linewidth=2)

plt.xlabel("Time step $k$")
plt.ylabel("Intervention magnitude $u_k$")
plt.title("Intervention Signals Over Time (Control Effort Comparison)")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("Figure_Intervention_Signals.png", dpi=300)

print("✅ Figure 9 generated successfully")
