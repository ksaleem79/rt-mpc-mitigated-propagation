import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

from src.utils import set_seed
from src.mpc_controller import mpc_step

set_seed(0)

# Parameters
n = 30
m = 3
T = 200

A = np.eye(n) * 0.98
B = np.random.randn(n, m) * 0.05
G = np.eye(n)

Q = np.eye(n)
R = np.eye(m)

noise_std = 0.2
u_bounds = (-0.5, 0.5)

# Initial state
x_nc = np.ones(n) * 0.1
x_ht = np.ones(n) * 0.1
x_mpc = np.ones(n) * 0.1

# Storage
exp_nc = []
exp_ht = []
exp_mpc = []

threshold = 5.0

for t in range(T):
    w = np.random.randn(n) * 0.02

    # ----- NC (no control) -----
    u_nc = np.zeros(m)
    x_nc = A @ x_nc + B @ u_nc + G @ w
    exp_nc.append(np.sum(x_nc))

    # ----- HT (heuristic threshold) -----
    total_ht = np.sum(x_ht)
    if total_ht > threshold:
        u_ht = -0.3 * np.ones(m)
    else:
        u_ht = np.zeros(m)

    x_ht = A @ x_ht + B @ u_ht + G @ w
    exp_ht.append(np.sum(x_ht))

    # ----- RT-MPC -----
    sigma = 1 - noise_std
    u_mpc = mpc_step(x_mpc, A, B, Q, R, u_bounds, sigma)

    x_mpc = A @ x_mpc + B @ u_mpc + G @ w
    exp_mpc.append(np.sum(x_mpc))

# Convert arrays
exp_nc = np.array(exp_nc)
exp_ht = np.array(exp_ht)
exp_mpc = np.array(exp_mpc)

# Plot
plt.figure()
plt.plot(exp_nc, label="NC (No Control)", linestyle="--")
plt.plot(exp_ht, label="HT (Heuristic)", linestyle=":")
plt.plot(exp_mpc, label="RT-MPC (Proposed)", linewidth=2)

plt.xlabel("Time")
plt.ylabel("Exposure")
plt.title("Exposure Trajectories under Adversarial Injection")

plt.legend()
plt.grid()

plt.savefig("Figure_Exposure_Trajectories.png", dpi=300)

print("✅ Figure 2 generated correctly")
