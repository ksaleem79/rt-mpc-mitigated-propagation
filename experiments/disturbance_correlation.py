import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.mpc_controller import mpc_step
from src.utils import set_seed

# ===========================
# SETTINGS
# ===========================
set_seed(0)

n = 30
m = 3
samples = 60   # number of disturbance samples

A = np.eye(n) * 1.002
B = np.random.randn(n, m) * 0.05
G = np.eye(n)

Q = np.eye(n)
R = np.eye(m)

u_bounds = (-5.0, 5.0)
sigma = 0.8

threshold = 5.0

# ===========================
# STORAGE
# ===========================
disturbances = []
resp_nc = []
resp_ht = []
resp_mpc = []

# ===========================
# MAIN LOOP
# ===========================
for i in range(samples):

    # Generate disturbance magnitude
    d = np.random.uniform(1.5, 5.5)

    disturbances.append(d)

    # scale disturbance vector
    w = np.random.randn(n) * 0.05 + d * 0.02

    # initial state
    x_nc = np.ones(n) * 0.1
    x_ht = np.ones(n) * 0.1
    x_mpc = np.ones(n) * 0.1

    # propagate short horizon
    for _ in range(10):

        # NC
        x_nc = A @ x_nc + G @ w

        # HT
        if np.sum(x_ht) > threshold:
            u_ht = -0.1 * np.ones(m)
        else:
            u_ht = np.zeros(m)

        x_ht = A @ x_ht + B @ u_ht + G @ w

        # MPC
        u_mpc = mpc_step(x_mpc, A, B, Q, R, u_bounds, sigma)
        x_mpc = A @ x_mpc + B @ u_mpc + G @ w

    # store exposure response
    resp_nc.append(np.mean(x_nc))
    resp_ht.append(np.mean(x_ht))
    resp_mpc.append(np.mean(x_mpc))


# ===========================
# SAVE DATA
# ===========================
df = pd.DataFrame({
    "disturbance": disturbances,
    "NC": resp_nc,
    "HT": resp_ht,
    "RT_MPC": resp_mpc
})

os.makedirs("results", exist_ok=True)
df.to_csv("results/disturbance_response.csv", index=False)

print("✅ Data saved to results/disturbance_response.csv")

# ===========================
# PLOT FIGURE 7
# ===========================
plt.figure(figsize=(8,5))

plt.scatter(disturbances, resp_nc, label="NC (Uncontrolled)", alpha=0.6)
plt.scatter(disturbances, resp_ht, label="HT (Heuristic)", alpha=0.6)
plt.scatter(disturbances, resp_mpc, label="RT-MPC (Proposed)", alpha=0.6)

plt.xlabel("Adversarial disturbance magnitude")
plt.ylabel("System response (Exposure level)")

plt.title("Disturbance–Response Relationship Across Mitigation Strategies")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("Figure_Disturbance_Response_Correlation.png", dpi=300)

print("✅ Figure 7 generated successfully")
