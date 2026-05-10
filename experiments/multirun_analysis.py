import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

from src.mpc_controller import mpc_step
from src.utils import set_seed

# ===============================
# ✅ PARAMETERS
# ===============================
n = 30
m = 3
T = 60
runs = 30   # Monte Carlo runs

A = np.eye(n) * 1.002
B = np.random.randn(n, m) * 0.05
G = np.eye(n)

Q = np.eye(n)
R = np.eye(m)

sigma = 0.8
u_bounds = (-6.0, 6.0)

threshold = 50.0

# ===============================
# ✅ STORAGE
# ===============================
results_nc = []
results_ht = []
results_mpc = []

# ===============================
# ✅ MAIN SIMULATION LOOP
# ===============================
for r in range(runs):

    set_seed(r)

    x_nc = np.ones(n) * 3.0
    x_ht = np.ones(n) * 3.0
    x_mpc = np.ones(n) * 3.0

    total_nc = 0
    total_ht = 0
    total_mpc = 0

    for t in range(T):

        w = np.random.randn(n) * 0.04 + 0.02

        # NC
        u_nc = np.zeros(m)
        x_nc = A @ x_nc + G @ w
        total_nc += np.sum(x_nc)

        # HT
        if np.sum(x_ht) > threshold:
            u_ht = -1.0 * np.ones(m)
        else:
            u_ht = np.zeros(m)

        x_ht = A @ x_ht + B @ u_ht + G @ w
        total_ht += np.sum(x_ht)

        # MPC
        u_mpc = mpc_step(x_mpc, A, B, Q, R, u_bounds, sigma)
        x_mpc = A @ x_mpc + B @ u_mpc + G @ w
        total_mpc += np.sum(x_mpc)

    results_nc.append(total_nc)
    results_ht.append(total_ht)
    results_mpc.append(total_mpc)

# ===============================
# ✅ SAVE RESULTS
# ===============================
os.makedirs("results", exist_ok=True)

np.save(
    "results/multirun_results.npy",
    np.array([results_nc, results_ht, results_mpc])
)

print("✅ Multi-run data saved in results folder")

# ===============================
# ✅ PLOT (ENHANCED BOXPLOT)
# ===============================
plt.figure(figsize=(8,5))

data = [results_nc, results_ht, results_mpc]

# ✅ Boxplot with color fill + mean markers
box = plt.boxplot(
    data,
    labels=["NC", "HT", "RT-MPC"],
    patch_artist=True,
    showmeans=True
)

# ✅ Colors
colors = ["lightblue", "orange", "lightgreen"]

for patch, color in zip(box["boxes"], colors):
    patch.set_facecolor(color)

# ✅ Styling median & mean
for median in box["medians"]:
    median.set(color="black", linewidth=2)

for mean in box["means"]:
    mean.set(marker="o", markerfacecolor="black", markeredgecolor="black")

# ===============================
# ✅ LABELS
# ===============================
plt.ylabel("Cumulative Exposure")
plt.title("Multi-Run Robustness Analysis")

plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("Figure_MultiRun_Robustness.png", dpi=300)

print("✅ Figure 10 generated successfully")
