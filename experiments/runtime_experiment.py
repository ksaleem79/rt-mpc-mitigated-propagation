import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import time

from src.mpc_controller import mpc_step
from src.utils import set_seed

# PARAMETERS
n = 30
m = 3
runs = 300
iterations = 100  # repeat operations to get stable measurement

A = np.eye(n) * 0.98
B = np.random.randn(n, m) * 0.05
Q = np.eye(n)
R = np.eye(m)

noise_std = 0.2
u_bounds = (-0.5, 0.5)
threshold = 5.0

rt_mpc_times = []
ht_times = []

for r in range(runs):
    set_seed(r)

    x = np.ones(n) * 0.1
    sigma = 1 - noise_std

    # ---------- RT-MPC timing ----------
    start = time.perf_counter()

    for _ in range(iterations):
        u = mpc_step(x, A, B, Q, R, u_bounds, sigma)

    end = time.perf_counter()

    rt_mpc_times.append((end - start) * 1000 / iterations)

    # ---------- HT timing ----------
    start = time.perf_counter()

    for _ in range(iterations):
        if np.sum(x) > threshold:
            u_ht = -0.3 * np.ones(m)
        else:
            u_ht = np.zeros(m)

    end = time.perf_counter()

    ht_times.append((end - start) * 1000 / iterations)

# Convert arrays
rt_mpc_times = np.array(rt_mpc_times)
ht_times = np.array(ht_times)

# Remove outliers (top 1%)
rt_mpc_times = rt_mpc_times[rt_mpc_times < np.percentile(rt_mpc_times, 99)]
ht_times = ht_times[ht_times < np.percentile(ht_times, 99)]

print("✅ Runtime stats:")
print("RT-MPC mean:", np.mean(rt_mpc_times), "ms")
print("HT mean:", np.mean(ht_times), "ms")

# ---------- Plot ----------
plt.figure(figsize=(8,5))

plt.hist(ht_times, bins=30, alpha=0.6, label="HT (Heuristic Threshold)")
plt.hist(rt_mpc_times, bins=30, alpha=0.6, label="RT-MPC (Proposed)")

plt.xlabel("Computation Time per Step (ms)")
plt.ylabel("Frequency")
plt.title("Computation-Time Distribution for Mitigation Algorithms")

plt.legend()
plt.grid(alpha=0.3)

plt.savefig("Figure_Computation_Times.png", dpi=300)

print("✅ Figure 4 generated")
