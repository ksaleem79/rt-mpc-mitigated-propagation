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

set_seed(0)

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

    x = np.random.randn(n)
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

# retain all runtime observations
# rt_mpc_times = rt_mpc_times[rt_mpc_times < np.percentile(rt_mpc_times, 99)]
# ht_times = ht_times[ht_times < np.percentile(ht_times, 99)]

print("✅ Runtime stats:")
print("RT-MPC mean:", np.mean(rt_mpc_times), "ms")
print("HT mean:", np.mean(ht_times), "ms")

print("HT median:",
      np.median(ht_times))

print("HT p95:",
      np.percentile(ht_times,95))

print("HT p99:",
      np.percentile(ht_times,99))

print("HT max:",
      np.max(ht_times))
      

# ---------- Plot ----------

plt.figure(figsize=(8,5))

plt.boxplot(
    [ht_times, rt_mpc_times],
    labels=[
        "HT",
        "RT-MPC"
    ]
)

plt.yscale("log")

plt.ylabel("Computation Time (ms, log scale)")
plt.title("Computation-Time Comparison")
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "Figure_Computation_Times.png",
    dpi=300,
    bbox_inches="tight"
)

print("✅ Figure 4 generated")