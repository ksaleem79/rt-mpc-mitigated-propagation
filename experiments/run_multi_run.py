import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

from src.utils import set_seed
from src.mpc_controller import mpc_step

# Parameters
n = 30
m = 3
T = 200
runs = 20

A = np.eye(n) * 0.98
B = np.random.randn(n, m) * 0.05
G = np.eye(n)

Q = np.eye(n)
R = np.eye(m)

noise_std = 0.2
u_bounds = (-0.5, 0.5)

all_exposures = []

for r in range(runs):
    set_seed(r)

    x = np.ones(n) * 0.1
    total_exposure = 0

    for t in range(T):
        sigma = 1 - noise_std

        u = mpc_step(x, A, B, Q, R, u_bounds, sigma)

        w = np.random.randn(n) * 0.02
        x = A @ x + B @ u + G @ w

        total_exposure += np.sum(x)

    all_exposures.append(total_exposure)

all_exposures = np.array(all_exposures)

print("✅ Multi-run results:")
print("Mean exposure:", np.mean(all_exposures))
print("Std deviation:", np.std(all_exposures))

# Plot
plt.figure()
plt.hist(all_exposures, bins=10)
plt.xlabel("Cumulative Exposure")
plt.ylabel("Frequency")
plt.title("Multi-Run Robustness")

plt.savefig("Figure_MultiRun_Robustness.png", dpi=300)

print("✅ Multi-run figure generated")
