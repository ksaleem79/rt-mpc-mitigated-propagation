import numpy as np
import matplotlib.pyplot as plt

from src.utils import set_seed
from src.mpc_controller import mpc_step

# Set seed for reproducibility
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
x = np.ones(n) * 0.1

# Store exposure
exposure = []

for t in range(T):
    sigma = 1 - noise_std

    # MPC control
    u = mpc_step(x, A, B, Q, R, u_bounds, sigma)

    # Disturbance
    w = np.random.randn(n) * 0.02

    # System update
    x = A @ x + B @ u + G @ w

    exposure.append(np.sum(x))

exposure = np.array(exposure)

# Save results
np.save("results_exposure.npy", exposure)

# Plot
plt.figure()
plt.plot(exposure, label="RT-MPC")
plt.xlabel("Time")
plt.ylabel("Exposure")
plt.legend()
plt.grid()

plt.savefig("Figure_Exposure_Trajectories.png", dpi=300)

print("✅ Simulation completed and figure generated")

