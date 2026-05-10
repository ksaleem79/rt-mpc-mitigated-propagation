import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

from src.mpc_controller import mpc_step
from src.utils import set_seed

# ===============================
# ✅ Reproducibility
# ===============================
set_seed(0)

# ===============================
# ✅ Parameters
# ===============================
n = 30
m = 3
T = 200

# SYSTEM DYNAMICS
# Slight growth → adversarial amplification possible
A = np.eye(n) * 1.002

# Control influence
B = np.random.randn(n, m) * 0.08
G = np.eye(n)

Q = np.eye(n)
R = np.eye(m)

# Detector confidence
noise_std = 0.2
sigma = 1 - noise_std

# Stronger control limits
u_bounds = (-5.0, 5.0)

# Heuristic threshold
threshold = 5.0

# ===============================
# ✅ Initial states
# ===============================
x_nc = np.ones(n) * 0.1
x_ht = np.ones(n) * 0.1
x_mpc = np.ones(n) * 0.1

# Storage
exp_nc = []
exp_ht = []
exp_mpc = []

# ===============================
# ✅ Simulation loop
# ===============================
for t in range(T):

    # Persistent adversarial injection
    w = np.random.randn(n) * 0.04 + 0.02

    # -------- NC (No Control)
    u_nc = np.zeros(m)
    x_nc = A @ x_nc + B @ u_nc + G @ w
    exp_nc.append(np.sum(x_nc))

    # -------- HT (Heuristic Threshold)
    if np.sum(x_ht) > threshold:
        u_ht = -0.1 * np.ones(m)
    else:
        u_ht = np.zeros(m)

    x_ht = A @ x_ht + B @ u_ht + G @ w
    exp_ht.append(np.sum(x_ht))

    # -------- RT‑MPC (Proposed)
    u_mpc = mpc_step(x_mpc, A, B, Q, R, u_bounds, sigma)
    x_mpc = A @ x_mpc + B @ u_mpc + G @ w
    exp_mpc.append(np.sum(x_mpc))

# Convert arrays
exp_nc = np.array(exp_nc)
exp_ht = np.array(exp_ht)
exp_mpc = np.array(exp_mpc)

# ===============================
# ✅ Plot
# ===============================
plt.figure(figsize=(8,5))

plt.plot(exp_nc, label="NC (Uncontrolled)", linewidth=2)
plt.plot(exp_ht, label="HT (Heuristic)", linestyle="--")
plt.plot(exp_mpc, label="RT-MPC (Controlled response)", linewidth=2)

plt.xlabel("Time step k")
plt.ylabel("Exposure level")
plt.title("Exposure Dynamics Under Persistent Adversarial Pressure")

# Annotation
plt.annotate("Controlled growth (RT-MPC)",
             xy=(150, exp_mpc[150]),
             xytext=(100, exp_mpc[150] + 20),
             arrowprops=dict(arrowstyle="->"))

# Inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

ax = plt.gca()

axins = inset_axes(ax, width="40%", height="30%", loc='lower right')

axins.plot(exp_nc[:50])
axins.plot(exp_ht[:50])
axins.plot(exp_mpc[:50])

axins.set_title("Early-phase behavior")
axins.grid(True)

plt.legend()
plt.grid(alpha=0.3)

plt.subplots_adjust()   # ✅ replaces tight_layout

plt.savefig("Figure_Adversarial_StressTest.png", dpi=300)

print("✅ Figure 5 generated successfully")
