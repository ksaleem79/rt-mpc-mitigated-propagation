import numpy as np
import matplotlib.pyplot as plt

data = np.load("results/exposure_runs.npy")

cum_nc, cum_ht, cum_mpc = data

plt.figure(figsize=(8,5))

plt.plot(cum_nc, label="NC (Uncontrolled)")
plt.plot(cum_ht, label="HT (Heuristic)")
plt.plot(cum_mpc, label="RT-MPC (Proposed)")

plt.xlabel("Time step")
plt.ylabel("Cumulative Exposure")
plt.title("Cumulative Exposure")

plt.legend()
plt.grid()

plt.savefig("Figure_Cumulative_Exposure.png", dpi=300)
