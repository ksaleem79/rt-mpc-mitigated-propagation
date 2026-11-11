# experiments/ablation_analysis.py

import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import numpy as np

from src.mpc_controller import mpc_step
from src.utils import set_seed

# ======================================================
# REPRODUCIBILITY
# ======================================================

set_seed(0)

# ======================================================
# PARAMETERS
# ======================================================

n = 30
m = 3

T = 60
runs = 30

A = np.eye(n) * 1.002

B = np.random.randn(n, m) * 0.05

G = np.eye(n)

Q = np.eye(n)
R = np.eye(m)

u_bounds = (-6.0, 6.0)

threshold = 50.0
ht_intervention = -0.1

# ======================================================
# HELPER
# ======================================================

def simulate(mode):

    exposures = []
    costs = []

    for r in range(runs):

        set_seed(r)

        x = np.ones(n) * 3.0

        total_exp = 0.0
        total_cost = 0.0

        for t in range(T):

            w = np.random.randn(n) * 0.04 + 0.02

            # -----------------------------------------
            # HT
            # -----------------------------------------

            if mode == "HT":

                if np.sum(x) > threshold:
                    u = ht_intervention * np.ones(m)
                else:
                    u = np.zeros(m)

            # -----------------------------------------
            # REMOVE UNCERTAINTY PENALTY
            # -----------------------------------------

            elif mode == "NO_UNCERTAINTY":

                sigma = 0.8

                measurement_noise = (
                    np.random.randn(n)
                    * (1.0 - sigma)
                    * 0.2
                )

                x_hat = x + measurement_noise

                u = mpc_step(
                    x_hat,
                    A,
                    B,
                    Q,
                    R,
                    u_bounds,
                    sigma,
                    horizon=5,
                    lambda_unc=0.0,
                    terminal_weight=1.0
                )

            # -----------------------------------------
            # REMOVE TERMINAL VALUE
            # -----------------------------------------

            elif mode == "NO_TERMINAL":

                sigma = 0.8

                measurement_noise = (
                    np.random.randn(n)
                    * (1.0 - sigma)
                    * 0.2
                )

                x_hat = x + measurement_noise

                u = mpc_step(
                    x_hat,
                    A,
                    B,
                    Q,
                    R,
                    u_bounds,
                    sigma,
                    horizon=5,
                    lambda_unc=1.0,
                    terminal_weight=0.0
                )

            # -----------------------------------------
            # FULL RT-MPC
            # -----------------------------------------

            else:

                sigma = 0.8

                measurement_noise = (
                    np.random.randn(n)
                    * (1.0 - sigma)
                    * 0.2
                )

                x_hat = x + measurement_noise

                u = mpc_step(
                    x_hat,
                    A,
                    B,
                    Q,
                    R,
                    u_bounds,
                    sigma,
                    horizon=5,
                    lambda_unc=1.0,
                    terminal_weight=1.0
                )

            x = (
                A @ x
                + B @ u
                + G @ w
            )

            total_exp += np.sum(x)

            total_cost += np.linalg.norm(u)

        exposures.append(total_exp)
        costs.append(total_cost)

    return (
        np.mean(exposures),
        np.mean(costs)
    )

# ======================================================
# ABLATION
# ======================================================

results = {}

results["HT"] = simulate("HT")
results["No Uncertainty"] = simulate("NO_UNCERTAINTY")
results["No Terminal"] = simulate("NO_TERMINAL")
results["Full RT-MPC"] = simulate("FULL")

print("\n=== Ablation Results ===\n")

for name, (exposure, cost) in results.items():

    print(
        f"{name}: "
        f"Exposure={exposure:.2f}, "
        f"Cost={cost:.2f}"
    )