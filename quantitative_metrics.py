import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..'
        )
    )
)

import numpy as np

from src.mpc_controller import mpc_step
from src.utils import set_seed

# =====================================================
# REPRODUCIBILITY
# =====================================================

set_seed(0)

# =====================================================
# CONFIGURATION
# =====================================================

n = 30
m = 3

T = 60
runs = 30

A = np.eye(n) * 1.002

B = np.random.randn(n, m) * 0.05

G = np.eye(n)

Q = np.eye(n)
R = np.eye(m)

sigma = 0.8

u_bounds = (-6.0, 6.0)

threshold = 50.0
ht_intervention = -0.1

# =====================================================
# STORAGE
# =====================================================

results = {}

# =====================================================
# MAIN EVALUATION
# =====================================================

for method in ["NC", "HT", "RT-MPC"]:

    cumulative_exposure = []
    control_costs = []
    violations = []

    for run in range(runs):

        set_seed(run)

        x = np.ones(n) * 3.0

        exposure_sum = 0.0
        cost_sum = 0.0
        violation_count = 0

        for t in range(T):

            w = np.random.randn(n) * 0.04 + 0.02

            # -----------------------------------------
            # NC
            # -----------------------------------------

            if method == "NC":

                u = np.zeros(m)

            # -----------------------------------------
            # HT
            # -----------------------------------------

            elif method == "HT":

                if np.sum(x) > threshold:
                    u = ht_intervention * np.ones(m)
                else:
                    u = np.zeros(m)

            # -----------------------------------------
            # RT-MPC
            # -----------------------------------------

            else:

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
                    sigma
                )

            # propagate

            x = (
                A @ x
                + B @ u
                + G @ w
            )

            exposure_sum += np.sum(x)

            cost_sum += np.linalg.norm(u)

            # box-bound violations

            if (
                np.any(u < u_bounds[0])
                or np.any(u > u_bounds[1])
            ):
                violation_count += 1

        cumulative_exposure.append(
            exposure_sum
        )

        control_costs.append(
            cost_sum
        )

        violations.append(
            violation_count
        )

    results[method] = {
        "Exposure":
            np.mean(cumulative_exposure),

        "ControlCost":
            np.mean(control_costs),

        "Variability":
            np.std(cumulative_exposure),

        "Violations":
            np.mean(violations)
    }

# =====================================================
# RESULTS
# =====================================================

print("\n=== Quantitative Results ===\n")

for method in results:

    r = results[method]

    print(
        f"{method}: "
        f"Exposure={r['Exposure']:.2f}, "
        f"Cost={r['ControlCost']:.2f}, "
        f"Variability={r['Variability']:.2f}, "
        f"Violations={r['Violations']:.2f}"
    )