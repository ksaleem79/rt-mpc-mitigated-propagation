import numpy as np
from scipy.optimize import minimize


def mpc_step(
    x,
    A,
    B,
    Q,
    R,
    u_bounds,
    sigma,
    horizon=5,
    lambda_unc=1.0,
    terminal_weight=1.0
):
    """
    Finite-horizon MPC controller.

    Parameters
    ----------
    x : ndarray
        Current state.

    A, B : ndarray
        System matrices.

    Q, R : ndarray
        State and control weights.

    u_bounds : tuple
        (umin, umax)

    sigma : float
        Detector confidence in [0,1].

    horizon : int
        Prediction horizon.

    lambda_unc : float
        Uncertainty penalty weight.

    terminal_weight : float
        Terminal value weight.

    Returns
    -------
    u : ndarray
        First MPC control action.
    """

    n = A.shape[0]
    m = B.shape[1]

    umin, umax = u_bounds

    U0 = np.zeros(horizon * m)

    bounds = [(umin, umax)] * (horizon * m)

    def cost(U):

        xp = x.copy()

        J = 0.0

        for k in range(horizon):

            uk = U[k * m:(k + 1) * m]

            # stage cost
            J += xp.T @ Q @ xp

            # control effort cost
            J += uk.T @ R @ uk

            # uncertainty-aware penalty
            J += (
                lambda_unc
                * (1.0 - sigma)
                * np.linalg.norm(uk, 1)
            )

            # prediction model
            xp = A @ xp + B @ uk

        # terminal cost
        J += (
            terminal_weight
            * (xp.T @ Q @ xp)
        )

        return float(J)

    result = minimize(
        cost,
        U0,
        method="SLSQP",
        bounds=bounds,
        options={
            "maxiter": 100,
            "ftol": 1e-6,
            "disp": False
        }
    )

    if not result.success:
        return np.zeros(m)

    return result.x[:m]