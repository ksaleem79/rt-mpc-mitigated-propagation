import numpy as np

def mpc_step(x, A, B, Q, R, u_bounds, sigma, horizon=5):
    """
    Simple uncertainty-aware MPC approximation.

    Parameters:
        x: current state
        A, B: system matrices
        Q, R: cost matrices
        u_bounds: (u_min, u_max)
        sigma: confidence level (0-1)
        horizon: planning horizon

    Returns:
        control input u
    """

    u = -np.sign(B.T @ (Q @ x))

    # balanced control gain
    u = u * (1.0 + sigma)

    u = np.clip(u, u_bounds[0], u_bounds[1])

    return u
