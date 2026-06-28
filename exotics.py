"""
exotics.py
----------
Monte Carlo pricing of path-dependent (exotic) options, which have no
closed-form solution in general and therefore require full path simulation.

Implemented:
  - simulate_paths            : risk-neutral GBM paths
  - asian_option              : arithmetic or geometric average options
  - barrier_option            : up/down & in/out barrier options
  - geometric_asian_closed_form : exact price for the DISCRETE geometric
                                  average (used to validate the path engine)

Author: Gabriel Egea
"""

import numpy as np
from math import sqrt, exp, log
from scipy.stats import norm


# ----------------------------------------------------------------------
# Path simulation
# ----------------------------------------------------------------------
def simulate_paths(S, T, r, sigma, n_steps, n_paths, q=0.0,
                   antithetic=False, seed=None):
    """Return an array of shape (n_paths, n_steps+1) of GBM paths.
    Column 0 is the initial price S; columns 1..n_steps are the steps.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    if antithetic:
        half = rng.standard_normal((n_paths // 2, n_steps))
        Z = np.vstack([half, -half])
    else:
        Z = rng.standard_normal((n_paths, n_steps))

    log_increments = (r - q - 0.5 * sigma ** 2) * dt + sigma * sqrt(dt) * Z
    log_paths = np.cumsum(log_increments, axis=1)
    paths = S * np.exp(log_paths)
    # prepend the initial price as column 0
    return np.hstack([np.full((paths.shape[0], 1), S), paths])


def _summary(discounted_payoffs, antithetic=False):
    """Helper: turn a vector of discounted payoffs into (price, stderr, ci).
    With antithetic variates the independent units are the M/2 PAIR AVERAGES,
    so the standard error must be computed on them, not on the M raw payoffs.
    """
    d = discounted_payoffs
    if antithetic:
        n = len(d) // 2
        d = 0.5 * (d[:n] + d[n:2 * n])
    price = d.mean()
    stderr = d.std(ddof=1) / sqrt(len(d))
    ci = (price - 1.96 * stderr, price + 1.96 * stderr)
    return price, stderr, ci


# ----------------------------------------------------------------------
# Asian options (payoff on the average price)
# ----------------------------------------------------------------------
def asian_option(S, K, T, r, sigma, n_steps=100, n_paths=200_000,
                 option_type="call", average="arithmetic", q=0.0,
                 antithetic=True, seed=None):
    """Price an Asian option. average = 'arithmetic' or 'geometric'.
    The average is taken over the monitoring dates (steps 1..n_steps).
    """
    paths = simulate_paths(S, T, r, sigma, n_steps, n_paths, q, antithetic, seed)
    monitor = paths[:, 1:]                       # exclude t=0

    if average == "arithmetic":
        avg = monitor.mean(axis=1)
    elif average == "geometric":
        avg = np.exp(np.log(monitor).mean(axis=1))
    else:
        raise ValueError("average must be 'arithmetic' or 'geometric'.")

    if option_type == "call":
        payoff = np.maximum(avg - K, 0.0)
    else:
        payoff = np.maximum(K - avg, 0.0)

    return _summary(exp(-r * T) * payoff, antithetic)


def geometric_asian_closed_form(S, K, T, r, sigma, n_steps,
                                option_type="call", q=0.0):
    """Exact price of a DISCRETE geometric-average Asian option.

    The geometric average of log-normal prices is itself log-normal, so a
    closed form exists. Used here to validate the Monte Carlo path engine.
    """
    dt = T / n_steps
    n = n_steps
    # mean and variance of ln(G), where G is the geometric average
    mu = log(S) + (r - q - 0.5 * sigma ** 2) * dt * (n + 1) / 2
    var = sigma ** 2 * dt * (n + 1) * (2 * n + 1) / (6 * n)
    sd = sqrt(var)

    d1 = (mu + var - log(K)) / sd
    d2 = d1 - sd
    if option_type == "call":
        price = exp(-r * T) * (exp(mu + 0.5 * var) * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        price = exp(-r * T) * (K * norm.cdf(-d2) - exp(mu + 0.5 * var) * norm.cdf(-d1))
    return price


# ----------------------------------------------------------------------
# Barrier options (payoff depends on touching a barrier level B)
# ----------------------------------------------------------------------
def barrier_option(S, K, T, r, sigma, B, n_steps=252, n_paths=100_000,
                   option_type="call", barrier_type="up-and-out",
                   q=0.0, antithetic=True, seed=None):
    """Price a barrier option.

    barrier_type : 'up-and-out', 'up-and-in', 'down-and-out', 'down-and-in'
    Note: the barrier is monitored at the discrete simulation steps only,
    which introduces a (downward, for knock-outs) discretization bias.
    """
    paths = simulate_paths(S, T, r, sigma, n_steps, n_paths, q, antithetic, seed)
    ST = paths[:, -1]

    if option_type == "call":
        vanilla = np.maximum(ST - K, 0.0)
    else:
        vanilla = np.maximum(K - ST, 0.0)

    if barrier_type.startswith("up"):
        breached = paths.max(axis=1) >= B
    elif barrier_type.startswith("down"):
        breached = paths.min(axis=1) <= B
    else:
        raise ValueError("barrier_type must start with 'up' or 'down'.")

    if barrier_type.endswith("out"):
        payoff = vanilla * (~breached)      # survives only if NOT breached
    elif barrier_type.endswith("in"):
        payoff = vanilla * breached         # active only if breached
    else:
        raise ValueError("barrier_type must end with 'out' or 'in'.")

    return _summary(exp(-r * T) * payoff, antithetic)
