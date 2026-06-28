"""
binomial.py
-----------
Option pricing with the Cox-Ross-Rubinstein (CRR) binomial tree.

Time to maturity T is split into N steps of length dt = T/N. At each step
the underlying moves up by a factor u or down by a factor d. Pricing is done
by backward induction from the leaves (maturity) to the root (today).

Unlike Black-Scholes, the tree also prices AMERICAN options, because early
exercise can be checked at every node.

Author: Gabriel Egea
"""

import numpy as np
from math import exp, sqrt


def binomial_price(S, K, T, r, sigma, N=500,
                   option_type="call", exercise="european", q=0.0):
    """Price a European or American option with a CRR binomial tree.

    Parameters
    ----------
    S, K, T, r, sigma, q : floats   -> same meaning as in black_scholes.py
    N : int                         -> number of time steps (more = more accurate)
    option_type : "call" or "put"
    exercise    : "european" or "american"
    """
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")
    if exercise not in ("european", "american"):
        raise ValueError("exercise must be 'european' or 'american'.")

    dt = T / N
    u = exp(sigma * sqrt(dt))         # up factor
    d = 1.0 / u                       # down factor (CRR symmetry)
    p = (exp((r - q) * dt) - d) / (u - d)   # risk-neutral probability
    disc = exp(-r * dt)               # one-step discount factor

    if not (0.0 < p < 1.0):
        raise ValueError("Risk-neutral probability outside (0,1): "
                         "increase N or check the inputs.")

    # --- payoff at maturity (the N+1 leaves) -------------------------------
    j = np.arange(N + 1)              # number of up-moves at each leaf
    ST = S * u**j * d**(N - j)        # terminal underlying prices
    if option_type == "call":
        values = np.maximum(ST - K, 0.0)
    else:
        values = np.maximum(K - ST, 0.0)

    # --- backward induction ------------------------------------------------
    for _ in range(N):
        # continuation value (risk-neutral discounted expectation)
        values = disc * (p * values[1:] + (1 - p) * values[:-1])

        if exercise == "american":
            level = len(values) - 1           # current tree level
            j = np.arange(level + 1)
            St = S * u**j * d**(level - j)     # underlying prices at this level
            if option_type == "call":
                immediate = np.maximum(St - K, 0.0)
            else:
                immediate = np.maximum(K - St, 0.0)
            values = np.maximum(values, immediate)   # early-exercise check

    return values[0]
