"""
monte_carlo.py
--------------
Monte Carlo pricing of European options under the risk-neutral measure.

The risk-neutral terminal price of a geometric Brownian motion is

    S_T = S0 * exp( (r - q - 0.5*sigma^2) T + sigma*sqrt(T)*Z ),   Z ~ N(0,1)

The option price is the discounted average payoff over many simulated S_T.
Because we average random samples, we also report the standard error and a
95% confidence interval. Antithetic variates are used to reduce variance.

Author: Gabriel Egea
"""

import numpy as np
from math import sqrt, exp


def mc_price_european(S, K, T, r, sigma, M=100_000,
                      option_type="call", q=0.0, antithetic=True, seed=None):
    """Monte Carlo price of a European option.

    Returns
    -------
    price : float                 -> Monte Carlo estimate
    stderr : float                -> standard error of that estimate
    ci : (float, float)           -> 95% confidence interval
    """
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")

    rng = np.random.default_rng(seed)

    # draw the normal shocks (antithetic: pair each Z with -Z)
    if antithetic:
        half = rng.standard_normal(M // 2)
        Z = np.concatenate([half, -half])
    else:
        Z = rng.standard_normal(M)

    # simulate terminal prices and payoffs
    ST = S * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * sqrt(T) * Z)
    if option_type == "call":
        payoff = np.maximum(ST - K, 0.0)
    else:
        payoff = np.maximum(K - ST, 0.0)

    discounted = exp(-r * T) * payoff

    # The independent sampling units are:
    #  - the M payoffs themselves (plain MC), or
    #  - the M/2 PAIR AVERAGES 0.5*(g(Z)+g(-Z)) for antithetic variates,
    #    since Z and -Z are not independent. Using the pair averages is what
    #    correctly reflects the variance reduction.
    if antithetic:
        n = M // 2
        samples = 0.5 * (discounted[:n] + discounted[n:2 * n])
    else:
        samples = discounted

    price = samples.mean()
    stderr = samples.std(ddof=1) / sqrt(len(samples))
    ci = (price - 1.96 * stderr, price + 1.96 * stderr)
    return price, stderr, ci
