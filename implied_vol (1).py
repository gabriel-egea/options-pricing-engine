"""
implied_vol.py
--------------
Recover the *implied volatility* of an option from its market price, by
inverting the Black-Scholes formula numerically.

Black-Scholes price is strictly increasing in sigma (vega > 0), so a unique
implied volatility exists. We use Newton-Raphson (which reuses vega as the
derivative) with a robust Brent fallback when vega becomes too small.

Author: Gabriel Egea
"""

from math import isnan
from scipy.optimize import brentq

from black_scholes import bs_price, vega


def implied_vol(price, S, K, T, r, option_type="call", q=0.0,
                tol=1e-8, max_iter=100):
    """Implied volatility of an option given its (market) price.

    Returns the volatility sigma, or NaN if the price is outside the
    no-arbitrage range (no solution exists).
    """
    # --- Newton-Raphson (fast), using vega as the derivative --------------
    sigma = 0.2  # starting guess: 20%
    for _ in range(max_iter):
        diff = bs_price(S, K, T, r, sigma, option_type, q) - price
        if abs(diff) < tol:
            return sigma
        v = vega(S, K, T, r, sigma, q)          # d(price)/d(sigma)
        if v < 1e-10:                            # vega too small -> stop, use fallback
            break
        sigma -= diff / v
        if sigma <= 0:                           # keep sigma positive
            sigma = 1e-4

    # --- Brent fallback (robust, cannot diverge) --------------------------
    try:
        return brentq(lambda s: bs_price(S, K, T, r, s, option_type, q) - price,
                      1e-6, 5.0, xtol=tol)
    except ValueError:
        return float("nan")   # price outside [intrinsic value, no-arb upper bound]
