"""
black_scholes.py
----------------
Closed-form (analytical) pricing of European options under the
Black-Scholes-Merton model, together with all first-order Greeks.

The model assumes the underlying asset price S follows a geometric
Brownian motion:

    dS = (r - q) S dt + sigma S dW

where
    r     = risk-free interest rate (continuously compounded)
    q     = continuous dividend yield
    sigma = volatility of the underlying's returns
    W     = standard Brownian motion

Author: Gabriel Egea
"""

from math import log, sqrt, exp
from scipy.stats import norm


# ----------------------------------------------------------------------
# Core building blocks
# ----------------------------------------------------------------------
def _d1_d2(S, K, T, r, sigma, q=0.0):
    """Return the Black-Scholes terms d1 and d2.

    Parameters
    ----------
    S : float   -> current price of the underlying
    K : float   -> strike price
    T : float   -> time to maturity, in years
    r : float   -> risk-free rate (e.g. 0.05 for 5%)
    sigma : float -> volatility (e.g. 0.20 for 20%)
    q : float   -> continuous dividend yield (default 0)
    """
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be strictly positive.")
    d1 = (log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return d1, d2


# ----------------------------------------------------------------------
# Price
# ----------------------------------------------------------------------
def bs_price(S, K, T, r, sigma, option_type="call", q=0.0):
    """Black-Scholes price of a European option.

    option_type : "call" or "put"
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_S = S * exp(-q * T)      # dividend-discounted spot
    disc_K = K * exp(-r * T)      # present value of the strike

    if option_type == "call":
        return disc_S * norm.cdf(d1) - disc_K * norm.cdf(d2)
    elif option_type == "put":
        return disc_K * norm.cdf(-d2) - disc_S * norm.cdf(-d1)
    raise ValueError("option_type must be 'call' or 'put'.")


# ----------------------------------------------------------------------
# Greeks  (sensitivities of the price to each input)
# ----------------------------------------------------------------------
def delta(S, K, T, r, sigma, option_type="call", q=0.0):
    """Sensitivity to the underlying price S."""
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return exp(-q * T) * norm.cdf(d1)
    return -exp(-q * T) * norm.cdf(-d1)


def gamma(S, K, T, r, sigma, q=0.0):
    """Sensitivity of delta to S (same for calls and puts)."""
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return exp(-q * T) * norm.pdf(d1) / (S * sigma * sqrt(T))


def vega(S, K, T, r, sigma, q=0.0):
    """Sensitivity to volatility (same for calls and puts).
    Returned per 1.00 (i.e. +100%) change in sigma.
    Divide by 100 to read it 'per +1% of vol'.
    """
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return S * exp(-q * T) * norm.pdf(d1) * sqrt(T)


def theta(S, K, T, r, sigma, option_type="call", q=0.0):
    """Sensitivity to the passage of time (time decay), per year.
    Divide by 365 to read it 'per calendar day'.
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    term1 = -(S * exp(-q * T) * norm.pdf(d1) * sigma) / (2 * sqrt(T))
    if option_type == "call":
        return (term1
                - r * K * exp(-r * T) * norm.cdf(d2)
                + q * S * exp(-q * T) * norm.cdf(d1))
    return (term1
            + r * K * exp(-r * T) * norm.cdf(-d2)
            - q * S * exp(-q * T) * norm.cdf(-d1))


def rho(S, K, T, r, sigma, option_type="call", q=0.0):
    """Sensitivity to the interest rate.
    Returned per 1.00 (+100%) change in r. Divide by 100 for 'per +1%'.
    """
    _, d2 = _d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return K * T * exp(-r * T) * norm.cdf(d2)
    return -K * T * exp(-r * T) * norm.cdf(-d2)


def greeks(S, K, T, r, sigma, option_type="call", q=0.0):
    """Return every Greek in one dictionary."""
    return {
        "price": bs_price(S, K, T, r, sigma, option_type, q),
        "delta": delta(S, K, T, r, sigma, option_type, q),
        "gamma": gamma(S, K, T, r, sigma, q),
        "vega":  vega(S, K, T, r, sigma, q),
        "theta": theta(S, K, T, r, sigma, option_type, q),
        "rho":   rho(S, K, T, r, sigma, option_type, q),
    }
