"""
market_smile.py
---------------
Build a REAL volatility smile from live option market data.

This is the "real data" companion to implied_vol_demo.py: instead of a
synthetic skew, it downloads a live option chain, computes the implied
volatility of each option with our own solver, and plots the smile.

Requirements:  pip install yfinance   (and an internet connection)
Run with:      python market_smile.py

Note: depending on the yfinance version and the underlying, some column
names or available expiries may differ slightly -- adjust if needed.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import yfinance as yf
from implied_vol import implied_vol

TICKER = "SPY"          # any liquid underlying (SPY, AAPL, MSFT, ...)
r = 0.04                # approximate risk-free rate
EXPIRY_INDEX = 4        # which listed expiry to use (0 = nearest)

tk = yf.Ticker(TICKER)
S = float(tk.history(period="1d")["Close"].iloc[-1])      # current spot price

expiry = tk.options[EXPIRY_INDEX]
T = (datetime.strptime(expiry, "%Y-%m-%d") - datetime.now()).days / 365.0

calls = tk.option_chain(expiry).calls.copy()

# keep liquid quotes and use the mid price (more reliable than last traded)
calls = calls[(calls["bid"] > 0) & (calls["ask"] > 0)]
calls["mid"] = 0.5 * (calls["bid"] + calls["ask"])

# focus on a sensible strike range around the spot
calls = calls[(calls["strike"] > 0.7 * S) & (calls["strike"] < 1.3 * S)]

# our own implied volatility from the mid price
calls["iv_mine"] = [implied_vol(m, S, k, T, r, "call")
                    for m, k in zip(calls["mid"], calls["strike"])]

plt.figure(figsize=(9, 5))
plt.plot(calls["strike"], calls["iv_mine"] * 100, "o-", label="My implied vol")
if "impliedVolatility" in calls:          # yfinance's own IV -> a nice cross-check
    plt.plot(calls["strike"], calls["impliedVolatility"] * 100, "x",
             alpha=0.6, label="yfinance implied vol")
plt.axvline(S, color="black", linestyle=":", label=f"Spot = {S:.0f}")
plt.title(f"{TICKER} volatility smile - expiry {expiry}")
plt.xlabel("Strike"); plt.ylabel("Implied volatility (%)")
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("market_smile.png", dpi=120)

print(f"{TICKER} spot = {S:.2f}, expiry {expiry}, T = {T:.3f} yr")
print("Saved figure: market_smile.png")
