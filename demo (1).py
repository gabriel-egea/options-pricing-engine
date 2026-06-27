"""
demo.py
-------
Small demonstration of the Black-Scholes pricer:
  1. price a European call and put,
  2. print all the Greeks,
  3. check put-call parity (a built-in sanity test),
  4. plot option price as a function of the underlying price.

Run it with:  python demo.py
"""

import numpy as np
import matplotlib.pyplot as plt

import black_scholes as bs

# ----------------------------------------------------------------------
# 1. Example contract parameters
# ----------------------------------------------------------------------
S = 100.0     # underlying price
K = 100.0     # strike (at-the-money here)
T = 1.0       # 1 year to maturity
r = 0.05      # 5% risk-free rate
sigma = 0.20  # 20% volatility
q = 0.0       # no dividends

call = bs.bs_price(S, K, T, r, sigma, "call", q)
put = bs.bs_price(S, K, T, r, sigma, "put", q)

print(f"Inputs: S={S}, K={K}, T={T}, r={r}, sigma={sigma}, q={q}\n")
print(f"Call price : {call:.4f}")
print(f"Put  price : {put:.4f}\n")

# ----------------------------------------------------------------------
# 2. Greeks of the call
# ----------------------------------------------------------------------
g = bs.greeks(S, K, T, r, sigma, "call", q)
print("Greeks (call):")
print(f"  delta = {g['delta']:.4f}")
print(f"  gamma = {g['gamma']:.4f}")
print(f"  vega  = {g['vega']:.4f}   (per +1% vol: {g['vega']/100:.4f})")
print(f"  theta = {g['theta']:.4f}  (per day:     {g['theta']/365:.4f})")
print(f"  rho   = {g['rho']:.4f}   (per +1% rate: {g['rho']/100:.4f})\n")

# ----------------------------------------------------------------------
# 3. Sanity test: put-call parity
#    C - P  must equal  S*e^{-qT} - K*e^{-rT}
# ----------------------------------------------------------------------
lhs = call - put
rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
print("Put-call parity check:")
print(f"  C - P            = {lhs:.6f}")
print(f"  S e^-qT - K e^-rT = {rhs:.6f}")
print(f"  match: {np.isclose(lhs, rhs)}\n")

# ----------------------------------------------------------------------
# 4. Plot: price vs underlying price
# ----------------------------------------------------------------------
spots = np.linspace(50, 150, 200)
calls = [bs.bs_price(s, K, T, r, sigma, "call", q) for s in spots]
puts = [bs.bs_price(s, K, T, r, sigma, "put", q) for s in spots]

plt.figure(figsize=(8, 5))
plt.plot(spots, calls, label="Call value")
plt.plot(spots, puts, label="Put value")
plt.axvline(K, color="grey", linestyle="--", linewidth=1, label="Strike K")
plt.title("Black-Scholes option value vs underlying price")
plt.xlabel("Underlying price S")
plt.ylabel("Option value")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("bs_price_vs_spot.png", dpi=120)
print("Saved figure: bs_price_vs_spot.png")
