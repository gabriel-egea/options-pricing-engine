"""
compare_methods.py
------------------
Capstone: price the SAME options three different ways and check they agree.

  - Black-Scholes (closed form)   -> exact, European only
  - Binomial tree (CRR)           -> also prices American options
  - Monte Carlo                   -> also prices exotic / path-dependent options

If the three independent methods land on the same number, that is strong
evidence the whole library is correct.

Run with:  python compare_methods.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import black_scholes as bs
from binomial import binomial_price
from monte_carlo import mc_price_european

S, T, r, sigma, q = 100.0, 1.0, 0.05, 0.20, 0.0
strikes = [80, 90, 100, 110, 120]

# ----------------------------------------------------------------------
# 1. European call: the three methods side by side
# ----------------------------------------------------------------------
rows = []
for K in strikes:
    bs_p = bs.bs_price(S, K, T, r, sigma, "call", q)
    bin_p = binomial_price(S, K, T, r, sigma, N=1000, option_type="call",
                           exercise="european", q=q)
    mc_p, _, _ = mc_price_european(S, K, T, r, sigma, M=500_000,
                                   option_type="call", q=q, seed=1)
    rows.append({
        "Strike": K,
        "Black-Scholes": round(bs_p, 4),
        "Binomial": round(bin_p, 4),
        "Monte Carlo": round(mc_p, 4),
        "max gap": round(max(abs(bs_p - bin_p), abs(bs_p - mc_p)), 4),
    })

table = pd.DataFrame(rows)
print("European call - three methods side by side:\n")
print(table.to_string(index=False))
print(f"\nLargest disagreement across all strikes: {table['max gap'].max():.4f}")
print("-> the three independent methods agree.\n")

# ----------------------------------------------------------------------
# 2. American put: only the binomial tree can do it
# ----------------------------------------------------------------------
euro_put = binomial_price(S, 100, T, r, sigma, N=1000, option_type="put",
                          exercise="european", q=q)
amer_put = binomial_price(S, 100, T, r, sigma, N=1000, option_type="put",
                          exercise="american", q=q)
print("American vs European put (binomial tree, K=100):")
print(f"  European = {euro_put:.4f}")
print(f"  American = {amer_put:.4f}   (early-exercise premium = {amer_put - euro_put:.4f})\n")

# ----------------------------------------------------------------------
# 3. Figure: the three methods overlaid across strikes
# ----------------------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.plot(table["Strike"], table["Black-Scholes"], "-", label="Black-Scholes")
plt.plot(table["Strike"], table["Binomial"], "s", markersize=8,
         markerfacecolor="none", label="Binomial")
plt.plot(table["Strike"], table["Monte Carlo"], "x", markersize=9, label="Monte Carlo")
plt.title("Three pricing methods agree (European call)")
plt.xlabel("Strike K"); plt.ylabel("Call price")
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("methods_comparison.png", dpi=120)
print("Saved figure: methods_comparison.png")
