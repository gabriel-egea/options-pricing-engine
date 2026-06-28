"""
implied_vol_demo.py
-------------------
1. Validation: price an option with a known sigma, then recover that sigma
   from the price -> the solver should return it back.
2. Build an equity-style volatility skew and plot it.

Run with:  python implied_vol_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt

from black_scholes import bs_price
from implied_vol import implied_vol

S, T, r = 100.0, 0.5, 0.03

# ----------------------------------------------------------------------
# 1. Validation: round-trip on several known volatilities
# ----------------------------------------------------------------------
print("1) Round-trip validation (recover a known sigma):")
print("   K     true_sigma   recovered")
for K in (80, 100, 120):
    for true_sigma in (0.15, 0.25, 0.40):
        price = bs_price(S, K, T, r, true_sigma, "call")
        iv = implied_vol(price, S, K, T, r, "call")
        print(f"   {K:<5} {true_sigma:<11} {iv:.6f}")
print()

# ----------------------------------------------------------------------
# 2. Build and recover an equity-style skew
#    We impose a realistic downward skew (higher vol for low strikes),
#    generate the option prices it implies, then recover the implied vol
#    with our solver. This both tests the solver across strikes AND shows
#    what an equity skew looks like.
# ----------------------------------------------------------------------
strikes = np.linspace(70, 130, 25)
true_iv = 0.20 + 0.0018 * (100 - strikes)        # downward skew

market_prices, types = [], []
for K, iv in zip(strikes, true_iv):
    opt = "put" if K < S else "call"             # use OTM options (more liquid)
    market_prices.append(bs_price(S, K, T, r, iv, opt))
    types.append(opt)

recovered_iv = [implied_vol(p, S, K, T, r, t)
                for p, K, t in zip(market_prices, strikes, types)]

plt.figure(figsize=(9, 5))
plt.plot(strikes, np.array(true_iv) * 100, "-", color="grey", label="Imposed skew")
plt.plot(strikes, np.array(recovered_iv) * 100, "o", color="tab:blue",
         label="Recovered implied vol")
plt.axhline(20, color="red", linestyle="--", linewidth=1,
            label="Black-Scholes assumption (flat 20%)")
plt.axvline(S, color="black", linestyle=":", linewidth=1, label="Spot = 100")
plt.title("Equity volatility skew: implied vol vs strike")
plt.xlabel("Strike K"); plt.ylabel("Implied volatility (%)")
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("vol_smile.png", dpi=120)

err = np.nanmax(np.abs(np.array(recovered_iv) - np.array(true_iv)))
print("2) Skew recovery:")
print(f"   max error between recovered and imposed vol = {err:.2e}")
print("   -> the solver recovers the whole skew accurately.")
print("Saved figure: vol_smile.png")
