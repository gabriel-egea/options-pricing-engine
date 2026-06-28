"""
binomial_demo.py
----------------
Demonstrates the binomial tree pricer:
  1. European prices converge to Black-Scholes as N grows,
  2. an American put is worth more than a European put (early-exercise premium),
  3. an American call (no dividend) equals the European call,
  4. a convergence plot saved to binomial_convergence.png.

Run with:  python binomial_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt

import black_scholes as bs
from binomial import binomial_price

S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.0

# ----------------------------------------------------------------------
# 1. European binomial vs Black-Scholes
# ----------------------------------------------------------------------
bs_call = bs.bs_price(S, K, T, r, sigma, "call", q)
bs_put = bs.bs_price(S, K, T, r, sigma, "put", q)

bin_call = binomial_price(S, K, T, r, sigma, N=1000, option_type="call", exercise="european", q=q)
bin_put = binomial_price(S, K, T, r, sigma, N=1000, option_type="put", exercise="european", q=q)

print("European call:")
print(f"  Black-Scholes   = {bs_call:.4f}")
print(f"  Binomial N=1000 = {bin_call:.4f}")
print("European put:")
print(f"  Black-Scholes   = {bs_put:.4f}")
print(f"  Binomial N=1000 = {bin_put:.4f}\n")

# ----------------------------------------------------------------------
# 2. American put vs European put  -> early-exercise premium
# ----------------------------------------------------------------------
amer_put = binomial_price(S, K, T, r, sigma, N=1000, option_type="put", exercise="american", q=q)
print("Put, early-exercise premium:")
print(f"  European = {bin_put:.4f}")
print(f"  American = {amer_put:.4f}")
print(f"  premium  = {amer_put - bin_put:.4f}\n")

# ----------------------------------------------------------------------
# 3. American call (no dividend) == European call
# ----------------------------------------------------------------------
amer_call = binomial_price(S, K, T, r, sigma, N=1000, option_type="call", exercise="american", q=q)
print("Call (no dividend), American vs European:")
print(f"  European = {bin_call:.4f}")
print(f"  American = {amer_call:.4f}")
print(f"  -> equal (early exercise never optimal): {np.isclose(amer_call, bin_call, atol=1e-3)}\n")

# ----------------------------------------------------------------------
# 4. Convergence plot
# ----------------------------------------------------------------------
Ns = range(1, 201)
prices = [binomial_price(S, K, T, r, sigma, N=n, option_type="call", exercise="european", q=q) for n in Ns]

plt.figure(figsize=(9, 5))
plt.plot(list(Ns), prices, label="Binomial price (European call)")
plt.axhline(bs_call, color="red", linestyle="--", label=f"Black-Scholes = {bs_call:.4f}")
plt.title("Binomial price converges to Black-Scholes as N grows")
plt.xlabel("Number of steps N")
plt.ylabel("Call price")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("binomial_convergence.png", dpi=120)
print("Saved figure: binomial_convergence.png")
