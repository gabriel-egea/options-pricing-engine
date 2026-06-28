"""
monte_carlo_demo.py
-------------------
Demonstrates the Monte Carlo pricer:
  1. MC price vs Black-Scholes, with standard error and 95% CI,
  2. antithetic variates reduce the standard error,
  3. convergence study: the error decreases like 1/sqrt(M).

Run with:  python monte_carlo_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt

import black_scholes as bs
from monte_carlo import mc_price_european

S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.0
bs_call = bs.bs_price(S, K, T, r, sigma, "call", q)

# ----------------------------------------------------------------------
# 1. MC vs Black-Scholes
# ----------------------------------------------------------------------
price, se, ci = mc_price_european(S, K, T, r, sigma, M=200_000,
                                  option_type="call", q=q, seed=42)
print("European call:")
print(f"  Black-Scholes        = {bs_call:.4f}")
print(f"  Monte Carlo (M=200k) = {price:.4f}  (std error {se:.4f})")
print(f"  95% CI               = [{ci[0]:.4f}, {ci[1]:.4f}]")
print(f"  BS inside CI: {ci[0] <= bs_call <= ci[1]}\n")

# ----------------------------------------------------------------------
# 2. Antithetic variates reduce the standard error
# ----------------------------------------------------------------------
_, se_plain, _ = mc_price_european(S, K, T, r, sigma, M=200_000, q=q,
                                   antithetic=False, seed=1)
_, se_anti, _ = mc_price_european(S, K, T, r, sigma, M=200_000, q=q,
                                  antithetic=True, seed=1)
print("Standard error (same M):")
print(f"  plain        = {se_plain:.5f}")
print(f"  antithetic   = {se_anti:.5f}")
print(f"  reduction    = {100*(1 - se_anti/se_plain):.1f}%\n")

# ----------------------------------------------------------------------
# 3. Convergence study: error vs M
# ----------------------------------------------------------------------
Ms = np.array([100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000])
errors = []
for m in Ms:
    p, _, _ = mc_price_european(S, K, T, r, sigma, M=int(m), q=q,
                                antithetic=False, seed=7)
    errors.append(abs(p - bs_call))
errors = np.array(errors)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# left: price + 95% band converging to BS
prices, los, his = [], [], []
for m in Ms:
    p, s, c = mc_price_european(S, K, T, r, sigma, M=int(m), q=q, seed=7)
    prices.append(p); los.append(c[0]); his.append(c[1])
ax[0].plot(Ms, prices, "o-", label="Monte Carlo price")
ax[0].fill_between(Ms, los, his, alpha=0.2, label="95% confidence interval")
ax[0].axhline(bs_call, color="red", linestyle="--", label=f"Black-Scholes = {bs_call:.4f}")
ax[0].set_xscale("log")
ax[0].set_xlabel("Number of simulations M"); ax[0].set_ylabel("Call price")
ax[0].set_title("Monte Carlo converges to Black-Scholes")
ax[0].legend(); ax[0].grid(alpha=0.3)

# right: error vs M on log-log, with 1/sqrt(M) reference
ax[1].loglog(Ms, errors, "o-", label="|MC - BS| error")
ref = errors[0] * np.sqrt(Ms[0]) / np.sqrt(Ms)
ax[1].loglog(Ms, ref, "--", color="grey", label="1/sqrt(M) reference")
ax[1].set_xlabel("Number of simulations M"); ax[1].set_ylabel("Absolute error")
ax[1].set_title("Error decreases like 1/sqrt(M)")
ax[1].legend(); ax[1].grid(alpha=0.3, which="both")

fig.tight_layout()
fig.savefig("monte_carlo_convergence.png", dpi=120)
print("Saved figure: monte_carlo_convergence.png")
