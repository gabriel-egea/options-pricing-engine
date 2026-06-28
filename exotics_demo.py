"""
exotics_demo.py
---------------
Prices and validates the exotic options:
  1. a vanilla call priced from the path engine matches Black-Scholes,
  2. geometric Asian: Monte Carlo matches its exact closed form (engine check),
  3. averaging lowers the price: geometric <= arithmetic Asian <= vanilla,
  4. barriers: up-and-out + up-and-in = vanilla, exactly (in-out parity),
  5. a plot of sample paths with the barrier level.

Run with:  python exotics_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt

import black_scholes as bs
from exotics import (simulate_paths, asian_option,
                     geometric_asian_closed_form, barrier_option)

S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.0
N_STEPS, N_PATHS, SEED = 100, 200_000, 5


def price_ci(disc, antithetic=True):
    """Mean price and 95% CI of discounted payoffs (pair-aware for antithetic)."""
    if antithetic:
        n = len(disc) // 2
        disc = 0.5 * (disc[:n] + disc[n:2 * n])
    m = disc.mean()
    se = disc.std(ddof=1) / np.sqrt(len(disc))
    return m, (m - 1.96 * se, m + 1.96 * se)


# ----------------------------------------------------------------------
# 1. Engine check: vanilla call from paths == Black-Scholes
# ----------------------------------------------------------------------
paths = simulate_paths(S, T, r, sigma, N_STEPS, N_PATHS, q, antithetic=True, seed=SEED)
van_disc = np.exp(-r * T) * np.maximum(paths[:, -1] - K, 0.0)
van_price, van_ci = price_ci(van_disc)
bs_call = bs.bs_price(S, K, T, r, sigma, "call", q)
print("1) Vanilla call (engine check):")
print(f"   Black-Scholes    = {bs_call:.4f}")
print(f"   from path engine = {van_price:.4f}   95% CI [{van_ci[0]:.4f}, {van_ci[1]:.4f}]")
print(f"   BS inside CI: {van_ci[0] <= bs_call <= van_ci[1]}\n")

# ----------------------------------------------------------------------
# 2. Geometric Asian: Monte Carlo vs exact closed form
# ----------------------------------------------------------------------
geo_mc, _, geo_ci = asian_option(S, K, T, r, sigma, N_STEPS, N_PATHS,
                                 "call", "geometric", q, seed=SEED)
geo_cf = geometric_asian_closed_form(S, K, T, r, sigma, N_STEPS, "call", q)
print("2) Geometric Asian call (validation):")
print(f"   closed form  = {geo_cf:.4f}")
print(f"   Monte Carlo  = {geo_mc:.4f}   95% CI [{geo_ci[0]:.4f}, {geo_ci[1]:.4f}]")
print(f"   closed form inside CI: {geo_ci[0] <= geo_cf <= geo_ci[1]}\n")

# ----------------------------------------------------------------------
# 3. Averaging lowers the price: geometric <= arithmetic <= vanilla
# ----------------------------------------------------------------------
ari_mc, _, _ = asian_option(S, K, T, r, sigma, N_STEPS, N_PATHS,
                            "call", "arithmetic", q, seed=SEED)
print("3) Averaging lowers the price:")
print(f"   geometric Asian  = {geo_mc:.4f}")
print(f"   arithmetic Asian = {ari_mc:.4f}")
print(f"   vanilla call     = {bs_call:.4f}")
print(f"   ordering geo <= arith <= vanilla: {geo_mc <= ari_mc <= bs_call}\n")

# ----------------------------------------------------------------------
# 4. Barrier in-out parity (exact, on identical paths)
# ----------------------------------------------------------------------
B, STEPS_B, PATHS_B = 130.0, 252, 100_000
pB = simulate_paths(S, T, r, sigma, STEPS_B, PATHS_B, q, antithetic=True, seed=SEED)
van_pay = np.exp(-r * T) * np.maximum(pB[:, -1] - K, 0.0)
breached = pB.max(axis=1) >= B
out_price, _ = price_ci(van_pay * (~breached))
in_price, _ = price_ci(van_pay * breached)
van_same, _ = price_ci(van_pay)
print(f"4) Up barrier B={B:.0f}, in-out parity (same paths):")
print(f"   up-and-out = {out_price:.4f}")
print(f"   up-and-in  = {in_price:.4f}")
print(f"   out + in   = {out_price + in_price:.4f}")
print(f"   vanilla    = {van_same:.4f}   (exact match: {np.isclose(out_price + in_price, van_same)})")
print(f"   knock-out cheaper than vanilla: {out_price < van_same}\n")

# ----------------------------------------------------------------------
# 5. Sample-paths plot with the barrier
# ----------------------------------------------------------------------
demo_paths = simulate_paths(S, T, r, sigma, n_steps=252, n_paths=40, seed=SEED)
t = np.linspace(0, T, demo_paths.shape[1])
plt.figure(figsize=(10, 6))
for path in demo_paths:
    knocked = path.max() >= B
    plt.plot(t, path, color=("tab:red" if knocked else "tab:blue"),
             alpha=0.6, linewidth=0.9)
plt.axhline(B, color="black", linestyle="--", linewidth=1.5, label=f"Barrier B = {B:.0f}")
plt.title("Sample paths - red paths breach the up barrier (knocked out)")
plt.xlabel("Time (years)"); plt.ylabel("Underlying price")
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("barrier_paths.png", dpi=120)
print("Saved figure: barrier_paths.png")
