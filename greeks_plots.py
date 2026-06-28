"""
greeks_plots.py
---------------
Visualise the Black-Scholes Greeks. Two figures are produced:

  1. greeks_vs_spot.png   : delta, gamma, vega, theta as functions of the
                            underlying price S (the four core sensitivities).
  2. greeks_by_maturity.png : how gamma and vega change with time to
                              maturity T -- gamma sharpens at-the-money as
                              expiry approaches, while vega grows with T.

Run with:  python greeks_plots.py
"""

import numpy as np
import matplotlib.pyplot as plt

import black_scholes as bs

# Fixed contract parameters (we vary S, and later T)
K, r, sigma, q = 100.0, 0.05, 0.20, 0.0
spots = np.linspace(40, 160, 300)


# ----------------------------------------------------------------------
# Figure 1 -- the four core Greeks vs spot
# ----------------------------------------------------------------------
T = 1.0
delta_c = [bs.delta(s, K, T, r, sigma, "call", q) for s in spots]
delta_p = [bs.delta(s, K, T, r, sigma, "put",  q) for s in spots]
gamma_  = [bs.gamma(s, K, T, r, sigma, q)          for s in spots]
vega_   = [bs.vega(s, K, T, r, sigma, q)           for s in spots]
theta_c = [bs.theta(s, K, T, r, sigma, "call", q)  for s in spots]

fig, ax = plt.subplots(2, 2, figsize=(11, 8))

ax[0, 0].plot(spots, delta_c, label="Call")
ax[0, 0].plot(spots, delta_p, label="Put")
ax[0, 0].set_title("Delta  (sensitivity to spot)")
ax[0, 0].legend()

ax[0, 1].plot(spots, gamma_, color="tab:green")
ax[0, 1].set_title("Gamma  (peaks at-the-money)")

ax[1, 0].plot(spots, vega_, color="tab:red")
ax[1, 0].set_title("Vega  (sensitivity to volatility)")

ax[1, 1].plot(spots, theta_c, color="tab:purple")
ax[1, 1].set_title("Theta of a call, per year  (time decay)")

for a in ax.flat:
    a.axvline(K, color="grey", linestyle="--", linewidth=1)
    a.set_xlabel("Underlying price S")
    a.grid(alpha=0.3)

fig.suptitle("Black-Scholes Greeks vs underlying price  (K=100, T=1, r=5%, sigma=20%)",
             fontsize=12)
fig.tight_layout()
fig.savefig("greeks_vs_spot.png", dpi=120)
print("Saved figure: greeks_vs_spot.png")


# ----------------------------------------------------------------------
# Figure 2 -- effect of maturity on gamma and vega
# ----------------------------------------------------------------------
maturities = [0.1, 0.25, 0.5, 1.0, 2.0]

fig2, ax2 = plt.subplots(1, 2, figsize=(12, 5))

for Tm in maturities:
    g = [bs.gamma(s, K, Tm, r, sigma, q) for s in spots]
    ax2[0].plot(spots, g, label=f"T = {Tm} yr")
ax2[0].set_title("Gamma sharpens at-the-money as expiry approaches")
ax2[0].set_xlabel("Underlying price S"); ax2[0].set_ylabel("Gamma")
ax2[0].axvline(K, color="grey", linestyle="--", linewidth=1)
ax2[0].legend(); ax2[0].grid(alpha=0.3)

for Tm in maturities:
    v = [bs.vega(s, K, Tm, r, sigma, q) for s in spots]
    ax2[1].plot(spots, v, label=f"T = {Tm} yr")
ax2[1].set_title("Vega grows with time to maturity")
ax2[1].set_xlabel("Underlying price S"); ax2[1].set_ylabel("Vega")
ax2[1].axvline(K, color="grey", linestyle="--", linewidth=1)
ax2[1].legend(); ax2[1].grid(alpha=0.3)

fig2.tight_layout()
fig2.savefig("greeks_by_maturity.png", dpi=120)
print("Saved figure: greeks_by_maturity.png")
