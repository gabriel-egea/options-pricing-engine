# options-pricing-engine

European & exotic option pricer: Black-Scholes, binomial tree, Monte Carlo, Greeks.

A from-scratch implementation of option pricing methods, built to make the
underlying mathematics explicit rather than hiding it behind a library call.

---

## Status / roadmap

- [x] **Black-Scholes** closed-form pricing of European calls & puts (with dividend yield)
- [x] **Greeks**: delta, gamma, vega, theta, rho (analytical)
- [ ] Binomial tree pricing (European & American)
- [ ] Monte Carlo pricing + convergence study
- [ ] Exotic options (Asian, barrier) via Monte Carlo
- [ ] Implied volatility solver

This README is updated as each block is added.

---

## The model

Under Black-Scholes-Merton, the underlying price $S$ follows a geometric
Brownian motion:

$$dS = (r - q)\,S\,dt + \sigma\,S\,dW$$

The price of a European **call** and **put** is:

$$C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)$$
$$P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1)$$

with

$$d_1 = \frac{\ln(S/K) + (r - q + \tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}$$

where $N(\cdot)$ is the standard normal CDF, $r$ the risk-free rate, $q$ the
dividend yield, $\sigma$ the volatility and $T$ the time to maturity.

The **Greeks** measure how the price reacts to each input: delta to the spot,
gamma to delta, vega to volatility, theta to time, rho to the rate.

---

## Project structure

```
options-pricing-engine/
├── black_scholes.py   # core pricing + Greeks
├── demo.py            # runnable example (prices, Greeks, parity check, plot)
├── requirements.txt   # dependencies
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
import black_scholes as bs

price = bs.bs_price(S=100, K=100, T=1, r=0.05, sigma=0.20, option_type="call")
print(price)            # 10.4506

g = bs.greeks(S=100, K=100, T=1, r=0.05, sigma=0.20, option_type="call")
print(g["delta"])       # 0.6368
```

Run the full demonstration:

```bash
python demo.py
```

Example output:

```
Call price : 10.4506
Put  price : 5.5735

Greeks (call):
  delta = 0.6368
  gamma = 0.0188
  vega  = 37.5240
  theta = -6.4140
  rho   = 53.2325

Put-call parity check: match: True
```

![Option value vs underlying price](bs_price_vs_spot.png)

---

## Notes

- Greeks are returned in their raw (per-unit) form. Common conventions:
  vega per +1% of vol = `vega / 100`, theta per day = `theta / 365`.
- Put-call parity ( $C - P = S e^{-qT} - K e^{-rT}$ ) is used in `demo.py`
  as an automatic correctness check.
