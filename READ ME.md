# options-pricing-engine

A personal project built to understand how options are priced in practice.
I implemented three different pricing methods from scratch, then checked they
give the same answer — which turned out to be the most instructive part.

---

## 1. Black-Scholes

The first method I studied is the Black-Scholes formula, which gives a closed-form price for European options. The key assumption is that the stock price follows a random walk with constant volatility — which is a simplification, but a useful starting point.

For a standard at-the-money call ($S = K = 100$, $T = 1$ year, $r = 5\%$, $\sigma = 20\%$):

```
Call = 10.45   Put = 5.57   Put-call parity: True
```

I also implemented the five Greeks — the sensitivities of the price to each input. A few things I found interesting:

- **Delta = 0.64**: if the stock goes up by $1, the call gains about $0.64. This is also the hedge ratio — to stay market-neutral, you'd hold 0.64 shares per option sold.
- **Gamma** peaks at-the-money: this is where delta changes fastest, so where the hedge needs to be adjusted most often.
- **Vega** also peaks at-the-money and grows with maturity: longer options are more sensitive to changes in volatility.
- **Theta = −$0.018/day**: the option loses a small amount of value every day just from time passing.

The theta/gamma relationship stood out to me: being long gamma (profiting from big moves) always comes with negative theta (paying a daily cost). You can't have one without the other — it's baked into the Black-Scholes equation.

One subtlety I learned: N(d₂) is the risk-neutral probability that the option expires in the money, while N(d₁) — which equals the delta — is not. The two are often confused but the gap between them grows with volatility.

![Greeks vs spot](greeks_vs_spot.png)
![Greeks by maturity](greeks_by_maturity.png)

---

## 2. Binomial tree

The binomial tree (Cox-Ross-Rubinstein) prices options differently: it splits time into small steps, and at each step the stock can only go up or down by a fixed factor. Working backwards from the known payoffs at maturity gives the price today.

The main reason to use a tree instead of the formula is **American options**: at every node you can check whether early exercise is better than holding on, and take the maximum. Black-Scholes has no way to do this.

```
European put = 5.57
American put = 6.09   (early-exercise premium = +0.52)
American call = European call  (early exercise never optimal without dividends)
```

The American put is worth more because sometimes it's better to exercise immediately and collect the intrinsic value rather than wait. For calls without dividends, this is never the case — you'd always be giving up remaining time value.

As the number of steps grows, the tree price converges to Black-Scholes. The oscillation at low N is because the strike sometimes falls on a node and sometimes between two nodes, which alternately over- and underestimates the price.

![Binomial convergence](binomial_convergence.png)

---

## 3. Monte Carlo

Instead of a formula or a tree, Monte Carlo simulates thousands of random stock paths and averages the payoff. For a European option the terminal price has an exact distribution, so I can skip the full path and just draw the final values directly.

Two things I paid attention to:

**Antithetic variates**: for each random draw $Z$ I also use $-Z$. Since the two payoffs move in opposite directions, their average is less noisy. In practice this cut the standard error by ~29% at equal computation cost.

**Correct standard error**: with antithetic pairs, the independent units are the pair averages, not the individual draws. Using individual draws gives an overconfident interval — a mistake I initially made and then corrected.

```
MC price (M = 200k) = 10.48    95% CI [10.43, 10.52]
Black-Scholes       = 10.45  →  inside the CI ✓
Antithetic SE reduction: 29%
```

The error decreases like $1/\sqrt{M}$: to halve the error, you need four times as many simulations. This is Monte Carlo's main limitation compared to the other two methods.

![Monte Carlo convergence](monte_carlo_convergence.png)

---

## 4. Exotic options

Some options depend on the whole price path, not just the final value. For these, there is generally no formula and the tree becomes impractical — Monte Carlo with full path simulation is the natural approach.

**Asian options** pay on the average price over the life of the option, not the final price. Because the average is smoother than the terminal value, the option is cheaper. There is no closed form for the arithmetic average, but the geometric average has one — I used it to check that my path simulation was correct before pricing the harder case.

```
Geometric Asian  =  5.60   (closed form = 5.59 ✓)
Arithmetic Asian =  5.81
Vanilla call     = 10.45

Ordering: geometric ≤ arithmetic ≤ vanilla ✓
```

**Barrier options** knock out (disappear) or knock in (activate) if the stock crosses a level. The most useful identity is in-out parity: knock-in + knock-out = vanilla, because either the barrier is crossed or it isn't. Testing this on the same paths removes simulation noise and gives an exact check:

```
Up-and-out (B = 130) = 3.57
Up-and-in  (B = 130) = 6.91
Sum = 10.48  =  vanilla from same paths ✓
```

![Sample paths with barrier](barrier_paths.png)

One limitation: I check the barrier only at the simulation steps, so a path that crosses and comes back between two steps goes undetected. More steps reduce this error.

---

## 5. Implied volatility

So far I assumed volatility was known. In practice, traders observe option prices in the market and work backwards to find the volatility that the market is "implying". This is the implied volatility.

Since Black-Scholes price is strictly increasing in $\sigma$, there is always a unique solution. I found it with Newton-Raphson (which uses vega as the derivative) and added a Brent bisection fallback for deep out-of-the-money options where vega is close to zero and Newton can fail.

Validation: I priced options with known volatilities, then recovered those volatilities from the prices. The error was below $5 \times 10^{-10}$ across all test cases.

The interesting part is what the implied vol looks like across strikes:

![Volatility skew](vol_smile.png)

If Black-Scholes were correct, the implied vol would be flat (the red dashed line). It isn't — lower strikes carry higher implied vol. This is the equity skew: the market charges more for crash protection (low-strike puts) than the model would suggest, reflecting fear of large downward moves. This tells you that real stock returns have fatter left tails than a Gaussian model assumes.

`market_smile.py` reproduces this plot using live option data from `yfinance`.

---

## 6. Cross-validation

`compare_methods.py` prices the same call across five strikes with all three methods at once:

```
Strike  Black-Scholes  Binomial  Monte Carlo
    80         24.589    24.589       24.581
    90         16.699    16.700       16.686
   100         10.451    10.449       10.431
   110          6.040     6.042        6.022
   120          3.247     3.248        3.237
```

The largest gap is 0.02 — Monte Carlo noise that shrinks with more paths. Three independent implementations agreeing is the best check I could do that the code is correct.

![Three methods agree](methods_comparison.png)

---

## How to run

```bash
pip install -r requirements.txt

python demo.py               # Black-Scholes prices, Greeks, parity
python greeks_plots.py       # Greek curves
python binomial_demo.py      # convergence, American options
python monte_carlo_demo.py   # Monte Carlo convergence
python exotics_demo.py       # Asian & barrier options
python implied_vol_demo.py   # implied vol solver + skew
python compare_methods.py    # all three methods side by side
```
