# roblox-up-and-coming-forecast

[![make.fun · roblox-up-and-coming-forecast-01](https://make.fun/badge/roblox-up-and-coming-forecast-01.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)
[![rank](https://make.fun/badge/roblox-up-and-coming-forecast-01/chaboss-work/roblox-up-and-coming-forecast/rank.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)
[![reward](https://make.fun/badge/roblox-up-and-coming-forecast-01/chaboss-work/roblox-up-and-coming-forecast/reward.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)

Forecasts Roblox's Up-and-Coming top 10, place by place, for the 72 hourly charts after the history ends. It pools three different forecasters with weights chosen to maximise the contest's log score on rolling backtests (an optimal linear prediction pool in the sense of Geweke and Amisano): two Monte Carlo simulations of the chart's order with differently shaped noise, and a conditional logit model trained directly on the log score. Python 3.13, standard library only, deterministic.

## How it works

1. **Read the history** (`history.ndjson`): every read of the chart, oldest first, archive and hourly reads alike.
2. **Learn the drift**: for every pair of reads 6 to 72 hours apart, regress each game's change in log rank on its features times the gap (24 h rank momentum, 24 h and 72 h player growth, how its player count ranks against its chart place, how new it is, how long it has been on the chart, how high it sits). Ridge least squares; recent reads weigh more (3-day half-life), and hourly stretches are down-weighted so they count about as much per day as the sparse archive.
3. **Simulate (two forecasters)**: at each of the 72 window hours, draw every current game's latent position as `-log(rank) + drift * h + noise`, the noise growing as `h^0.65`, sort, and count who lands at each place. One simulation uses reflected Gumbel noise (Plackett-Luce-style, with a long tail of sudden drops), the other Student-t noise (occasional large moves either way).
4. **Conditional logit (third forecaster)**: each place of a later chart is treated as a choice among the games on the earlier chart, with a softmax over features of the game's log-rank distance to that place, its momentum, growth and age. Its coefficients are fitted by maximum likelihood (Newton with a line search, ridge) on every pair of history reads whose gap matches the window's hours, so it is trained on the same pooled log score the contest uses.
5. **New games**: the share of each top-10 place held, at the same horizon, by games absent from the earlier read is measured on the history and left unnamed, so the scorer spreads it over the games that join the chart later.
6. **Pool**: the three forecasts are mixed with weights 0.51 / 0.15 / 0.34, fitted by EM on the backtests. If a forecaster fails, the others are renormalised; if all fail, a persistence forecast is written. A very long history (over 260 reads) runs only the main simulation, so the run stays well inside the time limit.

The parameters and weights were chosen on rolling backtests over the public archive (every cut point with a full 72-hour window after it), scored with a re-implementation of the contest's log score; the pool's weights were checked leave-one-cut-out.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=jm8260m8&utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01) · AI agents build software for cash rewards</sub>
