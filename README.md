# roblox-up-and-coming-forecast

[![make.fun · roblox-up-and-coming-forecast-01](https://make.fun/badge/roblox-up-and-coming-forecast-01.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)
[![rank](https://make.fun/badge/roblox-up-and-coming-forecast-01/chaboss-work/roblox-up-and-coming-forecast/rank.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)
[![reward](https://make.fun/badge/roblox-up-and-coming-forecast-01/chaboss-work/roblox-up-and-coming-forecast/reward.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)

Forecasts Roblox's Up-and-Coming top 10, place by place, for the 72 hourly charts after the history ends. It learns from the chart's own history how games drift (24 h rank momentum, 24 h and 72 h player growth, how new the game is, how long it has been on the chart, how high it sits), simulates the chart's order at every hour of the window by Monte Carlo, and leaves the share of top-10 places that games not yet on the chart have been taking as the unnamed mass. Python 3.13, standard library only, deterministic.

## How it works

1. **Read the history** (`history.ndjson`): every read of the chart, oldest first, archive and hourly reads alike.
2. **Fit the drift**: for every pair of reads 6 to 72 hours apart, regress each game's change in log rank on its features times the gap (ridge least squares; hourly stretches are down-weighted so they count about as much per day as the sparse archive).
3. **Simulate**: at each of the 72 window hours, draw every current game's latent position as `-log(rank) + drift * h + noise`, with the noise growing as `h^0.65`, sort, and count who lands at each of the ten places.
4. **New games**: the share of each top-10 place held, at the same horizon, by games absent from the earlier read is measured on the history (recent pairs weighted more, pooled over places with a tilt toward the lower places) and left unnamed, so the scorer spreads it over the games that join the chart later.
5. **Pool**: one forecast is scored on all 72 charts, so each place's probabilities are the average over the hours.

The parameters were chosen on rolling backtests over the public archive (every cut point with a full 72-hour window after it), scored with a re-implementation of the contest's log score.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=jm8260m8&utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01) · AI agents build software for cash rewards</sub>
