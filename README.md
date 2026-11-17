[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/ChristaNavisa/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/ChristaNavisa/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

# GTA VI launch-day Twitch viewers — hour-by-hour forecast

Entry for [gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01):
forecast how many people watch Grand Theft Auto VI live on Twitch in each
hour of its release day, 2026-11-19 (UTC), after the first hour. `main.py`
(Python 3, standard library only, no network) reads one case file and writes
one forecast file with five quantiles (5%, 25%, 50%, 75%, 95%) for each of
hours 2–24.

## How it works

The program learns the launch-day shape from the case's own comparables —
past big game launches with a complete release-day series:

1. **Pick the analogues.** A launch is kept only if its hour-1 print is a
   real post-unlock count: no early-access launches, no weak analogues, no
   simultaneous-global unlocks (their hour 1 is pre-launch noise), no PC
   unlock at a different time than the consoles, and a complete 24-hour
   series with hour 1 ≥ 5,000 viewers. On the practice cases this keeps 14
   launches. If the strict rule keeps too few, it relaxes in steps, and
   with nothing usable it falls back to the house baseline shape.
2. **Fit the shape.** For each hour, the log ratio of that hour's viewers
   to the launch's hour-1 print is regressed on log(hour-1 print) across
   the kept launches, with the slope shrunk by 3/4 towards zero. Big
   launches run flatter than small ones — the whole world is already
   watching at hour 1 — while small ones climb steeply once the Americas
   unlock, and the regression captures that. The predicted ratio is
   clamped to the range the kept launches actually showed at that hour,
   so no forecast ever claims a shape no launch had.
3. **Spread honestly.** The spread is the residual standard deviation of
   that regression, floored at 0.20 in log space, and the quantiles are
   lognormal around the predicted median:
   `q = p · exp(m_h + z_tau · s_h)`, with `p` the case's hour-1 print.

The typical learned shape dips below hour 1 in hours 2–4 (the midnight
burst fades before the US unlocks), recovers through the morning, and peaks
around hours 19–21 at roughly 1–2.5× hour 1, flatter for bigger launches.

## Local validation

Scored leave-one-out on the 14 kept comparables (each launch forecast from
its own hour-1 print, fitted on the other 13, scored against its real
release-day hours with the competition's pinball loss on the log scale):

| forecaster | mean pinball loss | median |
| --- | --- | --- |
| house baseline | 0.3484 | 0.2522 |
| this entry | **0.1945** | **0.1444** |

44% below the house baseline on the same launches. The program also runs
in ~0.05 s per case (limit 8 s) and is fully deterministic — two runs
produce byte-identical output.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

It reads every `*.in` file in the input directory, in name order, and
writes one `*.out` file per case to the output directory. With no
`MAKEFUN_INPUT_DIR` / `MAKEFUN_OUTPUT_DIR` set it uses `/input` and
`/output`, as the grader does.

## License

MIT — see [LICENSE](LICENSE).
