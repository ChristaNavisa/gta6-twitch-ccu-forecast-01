#!/usr/bin/env python3
"""Forecast GTA VI's launch-day Twitch viewers, hour by hour.

Reads one case file per run (schema makefun/twitch-ccu-case/v1) and writes
one forecast file with five quantiles (5/25/50/75/95%) for each of hours 2-24.

Model: learn the launch-day shape from the case's own comparables. For every
past big launch with a complete release-day series we compute, per hour,
the log ratio of that hour's viewers to the launch's hour-1 print. Launches
whose hour 1 is not a real post-unlock print are left out: early-access
launches, weak analogues, the target category's own pre-release history,
and launches whose unlock was one simultaneous global moment (their hour 1
is pre-launch noise) or a PC unlock at a different time than the consoles.

For each hour we then regress the log ratio on log(hour-1 print) across the
kept launches, with the slope shrunk by 0.90 towards zero, and centre the
prediction on the median ratio of the kept launches (not their mean, so one
breakout launch cannot drag the centre), then clamp the predicted ratio to
the range the kept launches actually show at that hour.
Big launches run flatter than small ones (the whole world is already
watching at hour 1), small ones climb steeply once the Americas unlock, and
the regression captures that without ever predicting a shape no launch
showed. The spread is the residual standard deviation of that regression,
floored at 0.20 in log space and widened by 1.15, because the residuals
are measured in-sample and so understate the error on a new launch; the
widening keeps the 5%-95% band honest about how much launch days differ
from each other.

Quantiles are lognormal around the predicted median:
    q_tau = p * exp(m_h + z_tau * s_h)
with p the case's hour-1 print. No network, no randomness, standard library
only; the same input always produces the same bytes.
"""

import json
import math
import os
import re
import statistics
import sys
from datetime import datetime, timedelta

# Standard normal quantiles at the five forecast levels.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# Slope shrinkage for the size regression (0 = plain median profile,
# 1 = full least-squares slope).
SHRINK = 0.90
# Floor on the log-space spread, so no hour is forecast overconfidently.
SPREAD_FLOOR = 0.20
# The residuals are measured in-sample, so they understate the error on a
# new launch; widen the spread by this factor to compensate.
SPREAD_MULT = 1.15
# A launch needs at least this many viewers in hour 1 for the print to be
# a real post-unlock count rather than pre-launch noise.
MIN_HOUR1 = 5000
# How many kept launches the full rule must yield before we relax it.
MIN_KEPT_STRICT = 8
MIN_KEPT_RELAXED = 5

# Phrases in a comparable's unlock note that mark a launch whose hour 1
# does not measure the same thing as a rolling-midnight console unlock.
BAD_PHRASES = (
    "single global",
    "simultaneous across",
    "simultaneous midnight",
    "global unlock",
    "pc unlock later",
    "unlock later",
    "steam global",
    "beta/early",
)

# Tiers that are not usable analogues for this launch.
BAD_TIERS = ("early-access-contaminated", "weak", "target-history")


def _tier(comp):
    match = re.search(r"Tier: ([a-z\-]+)", comp.get("unlock", ""))
    return match.group(1) if match else None


def _day_series(comp):
    """The comparable's 24 release-day hourly viewer counts, or None."""
    try:
        base = datetime.strptime(comp["day_start"], "%Y-%m-%dT%H:%M:%SZ")
    except (KeyError, ValueError):
        return None
    by_hour = {}
    for row in comp.get("hourly") or []:
        try:
            by_hour[row["from"]] = row["viewers"]
        except (KeyError, TypeError):
            continue
    series = []
    for h in range(24):
        stamp = (base + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ")
        value = by_hour.get(stamp)
        if value is None or value <= 0:
            return None
        series.append(value)
    return series


def _keep(comp, series, use_tier, use_phrases):
    if use_tier and _tier(comp) in BAD_TIERS:
        return False
    if use_phrases:
        text = comp.get("unlock", "").lower()
        if any(phrase in text for phrase in BAD_PHRASES):
            return False
    if series is None or series[0] < MIN_HOUR1:
        return False
    return True


def _select(comparables):
    """Pick the analogue launches, relaxing the rule if it keeps too few."""
    parsed = []
    for comp in sorted(comparables, key=lambda c: c.get("name", "")):
        series = _day_series(comp)
        if series is not None:
            parsed.append((comp, series))
    for use_tier, use_phrases, minimum in (
        (True, True, MIN_KEPT_STRICT),
        (True, False, MIN_KEPT_STRICT),
        (False, False, MIN_KEPT_RELAXED),
    ):
        kept = [s for comp, s in parsed if _keep(comp, s, use_tier, use_phrases)]
        if len(kept) >= minimum:
            return kept
    return []


def _fit(series_list, hour1):
    """Per hour, the predicted median log ratio and spread.

    Returns a dict hour -> (median_log_ratio, spread) for hours 2..24.
    """
    log_h1 = math.log(max(hour1, 1))
    xs = [math.log(s[0]) for s in series_list]
    x_mean = statistics.mean(xs)
    x_target = log_h1 - x_mean
    denom = sum((x - x_mean) ** 2 for x in xs)
    out = {}
    for h in range(2, 25):
        ratios = [math.log(s[h - 1] / s[0]) for s in series_list]
        y_mean = statistics.mean(ratios)
        if denom > 0:
            slope = sum((x - x_mean) * (y - y_mean)
                        for x, y in zip(xs, ratios)) / denom
        else:
            slope = 0.0
        slope *= SHRINK
        # Centre on the median ratio, not the mean: one breakout launch
        # (a small game that explodes) must not drag the whole centre up.
        median = statistics.median(ratios) + slope * x_target
        # Never predict a shape no kept launch showed at that hour.
        median = max(min(median, max(ratios)), min(ratios))
        residuals = [y - (y_mean + slope * (x - x_mean))
                     for x, y in zip(xs, ratios)]
        spread = statistics.stdev(residuals) if len(residuals) > 1 else 0.3
        out[h] = (median, max(spread, SPREAD_FLOOR) * SPREAD_MULT)
    return out


def _house_fallback(hour1):
    """The house baseline shape: flat median, spread widening with time."""
    return {h: (0.0, 0.15 * math.sqrt(h - 1)) for h in range(2, 25)}


def forecast(case):
    """Build the forecast object for one case dict."""
    known = case.get("known") or []
    hour1 = 1
    if known:
        try:
            hour1 = max(int(known[0].get("viewers") or 1), 1)
        except (TypeError, ValueError):
            hour1 = 1
    kept = _select(case.get("comparables") or [])
    if kept:
        profile = _fit(kept, hour1)
    else:
        profile = _house_fallback(hour1)
    rows = []
    for h in range(2, 25):
        median, spread = profile[h]
        quantiles = []
        for z in Z:
            value = hour1 * math.exp(median + z * spread)
            value = round(value, 2)
            value = max(1.0, min(value, 1_000_000_000.0))
            quantiles.append(value)
        # Rounding must not break the non-decreasing order.
        for i in range(1, len(quantiles)):
            if quantiles[i] < quantiles[i - 1]:
                quantiles[i] = quantiles[i - 1]
        rows.append({"hour": h, "q": quantiles})
    return {"hours": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    if not os.path.isdir(in_dir):
        raise SystemExit("no input directory: %s" % in_dir)
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        try:
            with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
                case = json.load(fh)
            result = forecast(case)
        except Exception as exc:  # one bad file must not kill the rest
            print("skipping %s: %s" % (name, exc), file=sys.stderr)
            continue
        out_name = name[:-3] + ".out"
        with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as fh:
            json.dump(result, fh, separators=(",", ":"))
            fh.write("\n")


if __name__ == "__main__":
    main()
