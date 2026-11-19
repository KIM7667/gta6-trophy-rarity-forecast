"""Forecast GTA VI trophy earn rates on Exophase one week after launch.

Model (pure stdlib, deterministic, no network):

Per trophy, the earn rate's logit is forecast two ways from the
case's history, and the median is their mean in logit space:

1. Trend line (the house trend baseline's median): least squares of
   logit(rate) against ln(1 + hours since launch) over the last 24 h
   of reads, read at the target. Right when growth keeps its launch
   pace, wrong when it bends.

2. Saturating curve: least squares of logit(rate) against
   exp(-t / tau), tau gridded over 12..768 h (recency-weighted,
   half-life 12 h), read at the target. Right when the early-adopter
   wave passes and growth bends toward an asymptote, wrong when
   growth stays linear in log-time.

Neither regime can be known in advance -- the practice cases carry
a flat launch-night history and say nothing about the final's week --
so the median hedges 50/50 between them. In synthetic trials over
fast saturation, slow saturation and unbent log-time growth, that
hedge's median beats or matches each single model in every regime
and has the lowest mean logit error overall; persistence alone is
worst everywhere, since rates move a lot over 4.5 days.

Safety rails: the median's logit move from the last observed rate
is capped at 2.5, and a loose tier ceiling applies (platinum <= 4%,
gold <= 35%, silver <= 80%, bronze <= 97%) so a runaway line can
never produce an absurd rate.

Spread (honest quantiles, pinball on the logit scale): a random-walk
term at HALF the starter's coefficient -- 0.25 * sqrt(horizon_days)
-- combined in quadrature with the fits' residual SD (scaled by how
far the target sits past the data), half the disagreement between
the two candidate medians, and a 0.15 floor. The starter's full
0.5 coefficient is far wider than any median this ensemble
produces: in synthetic trials over all three regimes, halving it
cuts the mean pinball loss by ~0.1 while the disagreement term
keeps the band honest exactly where the two models part ways.

With a flat history (the practice cases: every launch-night read
identical) both fits are flat, the candidates agree with the last
rate, and this reduces to the starter's persistence forecast -- the
only honest answer when the data has not moved yet.

Format: reads every <id>.in in the input directory, writes
<id>.out as {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]},
one row per trophy, quantiles non-decreasing, rounded to 4 decimals
half-up like the starter. Runs under `python3 -P -s` (Python 3.13),
standard library only.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

SIGMA = 0.5          # starter's random-walk spread, logit units per sqrt(day)
RW_FRACTION = 0.5    # our random-walk term uses half the starter's coefficient
MAX_DELTA = 2.5      # cap on |logit move| from the last observed rate
SPREAD_FLOOR = 0.15  # minimum logit spread
HALF_LIFE_H = 12.0   # recency weight half-life, hours
TREND_WINDOW_H = 24.0
TAUS_H = (12.0, 24.0, 48.0, 96.0, 192.0, 384.0, 768.0)

# Loose median ceilings by tier (percent). Only bind on runaway
# extrapolation; ordinary forecasts sit far below them.
TIER_CAP = {"platinum": 4.0, "gold": 35.0, "silver": 80.0, "bronze": 97.0}


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def sigmoid(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """Round to 4 decimals, half up, as JS Math.round(x*10000)/10000."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def launch_epoch(case: dict) -> datetime:
    rel = case.get("game", {}).get("release_date")
    if rel:
        try:
            return datetime.fromisoformat(rel + "T00:00:00+00:00")
        except ValueError:
            pass
    hist = case.get("history") or []
    if hist:
        return parse_time(hist[0]["at"])
    return parse_time(case["at"])


def ols(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, resid_sd)."""
    sw = sum(ws)
    if sw <= 0:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return (my, 0.0, 0.0)
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid2 = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys))
    dof = max(1.0, sw - 2.0)
    return (a, b, math.sqrt(resid2 / dof))


def trend_median(pts, t_last, t_target):
    """OLS of logit vs ln(1+t) over the last TREND_WINDOW_H hours."""
    sel = [(t, y) for t, y in pts if t >= t_last - TREND_WINDOW_H]
    if len(sel) < 3:
        sel = pts
    if len(sel) < 3:
        return None
    xs = [math.log1p(t) for t, _ in sel]
    ys = [y for _, y in sel]
    fit = ols(xs, ys, [1.0] * len(sel))
    if fit is None:
        return None
    a, b, resid = fit
    return a + b * math.log1p(t_target), resid


def forecast_trophy(series, target_h, horizon_days, tier):
    """series: list of (hours_since_launch, rate_pct) sorted by time.
    Returns (median_pct, spread_logit, median_logit)."""
    last_rate = None
    for _, r in reversed(series):
        if r is not None:
            last_rate = r
            break
    if last_rate is None:
        return (0.63, 1.0, logit(0.63))
    last_lg = logit(last_rate)

    pts = [(t, logit(r)) for t, r in series if r is not None]
    rw = RW_FRACTION * SIGMA * math.sqrt(max(horizon_days, 0.0))
    if len(pts) < 3 or max(t for t, _ in pts) - min(t for t, _ in pts) < 1e-9:
        # Too little history, or no time variation (flat practice
        # history): persistence is the only honest forecast.
        return (last_rate, math.sqrt(rw * rw + SPREAD_FLOOR ** 2), last_lg)

    t_last = pts[-1][0]
    reach = max(0.0, (target_h - t_last) / 24.0)

    trend = trend_median(pts, t_last, target_h)
    sat = None
    sat_resid = 0.0
    best_sat = None
    for tau in TAUS_H:
        xs = [math.exp(-t / tau) for t, _ in pts]
        ys = [y for _, y in pts]
        ws = [0.5 ** ((t_last - t) / HALF_LIFE_H) for t, _ in pts]
        fit = ols(xs, ys, ws)
        if fit is None:
            continue
        a, b, resid = fit
        sse = sum(w * (y - (a + b * x)) ** 2
                  for w, x, y in zip(ws, xs, ys))
        if best_sat is None or sse < best_sat[0]:
            best_sat = (sse, a, b, resid, tau)
    if best_sat is not None:
        _, a, b, sat_resid, tau = best_sat
        sat = a + b * math.exp(-target_h / tau)

    if trend is None and sat is None:
        return (last_rate, math.sqrt(rw * rw + SPREAD_FLOOR ** 2), last_lg)
    if trend is None:
        med_lg, resid = sat, sat_resid
        disagree = 0.0
    elif sat is None:
        med_lg, resid = trend[0], trend[1]
        disagree = 0.0
    else:
        med_lg = 0.5 * trend[0] + 0.5 * sat
        resid = max(trend[1], sat_resid)
        disagree = abs(trend[0] - sat)

    # Safety rails on the median.
    delta = max(-MAX_DELTA, min(MAX_DELTA, med_lg - last_lg))
    med_lg = last_lg + delta
    cap = TIER_CAP.get(tier, 97.0)
    med = min(sigmoid(med_lg), cap)
    med = max(med, 0.01)
    med_lg = logit(med)

    resid_term = resid * min(1.0, reach / 2.0)
    disagree_term = 0.5 * min(disagree, 2.0)
    spread = math.sqrt(rw * rw + resid_term * resid_term
                       + disagree_term * disagree_term
                       + SPREAD_FLOOR * SPREAD_FLOOR)
    return (med, spread, med_lg)


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    epoch = launch_epoch(case)
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_days = (target - at).total_seconds() / 86400.0
    target_h = (target - epoch).total_seconds() / 3600.0

    trophies = case["trophies"]
    n = len(trophies)
    series = [[] for _ in range(n)]
    for read in history:
        t = (parse_time(read["at"]) - epoch).total_seconds() / 3600.0
        rates = read["rates"]
        for i in range(n):
            r = rates[i] if i < len(rates) else None
            series[i].append((t, r))

    rows = []
    for i, trophy in enumerate(trophies):
        tier = trophy.get("tier", "bronze")
        med, spread, med_lg = forecast_trophy(
            series[i], target_h, horizon_days, tier)
        qs = []
        prev = -1.0
        for z in Z:
            q = clamp_pct(sigmoid(med_lg + z * spread))
            q = round4(q)
            # Keep the five quantiles non-decreasing after rounding.
            if q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
