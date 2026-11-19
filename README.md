# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/KIM7667/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/KIM7667/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every GTA VI PlayStation 5 trophy's Exophase earn rate at 2026-11-26T00:00:00Z, as five quantiles (5/25/50/75/95%) per trophy. Pure Python 3.13 standard library, deterministic, no network.

The model: per trophy, the earn rate's logit is forecast two ways from the case history and the median is their mean in logit space — (1) a least-squares trend line of logit(rate) against ln(1 + hours since launch) over the last 24 h of reads, and (2) a saturating curve, logit(rate) against exp(-t/tau) with tau gridded over 12–768 h. Neither regime can be known in advance (the practice cases carry a flat launch-night history), so the median hedges 50/50 between them. Safety rails cap the median's logit move from the last observed rate at 2.5 and apply loose tier ceilings (platinum ≤ 4%, gold ≤ 35%, silver ≤ 80%, bronze ≤ 97%). Spread is a random-walk term at half the starter's coefficient, combined with the fits' residual SD, half the disagreement between the two candidate medians, and a 0.15 floor.

Validated locally on the kit's practice cases (valid output, deterministic across runs) and on synthetic launch-week histories in three regimes (fast saturation, slow saturation, unbent log-time growth), where it beats the starter persistence baseline in two of three regimes and the house trend baseline in two of three.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=heefpym9&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
