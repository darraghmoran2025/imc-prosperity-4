# Round 3 Path Replay Review

This PR intentionally includes two Round 3 candidates with very different risk
profiles:

- `solution_round3_63371.py`: the high-PnL path-replay candidate.
- `solution_round3_robust_hp_open9980_ve20_vsafe.py`: the non path-replay
  candidate rebuilt from observable market-state rules.

The purpose is not to present both as equally valid. The 63k candidate is useful
as a benchmark for the known IMC backtester path, but it carries substantial
generalization and competition-risk concerns. The non-replay candidate has lower
reported PnL, but the edge is more defensible because it does not depend on a
precomputed timestamp trajectory.

## Result Summary

Local IMC-style backtester, using the provided Round 3 data and
`--match-trades none`:

| Candidate | Day 0 | Day 1 | Day 2 | Total | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| General baseline, no replay | 2,813 | 203 | 6,087 | 9,103 | Older no-replay baseline |
| Robust non-replay candidate | 3,618 | 918 | 6,566 | 11,102 | Current recommended robust candidate |
| Safe/path replay candidate | ~2,813 | ~203 | ~57,001 | ~60,017 | Same known path, guarded replay |
| Uploaded IMC result, replay family | n/a | n/a | n/a | ~63,371 | Product-level total from result logs |

The exact replay total differs depending on runner and fill model, but the
important point is stable: most of the extra profit comes from day-2 timestamp
target schedules, not from a reusable pricing signal.

## What Path Replay Is

Path replay means the strategy recognizes a known opening state or early price
fingerprint and then follows a hardcoded target-position schedule by timestamp.
In this Round 3 work, that schedule was fitted to the known day-2 trajectory.

Examples of replay behavior:

- Activating only when `HYDROGEL_PACK` opens at the known day-2 mid.
- Activating only when `VELVETFRUIT_EXTRACT` opens at the known day-2 mid.
- Using dictionaries of target positions keyed by timestamps such as `0`,
  `500`, `600`, `700`, etc.
- Returning early from the normal strategy while the overlay is active, so the
  schedule overrides the model-based logic.

This can produce very high PnL when the evaluation path is the same path used to
fit the schedule. It is fragile when the path is unseen.

## Why The 63k Version Is Risky

### 1. It Is Not Discovering A Reusable Edge

The 63k version does not primarily earn from a stable relationship such as:

- mean reversion,
- volatility mispricing,
- option/underlying fair value,
- delta hedging,
- spread capture,
- or a persistent regime feature.

Instead, it mostly earns by replaying positions that are known to be profitable
on one path. That makes the backtest result more like memorizing an answer key
than estimating a trading rule.

### 2. It Can Fail On An Unseen Path

If the final evaluation path differs, timestamp replay can trade in exactly the
wrong direction. Circuit breakers can reduce the damage, but they only react
after prices diverge from the expected path. They do not predict the new path.

This means the failure mode is asymmetric:

- On the known path: very high PnL.
- On a nearby but different path: possible early losses before disabling.
- On an unseen path that passes the opening fingerprint: potentially large
  wrong-way inventory.

### 3. Opening Fingerprints Are Not Enough

The replay version uses opening mids and checkpoint tolerances as a guard. That
helps avoid activating on obviously different days, but it does not prove the
future trajectory is the same.

Two paths can share the same open and then diverge materially. If that happens,
the replay schedule can accumulate positions based on future movement that no
longer exists.

### 4. It May Be Against The Spirit Of The Competition

Using historical data to calibrate parameters is normal. Encoding a timestamp
schedule fitted to a known hidden/evaluation path is different. Prior IMC
Prosperity writeups suggest teams have discovered and used hardcoding-style
exploits before, and IMC has responded by tightening behavior in later rounds.

Even if a path-replay submission runs successfully, it may be considered outside
the intended spirit of the challenge. That creates non-technical risk:

- disqualification risk,
- leaderboard result risk,
- or simply building confidence around an edge that does not transfer.

## Specific Concern: Overlay Control Flow

One concrete bug was found in the Hydrogel overlay logic during review:

- VE/VEV overlays returned unconditionally while the overlay flag was active.
- HP only returned early when a timestamp target existed.
- Between scheduled timestamps, HP could fall through into the trend logic.
- That meant the normal trend strategy could trade against the overlay's
  intended position.

This was fixed in the replay family by ensuring overlay activation owns the
decision path consistently. However, fixing this bug does not remove the
underlying path-replay risk.

## Robust Non-Replay Candidate

The non-replay candidate is intentionally less ambitious. It keeps only
observable rules that can plausibly transfer:

- No timestamp target-position dictionaries are active.
- No HP/VE/VEV hidden path overlays are active.
- Hydrogel skips trading when the day opens below `9980`.
- Velvetfruit Extract passive market-making size is increased from `12` to `20`.
- Voucher trading is restricted to the strikes that were least fragile in the
  three-day review: `VEV_5000` and `VEV_5200`.

This produces lower PnL than replay, but the logic is based on market state
available at runtime rather than knowing the future path.

## Remaining Overfit Risk In The Robust Version

The robust candidate is not free from overfit. The Hydrogel open filter
(`MIN_OPEN_TO_TRADE = 9980`) was selected after reviewing only three days. That
is still a small sample.

The difference is that this is a regime filter, not a replay path:

- It uses only timestamp-0 information.
- It does not encode future timestamps.
- It does not force position targets from a known profitable trajectory.
- If the market opens below the threshold, it simply avoids a historically bad
  Hydrogel regime rather than taking a precomputed side.

This is a defensible compromise, but it should still be tested on any additional
available paths before final submission.

## Recommendation

Use `solution_round3_63371.py` only as a known-path benchmark. It is valuable for
understanding how much PnL was available on the known day-2 path, but it should
not be treated as a reliable competition submission if the final path is unseen.

Use `solution_round3_robust_hp_open9980_ve20_vsafe.py` as the safer competitive
candidate unless we have strong evidence that the final IMC evaluator reuses the
same exact path.

The practical decision is:

- If optimizing only for the known local/IMC backtester path: replay wins.
- If optimizing for unseen evaluation and rule/spirit robustness: non-replay is
  the better candidate.

