# IMC Prosperity 4

This repository contains Python trading strategies, submission artifacts, and
research notes for the IMC Prosperity 4 challenge. The code is organized around
a modular strategy framework that can be bundled into the single-file
`solution.py` format required by the IMC exchange.

## Project Goals

- Keep product logic modular while still producing an upload-ready
  `solution.py`.
- Use only lightweight standard-library Python in submitted code.
- Keep logging disabled by default to avoid runtime and output-size issues.
- Preserve important submitted solutions and result artifacts for later review.
- Separate robust market-state logic from high-risk path-fitted experiments.

## Repository Layout

```text
.
|-- build.py
|-- datamodel.py
|-- solution.py
|-- solution_round1.py
|-- solution_round3_63371.py
|-- solution_round3_robust_hp_open9980_ve20_vsafe.py
|-- round4_solution_draft_upload.py
|-- round4_solution_draft_trailing_loose_upload.py
|-- src/
|   |-- base.py
|   |-- config.py
|   |-- trader.py
|   |-- products/
|   |   |-- emeralds.py
|   |   `-- tomatoes.py
|   `-- strategies/
|       |-- pair_trading.py
|       `-- seasonal.py
|-- docs/
|   `-- round3_path_replay_review.md
`-- data/
    |-- submission_517701/
    `-- submission_545991/
```

## Core Framework

The active modular strategy lives under `src/`.

- `src/trader.py` is the exchange entry point. It parses `traderData`, runs each
  product trader, and serializes persistent state back to compact JSON.
- `src/config.py` stores global settings such as `DEBUG`, product limits, and the
  master strategy mode.
- `src/base.py` provides shared order-book helpers, position-capacity checks,
  rolling statistics, insider-trade signal utilities, and a wash-trade guard.
- `src/products/` contains product-specific implementations.
- `src/strategies/` contains reusable strategy templates for pair trading,
  basket arbitrage, seasonal signals, and observation-driven signals.

## Active Product Strategies

### EMERALDS

`src/products/emeralds.py` implements a fixed fair-value market maker around
`10_000`.

The strategy:

- buys asks below fair value and sells bids above fair value;
- trades at fair value only when it helps flatten inventory;
- posts passive quotes inside common bot walls;
- adjusts quote placement based on inventory;
- removes any self-crossing buy/sell pairs before returning orders.

### TOMATOES

`src/products/tomatoes.py` implements an EMA-based market maker for a drifting
product.

The strategy:

- estimates fair value from a volume-weighted mid-price EMA;
- only takes liquidity when price is sufficiently displaced from the EMA;
- posts passive inventory-aware quotes;
- stores the EMA in `traderData` so it persists across ticks.

## Reusable Strategy Templates

The repository also contains strategy components that can be enabled when the
relevant products are available.

- `PairTrader` models a two-leg statistical arbitrage spread with rolling
  z-scores.
- `BasketArb` compares a basket product against a weighted synthetic basket.
- `SeasonalTrader` trades products with predictable timestamp behavior.
- `DerivativeSignalTrader` reacts to external observation changes, such as a
  signal product that leads a tradeable product.

These templates are intentionally generic and are not automatically active
unless wired into `src/trader.py`.

## Building A Submission

The IMC platform expects a single Python file. The modular code can be bundled
with:

```powershell
python build.py
```

`build.py` reads the files listed in its `SRC_FILES` array, strips internal
package imports, hoists imports to the top, and writes a self-contained
`solution.py`.

Use `solution.py` for upload when working from the modular framework. Historical
round-specific solution files are kept separately for reference and comparison.

## Historical Solutions And Notes

The repository preserves several standalone solutions and review notes:

- `solution_round1.py`: Round 1 competition submission.
- `solution_round3_63371.py`: high-PnL Round 3 path-replay candidate.
- `solution_round3_robust_hp_open9980_ve20_vsafe.py`: Round 3 non-replay
  candidate focused on observable market-state rules.
- `docs/round3_path_replay_review.md`: review of path-replay risk and why the
  robust candidate is safer from a generalization perspective.
- `round4_solution_draft_upload.py`: base Round 4 draft solution artifact.
- `round4_solution_draft_trailing_loose_upload.py`: Round 4 draft with a looser
  trailing drawdown guard.
- `ROUND4_DRAFTS_PR_NOTES.md`: notes comparing Round 4 draft variants.

## Submission Artifacts

Official downloaded result bundles are stored under `data/`:

- `data/submission_517701/`
- `data/submission_545991/`

Each folder contains the submitted `.py` file plus matching `.json` and `.log`
artifacts. Logs are generally ignored by `.gitignore`, but these result logs are
tracked intentionally because they document the exact submission outputs.

## Development Notes

- Keep `DEBUG = False` in competition submissions.
- Avoid adding heavy dependencies to submitted code.
- Keep position limits centralized in `src/config.py`.
- Run `python build.py` after changing files under `src/`.
- Treat path-replay or timestamp-fitted logic as high risk unless it is clearly
  separated from robust market-state logic.

## Git Branches

The cleaned repository keeps the main research branches from the work so far:

- `main`: current consolidated branch.
- `darra/prosperity4`: initial modular Prosperity 4 framework.
- `prosperity4-strategy`: Round 1 strategy branch.
- `round2-best-submission`: Round 2 best-submission artifact branch.
- `round3-path-replay-risk-review`: Round 3 replay-risk review branch.
- `round3-path-replay-risk-review-main`: matching Round 3 review branch.
- `round4-drafts-onto-darra-prosperity4`: Round 4 drafts on the modular base.
- `round4-solution-drafts-risk-review`: Round 4 draft-risk branch.
- `round4-solution-drafts-risk-review-clean`: branch containing the latest
  submission artifacts.
