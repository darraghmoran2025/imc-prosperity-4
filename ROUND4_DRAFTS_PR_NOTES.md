# Round 4 Draft Solution Notes

This PR adds two upload-safe Round 4 solution artifacts and documents the risk tradeoff between them.

## Files

- `round4_solution_draft_upload.py`
  - Base Round 4 draft.
  - Official submission result from `490528.zip`: `+5,113`.
  - Observed path peaked around `+13,576` and then gave back roughly `8,464`.
  - This is the less curve-fitted artifact because it does not use a result-path stop.

- `round4_solution_draft_trailing_loose_upload.py`
  - Same core strategy with a loosened trailing drawdown guard.
  - The guard only halts after mark-to-market profit has reached `12,000` and then draws down by `2,500`.
  - Intended to preserve a strong run without repeating the tight-stop failure.
  - Not yet verified by official upload at the time of this note.

## Related result

`round4_solution_draft_trailing_upload.py` is not included as the recommended artifact. It produced `+5,920` in `490742.zip`, but the stop was too tight: it halted around the early `+8,600` peak and missed the later move toward `+13,500`.

## Concerns

- Both drafts are still Round 4 tuned. They are not pure universal algorithms.
- The voucher strikes, fair anchors, and counterparty names are parameter choices from this round's data.
- The loose trailing stop is more overfit than the base draft because it is motivated by the observed official PnL path.
- Local replay can overstate official performance because fill assumptions differ from the exchange simulator.
- Hydrogel is disabled because the public/official behavior was unstable despite possible hidden upside.

## Recommendation

Use `round4_solution_draft_upload.py` if the priority is the least overfit submission.

Use `round4_solution_draft_trailing_loose_upload.py` only if the priority is protecting a large intraday gain after the observed run-up-and-drawdown behavior. It is a risk control, but it is also the more path-fitted choice.
