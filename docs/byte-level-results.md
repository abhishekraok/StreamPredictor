# Byte-level results (2026)

Status: paused 2026-06-11. The byte-level plateau at 3.83 bpc was judged not
worth more time. The goal had been a standard compression benchmark, with
Hutter Prize enwik8/enwik9 as the target and gzip parity as the first milestone.

All numbers are **intrinsic bpc** (no vocabulary cost), byte-level (one Pop per
byte). Reproduce with `python experiments/bpc_byte.py`.

## Prediction variants

PTB test file, 200K bytes train / 10K held out, `uniform_prediction_ratio=0.1`.

| Variant | bpc | Notes |
|---|---|---|
| longest (original) | 4.96 | longest suffix Pop, blended 90/10 with uniform |
| strength_blend | 4.75 | every matching-suffix Pop, raw strength weights |
| strength_blend × length | 4.50 | weight × length in chars |
| log_strength | 4.01 | `log1p(strength)` across Pops and within the parent vote |
| log_strength × length | 4.06 | the length term brings longest-match dominance back |
| gzip | 3.13 | `gzip.compress` on the same held-out bytes |

## `uniform_prediction_ratio` sweep on log_strength

| Ratio | bpc |
|---|---|
| 1e-6 to 1e-3 | 3.83 (flat; 1e-3 is the tuned value) |
| 5e-3, 1e-2 | 3.84 |
| 5e-2 | 3.92 |
| 1e-1 (default) | 4.01 |
| 5e-1 | 4.63 |

The extended sweep is commit `61d15bc`.

## Tuned configuration

```python
sp = StreamPredictor(
    prediction_mode='strength_blend',
    blend_log_strength=True,
    blend_length_bonus=False,
)
sp.generator.uniform_prediction_ratio = 0.001  # not a constructor argument
```

This gives 3.83 bpc, 0.70 above gzip.

## More data does not help

| Corpus | Model | gzip | Gap |
|---|---|---|---|
| PTB 200K train / 10K test | 3.83 | 3.13 | +0.70 |
| PTB 400K train / 40K test | 3.88 | 2.85 | +1.03 |
| pride.txt 600K / 60K test | 4.14 | 3.12 | +1.01 |

Doubling the training data made the model slightly worse while gzip improved.
Long-but-rare patterns accumulate as data grows and the greedy longest-match
parser commits to them, hiding short-pattern statistics. Closing the gap needs
a structural change (PPM-style exclusion, pattern formation beyond
adjacent-pair combine), not more data or a lower ratio.

## Where the bits go

From `experiments/visualize_predictions.py` on the tuned configuration
(3.8296 bpc against gzip's 3.1339):

- **Uniform fallback is not the problem:** 0.1% of positions, 0.3% of bits.
- **Loss is not concentrated after long matches:** mean bits are roughly flat
  across longest-match length (length 2 is worst at 4.40; length 6+ is 3.0-3.4).
- **Broad mediocrity:** the true byte is the rank-1 prediction at 25.2% of
  positions, and the worst 10% of positions carry only 21.3% of the bits.
- **Short matches:** the median longest suffix match is about 2 bytes, so the
  model behaves like a noisy bigram at most positions. The true byte is usually
  rank 5-16, with top probabilities of 6-20%.

Suspected bug, unfixed: `Pop.get_next_smallest_distribution` returns one
successor token per voting parent, and both prediction paths collapse them with
`dict(zip(words, probabilities))`. That keeps the last parent's probability
instead of summing the votes, so popular successors are under-weighted.
