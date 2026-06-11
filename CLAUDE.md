# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Install (editable, with test deps) and run the suite — all from the repo root:

```bash
pip install -e ".[dev]"
pytest
```

Run a single test:

```bash
pytest streampredictor/test_streampredictor.py::TestPatternOfPatterns::test_save
```

Run an experiment (works from any cwd; paths are anchored to the repo via `Path(__file__)`):

```bash
python experiments/perplexity.py        # train on PTB, save TSV, plot perplexity
python experiments/simple_generation.py # tiny end-to-end demo
python experiments/bpc.py               # word-level bits-per-character vs gzip
python experiments/bpc_byte.py          # byte-level bpc, A/B prediction modes, uniform sweep
python experiments/visualize_predictions.py  # self-contained HTML report: loss ribbon, 20 random + 20 worst positions with per-Pop attribution
```

Targets Python 3.10+ (CI matrix in `.github/workflows/ci.yml` runs 3.10, 3.11, 3.12). Packaging is in `pyproject.toml`; runtime deps are `nltk`, `numpy`, `matplotlib`; `pytest` lives under the `[dev]` extra. The first call to `data_fetcher.get_clean_words_from_file` lazy-downloads `punkt_tab` via `nltk.download` — no vendored `nltk_data`.

## Architecture

The library models sequential data (text) by building a hierarchical, evolving dictionary of patterns. The README and `docs/main.tex` describe the underlying paper; the points below are the implementation details a reader can't recover from skimming the files.

### The Pop tree (`streampredictor/pop.py`)

A `Pop` ("Pattern of Pattern") is a node in a binary tree. Invariant: `pop.unrolled_pattern == pop.first_component.unrolled_pattern + pop.second_component.unrolled_pattern`. The string is the identity — `PopManager.pattern_collection` is keyed by `unrolled_pattern`, so two Pops with the same string must not coexist.

Each Pop also tracks `first_child_parents`: every parent that has *this* Pop as its `first_component`. This back-pointer is what `get_next_smallest_distribution` walks to predict the next word — for each parent reachable from the current Pop, it descends `second_component.first_component...` to the leftmost leaf and uses that as a successor candidate, weighted by parent `strength` (or `log1p(strength)` when called with `use_log_strength=True`; see Prediction). Keeping `first_child_parents` consistent with `first_component` is fragile: any code that reassigns `first_component` must remove the old back-pointer first (see `change_components_string`).

### Evolution loop (`streampredictor/pop_manager.py`)

Training in `StreamPredictor.train`:
1. Greedy-longest-match split the input via `get_next_pop` (looks ahead up to `maximum_word_count = 40` tokens).
2. For each adjacent pair, `pop.combine` makes a new Pop and `ingest` adds + feeds it.
3. Every `constants.occasional_step_count` (1000) ingests, run `occasional_step`: `decay` all strengths, `cull` Pops below the limit, then `refactor` to re-parent any Pop whose existing `(first_component, second_component)` are weaker than some alternative split of its string.

Strength dynamics are driven by `constants.feed_ratio = [0.5, 0.25, 0.25]` (self / first child / second child, recursive) and capped at `max_strength`. Tuning these and the values in `PopManager.__init__` (`feed_strength_gain`, `required_repeats`, `maximum_word_count`) is the main way to change learning behavior.

### Prediction (`streampredictor/generator.py`)

`Generator` has two `prediction_mode` settings:

- `'longest'` (default, original behavior): `next_word_distribution` mixes (a) the longest-matching Pop's successor distribution from `get_next_smallest_distribution`, (b) — if the Pop has a `belongs_to_category` — a 50/50 blend with that category's successors, and (c) a uniform distribution over the full vocabulary, weighted by `constants.uniform_prediction_ratio = 0.1`. The uniform smoothing is what keeps `calculate_perplexity` from blowing up when the actual next word isn't in the predicted distribution, so don't remove it without a replacement smoother.
- `'strength_blend'`: instead of committing to the longest match, blend the successor distributions of *every* matching-suffix Pop, each weighted by `_pop_blend_weight(pop)`. The weight is `strength` by default; `blend_log_strength=True` makes it `log1p(strength)` (compresses the dynamic range so rarely-seen long patterns get a fairer voice and the `max_strength` cap matters less); `blend_length_bonus=True` multiplies by `len(unrolled_pattern)`. When `blend_log_strength` is on, the within-Pop parent vote in `Pop.get_next_smallest_distribution` is also log-weighted for consistency. A uniform floor sized to carry `uniform_prediction_ratio` of the total mass is added at the end.

Note that `belongs_to_category` / `members_of_category` are wired through the code but the population logic (`generalize`) is commented out in tests — categorization is partially implemented. The category blend in step (b) above only applies to `'longest'` mode.

Both modes flow through `_distribution_components`, and `next_word_distribution_traced` returns the final distribution plus every `(Pop, weight, votes)` component — the attribution API behind `experiments/visualize_predictions.py`, which writes a self-contained HTML report (`PatternStore/prediction_report.html`) and per-position JSONL trace.

**Known scaling ceiling on byte-level data.** With `prediction_mode='strength_blend', blend_log_strength=True, uniform_prediction_ratio=0.001` (the tuned configuration) the byte-level bpc on PTB plateaus around 3.83 — adding more training data slightly *worsens* it while gzip's bpc on the same data drops. The pathology is in the algorithm structure (long-but-rare patterns accumulate and dominate the greedy longest-match parser), not in the weighting. Don't propose "more data" or "tune the ratio lower" as the fix — both have already been swept. See `memory/algorithm_scaling_ceiling.md` and the commit log around `46881c7` for the data.

### Persistence (`streampredictor/file_manager.py`)

`save_tsv` / `load_tsv` round-trip the pattern collection through a tab-separated file. The format is `pattern \t strength \t component1 \t component2 \t parent1 \t parent2 ...`. Loading is two-pass: first instantiate every Pop with its strength, then wire `set_components` and `first_child_parents`. `test_save_load_equal` is the regression guard — any change to either the on-disk format or the Pop fields it serializes must keep that test green.

### Surface API

`StreamPredictor` (in `stream_predictor.py`) is the only thing experiments and the notebook import. It owns one `PopManager`, one `FileManager`, one `Generator`, and exposes `train(words)`, `generate(n, seed=None)`, `calculate_perplexity(words)`, and `calculate_bpc(words, separator=' ', encoding='utf-8')`. The constructor takes `prediction_mode`, `blend_log_strength`, `blend_length_bonus` and forwards them to `Generator`. The `Generator` is constructed once with references to `pop_manager.pattern_collection` and `pop_manager.vocabulary`, so those must remain the *same dict/set objects* across the lifetime of the predictor — don't reassign them on `PopManager`. Experiments that A/B prediction modes swap `sp.generator` for a fresh `Generator` built against the same collection/vocabulary objects (see `experiments/bpc_byte.py`).

`calculate_bpc` returns a dict with `bpc_intrinsic` (assumes the decoder already knows the vocabulary) and `bpc_with_vocab` (charges raw bytes for every previously-unseen test token via `separator.join(new_words).encode(encoding)`). The with-vocab number is the apples-to-apples comparison against gzip on the same reconstructed byte string. Word-level uses `separator=' '`, `encoding='utf-8'`; byte-level uses `separator=''`, `encoding='latin-1'` so the reconstruction is exactly the original byte stream. `data_fetcher.get_bytes_from_file(path, max_bytes)` is the byte-level loader (no rotation, no cleaning, latin-1 decode for 1:1 byte↔codepoint mapping).

### Peripheral code

- `experiments/` — driver scripts (`perplexity.py`, `simple_generation.py`); paths anchor to the repo via `Path(__file__).resolve().parent.parent`, and they create `PatternStore/` on demand.
- `notebook/` — Jupyter demos; `Stream Predictor Language Demo.ipynb` is the one linked from the README.
- `docs/main.tex` + `pop.png` — the paper source.
- `PatternStore/` — saved TSV models (gitignored except for `first_pattern.tsv`).
