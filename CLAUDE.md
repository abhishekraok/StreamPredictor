# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Install deps and run the test suite (matches `.travis.yml`):

```bash
pip install -r requirements.txt
cd streampredictor && nosetests
```

Run a single test:

```bash
cd streampredictor && nosetests test_streampredictor:TestPatternOfPatterns.test_save
```

Run an experiment (note: they `sys.path.insert` `..` and `../streampredictor`, so they MUST be launched from inside `experiments/`):

```bash
cd experiments && python perplexity.py        # train on PTB, save TSV, plot perplexity
cd experiments && python simple_generation.py # tiny end-to-end demo
```

Targets Python 3.5 (per `.travis.yml`); deps are `nltk`, `numpy`, `matplotlib`, `nose`. `nltk_data` is vendored in the repo root and copied to `/usr/share/nltk_data` in CI.

## Architecture

The library models sequential data (text) by building a hierarchical, evolving dictionary of patterns. The README and `docs/main.tex` describe the underlying paper; the points below are the implementation details a reader can't recover from skimming the files.

### The Pop tree (`streampredictor/pop.py`)

A `Pop` ("Pattern of Pattern") is a node in a binary tree. Invariant: `pop.unrolled_pattern == pop.first_component.unrolled_pattern + pop.second_component.unrolled_pattern`. The string is the identity — `PopManager.pattern_collection` is keyed by `unrolled_pattern`, so two Pops with the same string must not coexist.

Each Pop also tracks `first_child_parents`: every parent that has *this* Pop as its `first_component`. This back-pointer is what `get_next_smallest_distribution` walks to predict the next word — for each parent reachable from the current Pop, it descends `second_component.first_component...` to the leftmost leaf and uses that as a successor candidate, weighted by parent `strength`. Keeping `first_child_parents` consistent with `first_component` is fragile; `PopManager.fix_first_child_parents` exists to repair drift, and any code that reassigns `first_component` must remove the old back-pointer first (see `change_components_string`).

### Evolution loop (`streampredictor/pop_manager.py`)

Training in `StreamPredictor.train`:
1. Greedy-longest-match split the input via `get_next_pop` (looks ahead up to `maximum_word_count = 40` tokens).
2. For each adjacent pair, `pop.combine` makes a new Pop and `ingest` adds + feeds it.
3. Every `constants.occasional_step_count` (1000) ingests, run `occasional_step`: `decay` all strengths, `cull` Pops below the limit, then `refactor` to re-parent any Pop whose existing `(first_component, second_component)` are weaker than some alternative split of its string.

Strength dynamics are driven by `constants.feed_ratio = [0.5, 0.25, 0.25]` (self / first child / second child, recursive) and capped at `max_strength`. Tuning these and the values in `PopManager.__init__` (`feed_strength_gain`, `required_repeats`, `maximum_word_count`) is the main way to change learning behavior.

### Prediction (`streampredictor/generator.py`)

`next_word_distribution` mixes three things: (a) the longest-matching Pop's successor distribution from `get_next_smallest_distribution`, (b) — if the Pop has a `belongs_to_category` — a 50/50 blend with that category's successors, and (c) a uniform distribution over the full vocabulary, weighted by `constants.uniform_prediction_ratio = 0.1`. The uniform smoothing is what keeps `calculate_perplexity` from blowing up when the actual next word isn't in the predicted distribution, so don't remove it without a replacement smoother.

Note that `belongs_to_category` / `members_of_category` are wired through the code but the population logic (`generalize`) is commented out in tests — categorization is partially implemented.

### Persistence (`streampredictor/file_manager.py`)

`save_tsv` / `load_tsv` round-trip the pattern collection through a tab-separated file. The format is `pattern \t strength \t component1 \t component2 \t parent1 \t parent2 ...`. Loading is two-pass: first instantiate every Pop with its strength, then wire `set_components` and `first_child_parents`. `test_save_load_equal` is the regression guard — any change to either the on-disk format or the Pop fields it serializes must keep that test green.

### Surface API

`StreamPredictor` (in `stream_predictor.py`) is the only thing experiments and the notebook import. It owns one `PopManager`, one `FileManager`, one `Generator`, and exposes `train(words)`, `generate(n, seed=None)`, `calculate_perplexity(words)`. The `Generator` is constructed once with references to `pop_manager.pattern_collection` and `pop_manager.vocabulary`, so those must remain the *same dict/set objects* across the lifetime of the predictor — don't reassign them on `PopManager`.

### Peripheral code

- `experiments/` — driver scripts; they hardcode paths like `../Data/ptb.test.txt` (the actual file is `data/ptb.test.txt`, lowercase) and `../PatternStore/...`, so expect to fix paths or `cd` accordingly.
- `notebook/` — Jupyter demos; `Stream Predictor Language Demo.ipynb` is the one linked from the README.
- `django/` — an unrelated Django scaffold (`PotterTextGen`) for a web demo. Not wired into the library; ignore unless explicitly working on the web app.
- `docs/main.tex` + `pop.png` — the paper source.
- `PatternStore/` — saved TSV models (gitignored except for `first_pattern.tsv`).
