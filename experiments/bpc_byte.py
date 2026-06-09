"""
Byte-level bpc benchmark on PTB. One Pop token per byte (no word
tokenization, no rotation). Trains once, then evaluates bpc under several
prediction modes / weighting variants on identical model state. gzip on
the same byte string is the apples-to-apples baseline.
"""
import gzip
from pathlib import Path

from streampredictor.stream_predictor import StreamPredictor
from streampredictor import data_fetcher, generator as generator_mod

REPO_ROOT = Path(__file__).resolve().parent.parent
input_text_file = REPO_ROOT / 'data' / 'ptb.test.txt'
pattern_store = REPO_ROOT / 'PatternStore'
pattern_store.mkdir(exist_ok=True)

max_bytes = 2 * 10 ** 5
test_length = -10 ** 4

tokens = data_fetcher.get_bytes_from_file(str(input_text_file), max_bytes)
print('Read {0} bytes; train on {1}, test on {2}'.format(
    len(tokens), len(tokens) + test_length, -test_length))

sp = StreamPredictor()
sp.train(tokens[:test_length])
sp.file_manager.save_tsv(str(pattern_store / 'ptb_bpc_byte.tsv'))

variants = [
    ('longest',                dict(prediction_mode='longest')),
    ('strength_blend (pure)',  dict(prediction_mode='strength_blend')),
    ('strength_blend x len',   dict(prediction_mode='strength_blend', blend_length_bonus=True)),
    ('log_strength',           dict(prediction_mode='strength_blend', blend_log_strength=True)),
    ('log_strength x len',     dict(prediction_mode='strength_blend', blend_log_strength=True, blend_length_bonus=True)),
]

results = {}
for label, kwargs in variants:
    sp.generator = generator_mod.Generator(
        sp.pop_manager.pattern_collection,
        sp.pop_manager.vocabulary,
        **kwargs,
    )
    print('\n--- evaluating {0!r} ---'.format(label))
    results[label] = sp.calculate_bpc(tokens[test_length:], verbose=True,
                                      separator='', encoding='latin-1')

scored_text = ''.join(tokens[test_length:][1:]).encode('latin-1')
gzipped = gzip.compress(scored_text)
gzip_bpc = len(gzipped) * 8 / len(scored_text)

print()
print('=== byte-level bpc on {0}-byte PTB held-out slice ==='.format(len(scored_text)))
print('raw bytes                       : 8.0000 bpc')
print('gzip                            : {0:.4f} bpc ({1} bytes)'.format(gzip_bpc, len(gzipped)))
for label, _ in variants:
    r = results[label]
    print('{0:30s}  : {1:.4f} bpc'.format(label, r['bpc_intrinsic']))

# Sweep uniform_prediction_ratio on the winning variant.
print()
print('=== uniform_prediction_ratio sweep on log_strength ===')
sweep_results = {}
for ratio in (1e-6, 1e-5, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1):
    gen = generator_mod.Generator(
        sp.pop_manager.pattern_collection,
        sp.pop_manager.vocabulary,
        prediction_mode='strength_blend',
        blend_log_strength=True,
    )
    gen.uniform_prediction_ratio = ratio
    sp.generator = gen
    r = sp.calculate_bpc(tokens[test_length:], verbose=False,
                         separator='', encoding='latin-1')
    sweep_results[ratio] = r['bpc_intrinsic']
    print('  ratio={0:6.3f}  bpc={1:.4f}'.format(ratio, r['bpc_intrinsic']))
best_ratio, best_bpc = min(sweep_results.items(), key=lambda kv: kv[1])
print('Best: ratio={0}  bpc={1:.4f}  (gzip={2:.4f})'.format(best_ratio, best_bpc, gzip_bpc))
