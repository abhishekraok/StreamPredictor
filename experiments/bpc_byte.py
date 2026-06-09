"""
Byte-level bpc benchmark on PTB. One Pop token per byte (no word
tokenization, no rotation). Trains once, then evaluates bpc under both
prediction modes (longest-match vs strength-weighted blend across all
matching suffixes) so we can A/B them on identical model state. gzip on
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

results = {}
for mode in ('longest', 'strength_blend'):
    sp.generator = generator_mod.Generator(
        sp.pop_manager.pattern_collection,
        sp.pop_manager.vocabulary,
        prediction_mode=mode,
    )
    print('\n--- evaluating prediction_mode={0!r} ---'.format(mode))
    results[mode] = sp.calculate_bpc(tokens[test_length:], verbose=True,
                                     separator='', encoding='latin-1')

scored_text = ''.join(tokens[test_length:][1:]).encode('latin-1')
gzipped = gzip.compress(scored_text)
gzip_bpc = len(gzipped) * 8 / len(scored_text)

print()
print('=== byte-level bpc on {0}-byte PTB held-out slice ==='.format(len(scored_text)))
print('raw bytes                          : 8.0000 bpc')
print('gzip                               : {0:.4f} bpc ({1} bytes)'.format(gzip_bpc, len(gzipped)))
for mode in ('longest', 'strength_blend'):
    r = results[mode]
    print('streampredictor {0:14s} : {1:.4f} bpc intrinsic / {2:.4f} with-vocab ({3} new bytes)'.format(
        mode, r['bpc_intrinsic'], r['bpc_with_vocab'], r['new_words']))
