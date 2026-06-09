"""
Byte-level bpc benchmark on PTB. One Pop token per byte (no word
tokenization, no rotation), trained on a slice, evaluated on the held-out
tail, and compared against gzip on the same byte string. This is the
tokenization regime that matters for Hutter Prize — same loader will point
at enwik8 once we're satisfied with the per-byte numbers here.
"""
import gzip
from pathlib import Path

from streampredictor.stream_predictor import StreamPredictor
from streampredictor import data_fetcher

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

result = sp.calculate_bpc(tokens[test_length:], verbose=True,
                          separator='', encoding='latin-1')

scored_text = ''.join(tokens[test_length:][1:]).encode('latin-1')
gzipped = gzip.compress(scored_text)
gzip_bpc = len(gzipped) * 8 / len(scored_text)

print()
print('=== byte-level bpc on {0}-byte PTB held-out slice ==='.format(len(scored_text)))
print('raw bytes                    : 8.0000 bpc')
print('gzip                         : {0:.4f} bpc ({1} bytes)'.format(gzip_bpc, len(gzipped)))
print('streampredictor (intrinsic)  : {0:.4f} bpc  (decoder knows vocab)'.format(result['bpc_intrinsic']))
print('streampredictor (with vocab) : {0:.4f} bpc  (charges {1} bits for {2} new bytes)'.format(
    result['bpc_with_vocab'], result['vocab_bits'], result['new_words']))
