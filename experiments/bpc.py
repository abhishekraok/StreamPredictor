"""
Bits-per-character benchmark on PTB. Trains StreamPredictor on a slice,
measures bpc on the held-out tail, and compares against gzip on the same
reconstructed byte string. First step toward Hutter-Prize-style evaluation
before moving to char/byte-level + enwik8.
"""
import gzip
from pathlib import Path

from streampredictor.stream_predictor import StreamPredictor
from streampredictor import data_fetcher

REPO_ROOT = Path(__file__).resolve().parent.parent
input_text_file = REPO_ROOT / 'data' / 'ptb.test.txt'
pattern_store = REPO_ROOT / 'PatternStore'
pattern_store.mkdir(exist_ok=True)

max_input_length = 10 ** 4
test_length = -1000

words = data_fetcher.get_clean_words_from_file(str(input_text_file), max_input_length)
sp = StreamPredictor()
sp.train(words[:test_length])
sp.file_manager.save_tsv(str(pattern_store / 'ptb_bpc.tsv'))

result = sp.calculate_bpc(words[test_length:], verbose=True)

scored_text = ' '.join(words[test_length:][1:]).encode('utf-8')
gzipped = gzip.compress(scored_text)
gzip_bpc = len(gzipped) * 8 / len(scored_text)

print()
print('=== bpc on {0}-byte PTB held-out slice ==='.format(len(scored_text)))
print('raw bytes                    : 8.0000 bpc')
print('gzip                         : {0:.4f} bpc ({1} bytes)'.format(gzip_bpc, len(gzipped)))
print('streampredictor (intrinsic)  : {0:.4f} bpc  (decoder knows vocab)'.format(result['bpc_intrinsic']))
print('streampredictor (with vocab) : {0:.4f} bpc  (charges {1} bits for {2} new words)'.format(
    result['bpc_with_vocab'], result['vocab_bits'], result['new_words']))
