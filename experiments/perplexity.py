from pathlib import Path

import matplotlib.pyplot as plt

from streampredictor.stream_predictor import StreamPredictor
from streampredictor import data_fetcher

REPO_ROOT = Path(__file__).resolve().parent.parent
input_text_file = REPO_ROOT / 'data' / 'ptb.test.txt'
pattern_store = REPO_ROOT / 'PatternStore'
pattern_store.mkdir(exist_ok=True)

max_input_length = 10 ** 4
words = data_fetcher.get_clean_words_from_file(str(input_text_file), max_input_length)
sp = StreamPredictor()
test_length = -1000
sp.train(words[:test_length])
sp.file_manager.save_tsv(str(pattern_store / 'ptb.tsv'))
perplexity_list = sp.calculate_perplexity(words=words[test_length:], verbose=True)
plt.plot(perplexity_list)
plt.xlabel('Time')
plt.ylabel('Perplexity')
plt.title('Perplexity during training')
plt.show()
