from pathlib import Path

from streampredictor.stream_predictor import StreamPredictor

REPO_ROOT = Path(__file__).resolve().parent.parent
pattern_store = REPO_ROOT / 'PatternStore'
pattern_store.mkdir(exist_ok=True)

sp = StreamPredictor()
sp.occasional_step = 100

input_text = 'hello how are you hello who are you'
input_words = input_text.split(' ') * 1000
print(input_words[:20])
sp.train(input_words, verbose=True)
print('\n\nThe generated words are ')
print(sp.generate(20))
print('\nEnd of generated words\n')
sp.file_manager.save_tsv(str(pattern_store / 'simple_generate.tsv'))
print(sp.pop_manager.pattern_collection)
