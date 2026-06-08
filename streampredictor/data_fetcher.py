import nltk
import os
import random


def _ensure_punkt():
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)


def get_random_book_local(folder):
    file = random.choice(os.listdir(folder))
    with open(folder + file) as opened_file:
        return opened_file.read()


def get_clean_text_from_file(file, max_input_stream_length):
    with open(file) as opened_file:
        text = opened_file.read()
        return clean_text(text, max_input_stream_length)


def get_clean_words_from_file(file, max_input_length):
    _ensure_punkt()
    with open(file) as opened_file:
        text = opened_file.read()
        clean_words = nltk.word_tokenize(clean_text(text))[:max_input_length]
        print('Cleaned words, some of the first few are ', clean_words[:5])
        return clean_words


def get_words_from_ptb(file, max_input_length):
    with open(file) as opened_file:
        text = opened_file.read().replace('\n', '')
        return text.split(' ')[:max_input_length]


def clean_text(text, max_input_length=10 ** 10000):
    text = text.replace('\n', ' ')
    max_length = min(max_input_length, len(text))
    rotation = random.randint(0, max_length)
    text = text[rotation:max_length] + text[:rotation]
    # make sure to remove # for category separation
    text = ''.join(e for e in text if e.isalnum() or e in '.?", <>')
    return text
