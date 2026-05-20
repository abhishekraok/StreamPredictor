import nltk
import urllib.request, urllib.error, urllib.parse
import time
import os
import random


def _ensure_punkt():
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)


def gutenberg_random_book():
    for i in range(100):
        book_number = random.randint(9, 2000)
        url = 'http://www.gutenberg.org/cache/epub/' + str(book_number) + '/pg' + str(book_number) + '.txt'
        response = urllib.request.urlopen(url)
        text = response.read()
        if len(text) > 1000 and is_english(text):
            print(('Got book from ', url))
            return text
        else:
            print(('Didnt get book, waiting for some time, seconds = ' + str(10 + book_number / 10)))
            time.sleep(10 + book_number / 10)


def is_english(text):
    english_words = ['here', 'there', 'where', 'some', 'and', 'but']
    for word in english_words:
        if word not in text:
            return False
    return True


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


def get_online_words(max_input_length):
    _ensure_punkt()
    text = gutenberg_random_book()
    words = nltk.word_tokenize(clean_text(text, max_input_length))
    return words


