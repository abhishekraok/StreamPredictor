import numpy as np
from streampredictor import constants
from streampredictor import pop
from streampredictor import utitlities


class Generator():
    def __init__(self, pattern_collection, vocabulary):
        """
        Generates words from pattern collection

        :type pattern_collection:dict[str,pop.Pop]
        """
        self.pattern_collection = pattern_collection
        self.vocabulary = vocabulary
        self.uniform_prediction_ratio = constants.uniform_prediction_ratio

    @property
    def vocabulary_count(self):
        return len(self.vocabulary)

    def generate_words(self, word_length, seed=None):
        """
        Returns the list of generated words.

        :type word_length: int
        :type seed: str
        :rtype: list[str]
        """
        print('Generating words with word count = ', word_length)
        if not seed:
            current_word = self.get_random_base_word()
        else:
            current_word = seed
        generated_output = [current_word]
        for i in range(word_length - 1):
            next_word = self.choose_next_word(generated_output)
            generated_output.append(next_word)
        return generated_output

    def choose_next_word(self, previous_list_of_words):
        """
        Get the next word given previous list of words.

        :type previous_list_of_words:list[str]
        :rtype: str
        """
        words_distribution = self.next_word_distribution(previous_list_of_words)
        words = list(words_distribution.keys())
        probabilities = np.array([float(p) for p in words_distribution.values()])
        return np.random.choice(words, p=probabilities)

    def next_word_distribution(self, previous_list_of_words):
        """
        Gets the distribution for the predicted next word given list of past words.

        :type previous_list_of_words: list[str]
        :rtype: dict[str,float]
        """
        current_pop = self.longest_pop(previous_list_of_words)
        words, probabilities = current_pop.get_next_smallest_distribution()
        if current_pop.belongs_to_category:
            category_words, category_probabilities = current_pop.belongs_to_category.get_next_smallest_distribution()
            words = words + category_words
            probabilities = np.hstack([0.5 * probabilities, 0.5 * category_probabilities])
        predicted_distribution = dict((i, j) for i, j in zip(words, probabilities))
        uniform_distribution = self.get_uniform_distribution()
        final_distribution = utitlities.combine_normalize_distribution(
            [predicted_distribution, uniform_distribution],
            weights=[1 - self.uniform_prediction_ratio, self.uniform_prediction_ratio])
        return final_distribution

    def longest_pop(self, list_of_words):
        """
        Get the longest pop in the given list of words ending in the last word.

        :type list_of_words:list[str]
        :rtype: pop.Pop
        """
        end = len(list_of_words)
        start = max(0, end - constants.maximum_pattern_length)
        for j in range(start, end - 1):
            current_word = ''.join(list_of_words[j:])
            if current_word in self.pattern_collection:
                current_pop = self.pattern_collection[current_word]
                words, probabilities = current_pop.get_next_smallest_distribution()
                if len(words) < 1:
                    continue
                return current_pop
        return self.pattern_collection[list_of_words[-1]]

    def get_random_base_word(self):
        """
        Returns a random string from pattern collection

        :rtype: str
        """
        chosen_pop = np.random.choice(list(self.pattern_collection.values()))
        while chosen_pop.first_component:
            chosen_pop = chosen_pop.first_component
        return chosen_pop.unrolled_pattern

    def perplexity_step(self, N, log_running_perplexity, perplexity_list, previous_words, actual_next_word):
        distribution = self.next_word_distribution(previous_words)
        if actual_next_word not in distribution:
            raise ValueError('Next word not present in vocabulary:', actual_next_word)
        likelihood = distribution[actual_next_word]
        log_running_perplexity -= np.log2(likelihood)
        perplexity_list.append(2 ** (log_running_perplexity * (1 / float(N))))
        print('Current Perplexity = {0}\r'.format(perplexity_list[-1]), end=' ')
        N += 1
        return N, log_running_perplexity

    def get_uniform_distribution(self):
        return dict((i, 1.0 / self.vocabulary_count) for i in self.vocabulary)
