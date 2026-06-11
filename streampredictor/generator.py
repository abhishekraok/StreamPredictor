import numpy as np
from streampredictor import constants
from streampredictor import pop
from streampredictor import utitlities


class Generator():
    def __init__(self, pattern_collection, vocabulary, prediction_mode='longest',
                 blend_log_strength=False, blend_length_bonus=False):
        """
        Generates words from pattern collection.

        prediction_mode='longest' (default, original behavior): pick the
        longest-suffix Pop with any successors and blend with uniform.
        prediction_mode='strength_blend': blend every matching-suffix Pop
        from longest to shortest, each weighted by a function of its
        strength (and optionally its length), plus a uniform floor scaled
        by uniform_prediction_ratio.

        blend_log_strength: use log1p(strength) instead of raw strength.
        Compresses the dynamic range so the max_strength cap matters less
        and rarely-seen long patterns get a fairer voice.
        blend_length_bonus: multiply each Pop's weight by its character
        length. Gives longer matches a "context bonus" when their strength
        is comparable to a shorter level's.

        :type pattern_collection:dict[str,pop.Pop]
        """
        if prediction_mode not in ('longest', 'strength_blend'):
            raise ValueError('Unknown prediction_mode: ' + repr(prediction_mode))
        self.pattern_collection = pattern_collection
        self.vocabulary = vocabulary
        self.uniform_prediction_ratio = constants.uniform_prediction_ratio
        self.prediction_mode = prediction_mode
        self.blend_log_strength = blend_log_strength
        self.blend_length_bonus = blend_length_bonus

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
        distributions, weights, _ = self._distribution_components(previous_list_of_words)
        return utitlities.combine_normalize_distribution(distributions, weights)

    def next_word_distribution_traced(self, previous_list_of_words):
        """
        Same as next_word_distribution, but also returns the components that
        went into the blend so inspection tools can attribute the prediction.

        Returns (final_distribution, components) where components is a list
        of (source, weight, distribution) triples; source is the
        contributing Pop, or None for the uniform floor. A component's
        contribution to the final probability of word w is
        weight * distribution.get(w, 0) / Z, with
        Z = sum(weight * sum(distribution.values()) over components).

        :type previous_list_of_words: list[str]
        :rtype: (dict[str,float], list[(pop.Pop|None, float, dict[str,float])])
        """
        distributions, weights, sources = self._distribution_components(previous_list_of_words)
        final_distribution = utitlities.combine_normalize_distribution(distributions, weights)
        return final_distribution, list(zip(sources, weights, distributions))

    def _distribution_components(self, previous_list_of_words):
        """
        Returns (distributions, weights, sources): the parallel lists that
        combine_normalize_distribution turns into the final prediction.
        sources holds the Pop behind each distribution, or None for the
        uniform floor.
        """
        if self.prediction_mode == 'strength_blend':
            return self._strength_blend_components(previous_list_of_words)
        return self._longest_components(previous_list_of_words)

    def _longest_components(self, previous_list_of_words):
        current_pop = self.longest_pop(previous_list_of_words)
        words, probabilities = current_pop.get_next_smallest_distribution()
        if current_pop.belongs_to_category:
            category_words, category_probabilities = current_pop.belongs_to_category.get_next_smallest_distribution()
            words = words + category_words
            probabilities = np.hstack([0.5 * probabilities, 0.5 * category_probabilities])
        predicted_distribution = dict((i, j) for i, j in zip(words, probabilities))
        distributions = [predicted_distribution, self.get_uniform_distribution()]
        weights = [1 - self.uniform_prediction_ratio, self.uniform_prediction_ratio]
        sources = [current_pop, None]
        return distributions, weights, sources

    def _strength_blend_components(self, previous_list_of_words):
        """
        Successor distributions of every matching-suffix Pop, each weighted
        by its own strength, plus a uniform floor sized so it carries
        uniform_prediction_ratio of the total mass when any Pop matches.
        Falls back to uniform alone if no Pop matches.
        """
        distributions = []
        weights = []
        sources = []
        for current_pop in self._matching_suffix_pops(previous_list_of_words):
            words, probabilities = current_pop.get_next_smallest_distribution(
                use_log_strength=self.blend_log_strength)
            if len(words) == 0:
                continue
            distributions.append(dict(zip(words, probabilities)))
            weights.append(self._pop_blend_weight(current_pop))
            sources.append(current_pop)
        if distributions:
            uniform_weight = self.uniform_prediction_ratio * sum(weights) / max(1 - self.uniform_prediction_ratio, 1e-9)
        else:
            uniform_weight = 1.0
        distributions.append(self.get_uniform_distribution())
        weights.append(uniform_weight)
        sources.append(None)
        return distributions, weights, sources

    def _pop_blend_weight(self, current_pop):
        weight = max(current_pop.strength, 0)
        if self.blend_log_strength:
            weight = float(np.log1p(weight))
        if self.blend_length_bonus:
            weight = weight * len(current_pop.unrolled_pattern)
        return weight

    def _matching_suffix_pops(self, list_of_words):
        """
        Walk suffixes longest-to-shortest and yield every Pop in the
        collection that matches one. Unlike longest_pop this returns the
        full chain, not just the first hit.
        """
        end = len(list_of_words)
        start = max(0, end - constants.maximum_pattern_length)
        matched = []
        for j in range(start, end):
            current_word = ''.join(list_of_words[j:])
            if current_word in self.pattern_collection:
                matched.append(self.pattern_collection[current_word])
        return matched

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
