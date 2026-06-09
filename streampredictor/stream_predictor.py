import numpy as np

from streampredictor import file_manager
from streampredictor import pop
from streampredictor import pop_manager
from streampredictor import constants
from streampredictor import generator


class StreamPredictor:
    def __init__(self, pm=None, prediction_mode='longest',
                 blend_log_strength=False, blend_length_bonus=False):
        if pm:
            self.pop_manager = pm  # type: pop_manager
        else:
            self.pop_manager = pop_manager.PopManager()  # type: pop_manager
        self.file_manager = file_manager.FileManager(self.pop_manager)
        self.generator = generator.Generator(self.pop_manager.pattern_collection,
                                             self.pop_manager.vocabulary,
                                             prediction_mode=prediction_mode,
                                             blend_log_strength=blend_log_strength,
                                             blend_length_bonus=blend_length_bonus)
        print(self.pop_manager.stats())

    def train(self, list_of_words, verbose=False):
        self.pop_manager.add_words_to_vocabulary(list_of_words)
        previous_pop = self.pop_manager.pattern_collection[list_of_words[0]]
        remaining_sequence = list_of_words[1:]
        i = 1
        while remaining_sequence and len(remaining_sequence) > 0:
            next_pop, remaining_sequence = self.pop_manager.get_next_pop(remaining_sequence)
            if next_pop is None:
                break
            new_pop = pop.combine(previous_pop, next_pop)
            self.pop_manager.ingest(new_pop)
            if i % constants.occasional_step_count == 0:
                print('Occasional step at ', i)
                self.occasional_step(i, verbose)
            previous_pop = next_pop
            i += 1
        print('Finished training in {0} steps'.format(i))

    def occasional_step(self, step_count, verbose):
        self.pop_manager.occasional_step(step_count, verbose)

    def generate(self, word_length, seed=None):
        """
        Returns the list of generated words.

        :type word_length: int
        :type seed: str
        :rtype: list[str]
        """
        generated_output = self.generator.generate_words(word_length, seed=seed)
        for word in generated_output:
            if word not in self.pop_manager.vocabulary:
                raise ValueError('Generated word not in vocabulary :' + word)
        return generated_output

    def calculate_bpc(self, words, verbose=False, separator=' ', encoding='utf-8'):
        """
        Bits-per-character on the held-out token sequence. Scores words[1:]
        under the model (words[0] is treated as given, matching
        calculate_perplexity), and divides total bits by the byte length of
        the reconstruction of those same scored tokens.

        For word-level use the defaults (space separator, utf-8). For
        byte-level (one byte per token, latin-1 decoded), pass
        separator='' and encoding='latin-1' so the reconstruction is
        exactly the original byte string.

        Returns both 'intrinsic' bpc (assumes the decoder already knows the
        full vocabulary) and 'with_vocab' bpc (charges raw bytes for each
        previously-unseen test-set token, so a decoder starting only from
        the trained model could reconstruct the text). The with-vocab number
        is the apples-to-apples comparison against gzip on the same byte
        string.
        """
        pre_vocab = set(self.pop_manager.vocabulary)
        self.pop_manager.add_words_to_vocabulary(words, verbose=verbose)
        new_words = self.pop_manager.vocabulary - pre_vocab
        vocab_bits = len(separator.join(sorted(new_words)).encode(encoding)) * 8 if new_words else 0

        total_bits = 0.0
        for i in range(1, len(words)):
            distribution = self.generator.next_word_distribution(words[:i])
            if words[i] not in distribution:
                raise ValueError('Next word not present in distribution: ' + words[i])
            total_bits += -np.log2(distribution[words[i]])
        scored_text = separator.join(words[1:])
        total_chars = len(scored_text.encode(encoding))
        bpc_intrinsic = total_bits / total_chars
        bpc_with_vocab = (total_bits + vocab_bits) / total_chars
        if verbose:
            print('Pred bits: {0:.1f}  vocab bits: {1} ({2} new words)  total chars: {3}'.format(
                total_bits, vocab_bits, len(new_words), total_chars))
            print('Intrinsic bpc: {0:.4f}  with-vocab bpc: {1:.4f}'.format(
                bpc_intrinsic, bpc_with_vocab))
        return {
            'bpc_intrinsic': bpc_intrinsic,
            'bpc_with_vocab': bpc_with_vocab,
            'pred_bits': total_bits,
            'vocab_bits': vocab_bits,
            'total_chars': total_chars,
            'new_words': len(new_words),
        }

    def calculate_perplexity(self, words, verbose=False):
        self.pop_manager.add_words_to_vocabulary(words, verbose)
        word_count = len(words)
        if verbose:
            print('Started calculating perplexity with word count = ' + str(word_count))
        log_running_perplexity = 0
        perplexity_list = []
        N = 1
        while N < word_count:
            N, log_running_perplexity = self.generator.perplexity_step(N, log_running_perplexity, perplexity_list,
                                                                       words[:N],
                                                                       words[N])
        final_log_perplexity = log_running_perplexity * (1 / float(N))
        final_perplexity = 2 ** final_log_perplexity
        if verbose:
            print('Final perplexity is ', final_perplexity)
        return perplexity_list
