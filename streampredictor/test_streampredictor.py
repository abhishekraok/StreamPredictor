import os
import tempfile
from unittest import TestCase

from streampredictor.pop import Pop
from streampredictor.stream_predictor import StreamPredictor
from streampredictor import generator

save_filename = os.path.join(tempfile.gettempdir(), 'streampredictor_test.tsv')
training_words = 'cat hat mat bat sat in the barn'.split(' ')


class TestPatternOfPatterns(TestCase):
    @staticmethod
    def get_sample():
        sample = StreamPredictor()
        sample.train(training_words)
        return sample

    def test_save(self):
        sample = self.get_sample()
        sample.file_manager.save_tsv(save_filename)
        self.assertTrue(os.path.isfile(save_filename))
        os.remove(save_filename)

    def test_load(self):
        sample = self.get_sample()
        sample.file_manager.save_tsv(save_filename)
        self.assertTrue(os.path.isfile(save_filename))
        empty_sample = StreamPredictor()
        self.assertFalse(len(empty_sample.pop_manager.pattern_collection) > 10)
        empty_sample.file_manager.load_tsv(save_filename)
        self.assertTrue(len(empty_sample.pop_manager.pattern_collection) > 10)
        os.remove(save_filename)

    # def test_train_increases_patterns(self):
    #     sample = self.get_sample()
    #     pattern_count = len(sample.pop_manager.pattern_collection)
    #     sample.train(training_words)
    #     self.assertGreater(len(sample.pop_manager.pattern_collection), pattern_count)

    def test_save_load_equal(self):
        original = self.get_sample()
        original.pop_manager.pattern_collection['karma'] = Pop('karma')
        original_string = original.pop_manager.status()
        original.file_manager.save_tsv(save_filename)
        loaded = StreamPredictor()
        loaded.file_manager.load_tsv(save_filename)
        loaded_string = loaded.pop_manager.status()
        self.assertEqual(original_string, loaded_string)
        self.assertEqual(len(original.pop_manager.pattern_collection), len(loaded.pop_manager.pattern_collection))

    def test_change_component(self):
        a, ab, abc, sp, b, c, bc = self.setup_simple_patterns()
        self.assertTrue(len(a.first_child_parents) == 1)
        self.assertTrue(len(ab.first_child_parents) == 0)
        sp.pop_manager.change_components_string('ab', 'c', abc)
        self.assertTrue(len(a.first_child_parents) == 0)
        self.assertTrue(len(sp.pop_manager.pattern_collection['ab'].first_child_parents) == 1)

    def setup_simple_patterns(self):
        sp = StreamPredictor()
        abc = Pop('abc')
        ab = Pop('ab')
        bc = Pop('bc')
        a = Pop('a')
        b = Pop('b')
        c = Pop('c')
        sp.pop_manager.add_pop_to_vocabulary(abc)
        sp.pop_manager.add_pop_to_vocabulary(bc)
        sp.pop_manager.add_pop_to_vocabulary(ab)
        sp.pop_manager.add_pop_to_vocabulary(a)
        sp.pop_manager.add_pop_to_vocabulary(b)
        sp.pop_manager.add_pop_to_vocabulary(c)
        abc.set_components(a, bc)
        return a, ab, abc, sp, b, c, bc

    def test_is_right_child(self):
        a, ab, abc, sp, b, c, bc = self.setup_simple_patterns()
        self.assertTrue(abc.is_child(bc))
        self.assertFalse(abc.is_child(c))

    def test_has_common_child(self):
        sp, ab, abxe, abyd, xe, yd = self.form_simple_tree()
        self.assertTrue(abxe.has_common_child(abyd))
        self.assertTrue(ab.has_common_child(ab))
        self.assertFalse(xe.has_common_child(yd))

    def form_simple_tree(self):
        sp = StreamPredictor()
        abxe = Pop('abxe')
        abxe.strength = 600
        abyd = Pop('abyd')
        abyd.strength = 400
        ab = Pop('ab')
        xe = Pop('xe')
        yd = Pop('yd')
        a = Pop('a')
        b = Pop('b')
        d = Pop('d')
        e = Pop('e')
        x = Pop('x')
        y = Pop('y')
        sp.pop_manager.add_pop_to_vocabulary(abxe)
        sp.pop_manager.add_pop_to_vocabulary(ab)
        sp.pop_manager.add_pop_to_vocabulary(xe)
        sp.pop_manager.add_pop_to_vocabulary(yd)
        sp.pop_manager.add_pop_to_vocabulary(a)
        sp.pop_manager.add_pop_to_vocabulary(b)
        sp.pop_manager.add_pop_to_vocabulary(d)
        sp.pop_manager.add_pop_to_vocabulary(e)
        sp.pop_manager.add_pop_to_vocabulary(x)
        sp.pop_manager.add_pop_to_vocabulary(y)
        xe.set_components(x, e)
        yd.set_components(y, d)
        abxe.set_components(ab, xe)
        abyd.set_components(ab, yd)
        return sp, ab, abxe, abyd, xe, yd

    def test_pop_manager_repr(self):
        sample = self.get_sample()
        spm = sample.pop_manager.__repr__()
        self.assertGreater(len(spm), 10)

    def test_pop_get_next_prediction(self):
        sp, ab, abxe, abyd, xe, yd = self.form_simple_tree()
        words, probabilites = ab.get_next_smallest_distribution()
        self.assertTrue(len(words) > 1)
        self.assertEqual(probabilites[0], 0.6)
        self.assertEqual(probabilites[1], 0.4)
        self.assertTrue('x' in words)
        self.assertTrue('y' in words)

    def test_pop_get_smallest_next_prediction(self):
        sp, ab, abxe, abyd, xe, yd = self.form_simple_tree()
        words, probabilites = ab.get_next_smallest_distribution()
        self.assertTrue(len(words) > 1)
        self.assertEqual(probabilites[0], 0.6)
        self.assertEqual(probabilites[1], 0.4)
        self.assertTrue('x' in words)
        self.assertTrue('y' in words)

    def test_perplexity_step(self):
        sp, ab, abxe, abyd, xe, yd = self.form_simple_tree()
        previous_words = ['ab']
        actual_next_word = 'x'
        N = 1
        log_running_perplexity = 0
        perplexity_list = []
        abxe.strength = 500
        abyd.strength = 500
        N, log_running_perplexity = sp.generator.perplexity_step(N, log_running_perplexity, perplexity_list,
                                                                 previous_words, actual_next_word)
        self.assertEqual(N, 2)
        self.assertAlmostEqual(perplexity_list[0], 2.17, places=2)

    def test_find_next_word(self):
        sp, ab, abxe, abyd, xe, yd = self.form_simple_tree()
        words = ['ab', 'd']
        current_pop, remaining = sp.pop_manager.get_next_pop(words[1:])
        self.assertEqual(current_pop.unrolled_pattern, 'd')
        self.assertEqual(remaining, [])
        words = ['ab', 'x', 'e']
        current_pop, remaining = sp.pop_manager.get_next_pop(words[1:])
        self.assertEqual(current_pop.unrolled_pattern, 'xe')
        self.assertEqual(remaining, [])


class TestGenerator(TestCase):
    @staticmethod
    def sample_generator():
        pattern_collection = {'where': Pop('where'), 'are': Pop('are'), 'whereare': Pop('whereare')}
        pattern_collection['whereare'].set_components(pattern_collection['where'], pattern_collection['are'])
        return generator.Generator(pattern_collection, vocabulary={'where', 'are'})

    def test_generate_stream(self):
        sample = TestPatternOfPatterns.get_sample()
        generated = sample.generate(10)
        self.assertGreater(len(generated), 9)

    def test_choose_next_word_valid(self):
        test_generator = TestGenerator.sample_generator()
        test_generator.uniform_prediction_ratio = 0
        generated_word = test_generator.choose_next_word(['where'])
        self.assertEqual(generated_word, 'are')

    def test_choose_next_word_base(self):
        test_generator = TestGenerator.sample_generator()
        generated_word = test_generator.choose_next_word(['are'])
        self.assertTrue(generated_word in test_generator.pattern_collection.keys())

    def test_longest_pop(self):
        test_generator = TestGenerator.sample_generator()
        where_are = test_generator.longest_pop(['where', 'are'])
        self.assertEqual(where_are.unrolled_pattern, 'are')
        where = test_generator.longest_pop(['where'])
        self.assertEqual(where.unrolled_pattern, 'where')


class TestStreamPredictor(TestCase):
    def test_sequence_learn(self):
        sp = StreamPredictor()
        first_words_list = ['that', 'is', 'good']
        sp.train(first_words_list)
        first_next_words, first_probabilities = sp.pop_manager.get('that').get_next_smallest_distribution()
        self.assertEqual(['is'], first_next_words)
        self.assertEqual([1.0], first_probabilities)
        second_words_list = ['that', 'isnt', 'good']
        sp.train(second_words_list)
        second_next_words, second_probabilities = sp.pop_manager.get('that').get_next_smallest_distribution()
        self.assertEqual({'is', 'isnt'}, set(second_next_words))
        self.assertListEqual([0.5, 0.5], list(second_probabilities))

    def test_perplexity_handles_composite_collision(self):
        """Training combines 'sell'+'off' into the composite Pop 'selloff'.
        If 'selloff' later appears as a test token, it must still get
        promoted to vocabulary so perplexity_step finds it in the
        distribution instead of raising ValueError."""
        sp = StreamPredictor()
        sp.train(['sell', 'off', 'sell', 'off'])
        self.assertIn('selloff', sp.pop_manager.pattern_collection)
        self.assertNotIn('selloff', sp.pop_manager.vocabulary)
        sp.calculate_perplexity(['selloff', 'sell'], verbose=False)
        self.assertIn('selloff', sp.pop_manager.vocabulary)
