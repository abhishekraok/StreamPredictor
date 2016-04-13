"""
Pattern of patterns.

Sequence prediction by remembering pattern of patterns. When a new pattern is seen it is stored and combined with
existing patterns. Concrete example.
Suppose pattern A is seen, then pattern B is seen. New pattern AB is created from these.

Patterns decay over time, if not seen are culled. Patterns are fed if they are seen, and their components are fed
half of them. Kind of adaptive coding.

Todos:
1. Graphs - done
2. Multi step look ahead
3. Single character prediction
4. Measure of prediction , avg prediction rate
5. Measure of interesting, find interesting stuff to learn.
6. Crawl web.
7. Binary search in find next pattern()

Idea:
    1. (done) Refactoring: See if instead of current components, new components which are stronger can be set.
    e.g. if pattern is ABCD = {ABC:D} but AB and CD are stronger then set ABCD = {AB:CD}
    2. same category, must split into sub patterns. e.g. if ABXC and ABYC are found to be similar then only X
    is similar to Y
    3. Create category only if the maximum probability < 1.1 * 1/N
"""
import numpy as np

import ProtobufManager
from Pop import Pop
import MainFunctions

# constants
max_input_stream_length = 10000000
maxlen_word = 40  # maximum pattern length
required_repeats = 5  # if seen less than this many times, patterns won't survive on the long run.
feed_ratio_parent_category = 0.5
generalize_intersection_ratio = 0.75
generalize_common_required_count = 2


class PopManager:
    def __init__(self):
        self.patterns_collection = dict()
        self.feed_strength_gain = 10000

    def add_pop_string(self, string):
        if string in self.patterns_collection:
            raise Exception(string + 'is already present')
        self.patterns_collection[string] = Pop(string)

    def add_pop(self, pop):
        string = pop.unrolled_pattern
        if string in self.patterns_collection:
            raise Exception(string + 'is already present')
        self.patterns_collection[string] = pop

    def set_components_from_string(self, pop, first_string, second_string):
        if not first_string or not second_string:
            raise Exception("component cannot be empty")
        if first_string in self.patterns_collection:
            pop.first_component = self.patterns_collection[first_string]
        if second_string in self.patterns_collection:
            pop.second_component = self.patterns_collection[second_string]

    def train(self, string, generalize=False):
        input_length = self.setup_train(string)
        previous_pop = self.patterns_collection[string[0]]
        i = 1
        while i < input_length - maxlen_word:
            current_pop = self.find_next_pattern(string[i:i + maxlen_word])
            self.join_pattern(previous_pop, current_pop, found_pattern_feed_ratio=1)
            previous_pop = current_pop
            i += len(current_pop.unrolled_pattern)
            if i % 1000 == 0 and i > self.feed_strength_gain:
                # Refactor, adopt stronger children, as long as one's unrolled pattern is same.
                self.refactor()
                self.cull(0)
        self.refactor()
        if generalize:
            self.generalize()
        self.cull(0)
        return self.patterns_collection

    def setup_train(self, string):
        input_length = len(string)
        self.feed_strength_gain = 2 * input_length / required_repeats
        print 'Started training with string length ' + str(input_length)
        char_set = set(string)
        for char in char_set:
            self.patterns_collection[char] = Pop(char)
            self.patterns_collection[char].feed(self.feed_strength_gain)
        return input_length

    def join_pattern(self, first_pattern, second_pattern, found_pattern_feed_ratio):
        new_pattern = first_pattern.unrolled_pattern + second_pattern.unrolled_pattern
        if new_pattern not in self.patterns_collection:
            self.patterns_collection[new_pattern] = Pop(new_pattern)
            self.patterns_collection[new_pattern].set_components(first_pattern, second_pattern)
        self.patterns_collection[new_pattern].feed(self.feed_strength_gain * found_pattern_feed_ratio)
        # Category tasks
        if first_pattern.belongs_to_category is None and second_pattern.belongs_to_category is None:
            return
        first_category = first_pattern.belongs_to_category if first_pattern.belongs_to_category else first_pattern
        second_category = second_pattern.belongs_to_category if second_pattern.belongs_to_category else second_pattern
        self.join_pattern(first_category, second_category, found_pattern_feed_ratio * feed_ratio_parent_category)

    def find_next_pattern(self, long_word):
        """
        Returns the longest pattern in the given word.
        :param long_word: a string
        :return: PoP(), longest pattern from start.
        """
        for j in range(maxlen_word, 0, -1):  # how many chars to look ahead
            current_word = long_word[:j]
            if current_word in self.patterns_collection:
                return self.patterns_collection[current_word]

    def cull(self, limit):
        cull_list = self.cull_child_and_mark_self(limit)
        for cull_key in cull_list:
            self.patterns_collection.pop(cull_key)

    def cull_child_and_mark_self(self, limit):
        cull_list = []
        for key, pop in self.patterns_collection.iteritems():
            pop.decay()
            if pop.first_component:
                if pop.first_component.strength < limit:
                    pop.first_component = None
            if pop.second_component:
                if pop.second_component.strength < limit:
                    pop.second_component = None
            if pop.strength < limit and len(pop.unrolled_pattern) > 1:
                cull_list.append(key)
        return cull_list

    def status(self):
        out_string = ''
        for key, pop in sorted(self.patterns_collection.iteritems(), key=lambda ng: ng[1].strength):
            out_string += pop.__repr__()
        out_string += 'Status of Pattern of patterns with ' + str(len(self.patterns_collection)) + ' pops \n'
        return out_string

    def save_tsv(self, filename):
        save_string = 'pattern, strength, component1, component2, parents\n'
        for key, pop in sorted(self.patterns_collection.iteritems(), key=lambda ng: ng[1].strength):
            save_string += key + '\t' + str(pop.strength) + '\t'
            if pop.first_component:
                save_string += pop.first_component.unrolled_pattern
            save_string += '\t'
            if pop.second_component:
                save_string += pop.second_component.unrolled_pattern
            save_string += '\t'
            for parent_i in pop.first_child_parents:
                save_string += parent_i.unrolled_pattern
                save_string += '\t'
            save_string += '\n'
        with open(filename, mode='w') as file:
            file.write(save_string)
        print 'Saved file ' + filename

    def save_pb(self, filename):
        buf = ProtobufManager.ProtobufManager.PopManager_to_ProtobufPopManager(self)
        ProtobufManager.ProtobufManager.save_protobuf(buf, filename)

    def save_pb_plain(self, filename):
        buf = ProtobufManager.ProtobufManager.PopManager_to_ProtobufPopManager(self)
        ProtobufManager.ProtobufManager.save_protobuf_plain(buf, filename)

    def load_tsv(self, filename):
        limit = None  # doesn't work for now, some patterns will have first parent child which is not loaded
        with open(filename, mode='r') as file:
            all_lines = file.readlines()
            total_lines = len(all_lines)
            if limit:
                start_line = max(0, total_lines - limit)
                lines_to_read = all_lines[1 + start_line:]
            else:
                lines_to_read = all_lines[1:]
            for lines in lines_to_read:
                elements = lines.split('\t')
                key = elements[0]
                self.patterns_collection[key] = Pop(key)
                self.patterns_collection[key].strength = int(elements[1])
            for lines in all_lines[1:]:
                elements = lines.strip('\n').split('\t')
                key = elements[0]
                if elements[2] is not '' and elements[3] is not '':
                    self.set_components_from_string(self.patterns_collection[key], elements[2], elements[3])
                if elements[4] != '':
                    for parent_i in elements[4:]:
                        if parent_i != '' and parent_i in self.patterns_collection:
                            self.patterns_collection[key].first_child_parents.append(self.patterns_collection[parent_i])
        print 'Loaded file ' + filename + ' with number of patterns = ' + str(len(self.patterns_collection))

    def load_pb(self, filename):
        self.patterns_collection = ProtobufManager.ProtobufManager.load_PopManager(filename).patterns_collection

    def load_pb_plain(self, filename):
        buffy = ProtobufManager.ProtobufManager.load_protobuf_plain(filename)
        pm = ProtobufManager.ProtobufManager.protobuf_to_popmanager(buffy)
        self.patterns_collection = pm.patterns_collection

    def predict_next_word(self, input_word):
        start = max(0, len(input_word) - maxlen_word)
        for j in range(start, len(input_word)):
            current_word = input_word[j:]
            if current_word in self.patterns_collection:
                current_pop = self.patterns_collection[current_word]
                words, probabilities = current_pop.get_next_distribution()
                if current_pop.belongs_to_category:
                    category_words, category_probabilities = current_pop.belongs_to_category.get_next_distribution()
                    words = words + category_words
                    probabilities = np.hstack([0.5 * probabilities, 0.5 * category_probabilities])
                if len(words) < 1:
                    continue
                probabilities /= sum(probabilities)
                return np.random.choice(words, p=probabilities)
        print ' nothing after ', input_word
        return ''

    def generate_stream(self, word_length, seed=None):
        print 'Generating stream with word count = ', word_length
        current_pop = np.random.choice(self.patterns_collection.values()) \
            if seed is None or '' else self.find_next_pattern(seed)
        current_word = current_pop.unrolled_pattern
        generated_output = current_word
        for i in range(word_length):
            next_word = self.predict_next_word(generated_output)
            if next_word == '':
                next_word = np.random.choice([pop.unrolled_pattern
                                              for key, pop in self.patterns_collection.iteritems()])
            generated_output += next_word
        return generated_output

    def refactor(self):
        for key, pop in self.patterns_collection.iteritems():
            if pop.first_component and pop.second_component:
                current_components_strength = pop.first_component.strength + pop.second_component.strength
            else:
                current_components_strength = 0
            for i in range(1, len(key) - 1):
                new_first_component = key[:i]
                new_second_component = key[i:]
                if new_first_component in self.patterns_collection:
                    if new_second_component in self.patterns_collection:
                        refactored_components_strength = self.patterns_collection[new_first_component].strength + \
                                                         self.patterns_collection[new_second_component].strength
                        if refactored_components_strength > current_components_strength:
                            self.change_components_string(new_first_component, new_second_component, pop)

    def change_components_string(self, first_string, second_string, pop):
        if first_string not in self.patterns_collection or second_string not in self.patterns_collection:
            raise ValueError('One of these is not in pattern collection {', first_string, ' : ', second_string, '}')
        if pop.first_component:
            old_pop = pop.first_component
            if pop in old_pop.first_child_parents:
                old_pop.first_child_parents.remove(pop)
        pop.set_components(self.patterns_collection[first_string],
                           self.patterns_collection[second_string])

    def fix_first_child_parents(self):
        print 'Fixing incorrect first_child_parents'
        for pop in self.patterns_collection.values():
            for parent_pop in pop.first_child_parents:
                if parent_pop.first_component:
                    if parent_pop.first_component is pop:
                        continue
                print 'Mismatch ', pop.__repr__(), ' and ', parent_pop.__repr__()
                pop.first_child_parents.remove(parent_pop)

    def do_not_generalize(self, first_string, second_string):
        if first_string == second_string:
            return True
        if len(''.join(e for e in first_string if e.isalnum())) < 3:
            return True
        if len(''.join(e for e in second_string if e.isalnum())) < 3:
            return True
        if self.patterns_collection[first_string].is_child(self.patterns_collection[second_string]) or \
                self.patterns_collection[second_string].is_child(self.patterns_collection[first_string]):
            return True
        return False

    def similarity_all(self):
        for key1, pop1 in self.patterns_collection.iteritems():
            for key2, pop2 in self.patterns_collection.iteritems():
                print 'Similarity of ', pop1.unrolled_pattern, ' and ', pop2.unrolled_pattern, ' is ', pop1.similarity(
                    pop2)

    def generalize(self):
        print 'Generalizing ..'
        pops_list = self.patterns_collection.values()
        for pop in pops_list:
            next_to_next = dict()
            for next_pop in pop.next_patterns():
                next_to_next[next_pop.unrolled_pattern] = next_pop.next_patterns()
            for next_key_a, next_list_a in next_to_next.iteritems():
                for next_key_b, next_list_b in next_to_next.iteritems():
                    if self.do_not_generalize(next_key_a, next_key_b):
                        continue
                    same_length = len(set(next_list_a).intersection(next_list_b))
                    passing_length = generalize_intersection_ratio * min(len(next_list_a), len(next_list_b))
                    if same_length > passing_length and same_length > generalize_common_required_count:
                        self.set_similarity(next_key_a, next_key_b)

    def set_similarity(self, first_pattern, second_pattern):
        if first_pattern == second_pattern:
            print first_pattern, ' and ', second_pattern, ' are same!'
            return
        first_pop = self.patterns_collection[first_pattern]
        second_pop = self.patterns_collection[second_pattern]
        if first_pop.has_common_child(second_pop):
            if first_pop.first_component and second_pop.first_component and second_pop.second_component \
                    and second_pop.second_component:
                self.set_similarity(first_pop.first_component.unrolled_pattern,
                                    second_pop.first_component.unrolled_pattern)
                self.set_similarity(first_pop.second_component.unrolled_pattern,
                                    second_pop.second_component.unrolled_pattern)
                return
        print 'Perhaps ', first_pattern, ' and ', second_pattern, ' are similar?'
        new_category_string = 'category with ' + first_pattern + ' and ' + second_pattern
        if new_category_string not in self.patterns_collection:
            new_category = Pop(new_category_string)
            self.patterns_collection[new_category_string] = new_category
            new_category.set_category(self.patterns_collection[first_pattern],
                                      self.patterns_collection[second_pattern])
        self.patterns_collection[new_category_string].feed(self.feed_strength_gain)



if __name__ == '__main__':
    MainFunctions.sanity_check_run()
