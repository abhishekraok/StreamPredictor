import numpy as np
import random


from streampredictor import constants


def combine(first_pop, second_pop):
    """
    Combines two Pop into new Pop

    :type first_pop:  Pop
    :type second_pop:  Pop
    :rtype: Pop
    """
    new_string = first_pop.unrolled_pattern + second_pop.unrolled_pattern
    new_pop = Pop(new_string)
    new_pop.set_components(first_pop, second_pop)
    return new_pop


class Pop:
    def __init__(self, chars):
        self.unrolled_pattern = chars  # The actual characters that make up current pattern.
        self.strength = 1
        self.first_component = None  # type: Pop
        self.second_component = None  # type: Pop

        # Which parent has this Pop as it's first child
        self.first_child_parents = []  # type: list[Pop]

        # which category does this belong to?
        self.belongs_to_category = None  # type: Pop

        # who are the members of this category?
        self.members_of_category = []  # type: list[Pop]

    def set_components(self, first_component, second_component):
        """
        Sets the children.

        :type first_component: Pop
        :type second_component: Pop
        :rtype first_component: None
        """
        if (not first_component) or (not second_component):
            raise Exception("component cannot be None")
        self.first_component = first_component
        first_component.first_child_parents.append(self)
        self.second_component = second_component

    def get_sample(self):
        if len(self.members_of_category) < 1:
            return self.unrolled_pattern
        return random.choice(self.members_of_category).get_sample()

    def feed(self, gain):
        self.strength += min(constants.max_strength, int(gain * constants.feed_ratio[0]))
        if gain > 2:
            if self.first_component:
                self.first_component.feed(int(gain * constants.feed_ratio[1]))
            if self.second_component:
                self.second_component.feed(int(gain * constants.feed_ratio[2]))

    def decay(self, decay_amount=None):
        if decay_amount:
            self.strength -= decay_amount
        else:
            self.strength -= constants.decay_strength_loss

    def __repr__(self):
        out = self.unrolled_pattern + ': strength ' + str(self.strength)
        if self.first_component:
            out += '={ ' + self.first_component.unrolled_pattern + ': '
        if self.second_component:
            out += self.second_component.unrolled_pattern + '}'
        if self.belongs_to_category:
            out += ' # ' + self.belongs_to_category.unrolled_pattern
        return out

    def get_next_smallest_distribution(self):
        """ Gives the list of next predicted words and their associated probabilities.
        :return: List A, B. Where A is a list of strings, B is list of floats.
        """
        next_words = []
        strengths = []
        for parent_i in self.first_child_parents:
            if parent_i.second_component:
                next_pop = parent_i.second_component
                while next_pop.first_component:
                    next_pop = next_pop.first_component
                next_words.append(next_pop.get_sample())
                strengths.append(max(parent_i.strength, 0))
        total = sum(strengths)
        probabilities = np.array([float(i) / total for i in strengths])
        return next_words, probabilities

    def has_common_child(self, other_pop):
        if not (
                            self.first_component and self.second_component and other_pop.first_component and other_pop.second_component):
            return self.unrolled_pattern == other_pop.unrolled_pattern
        return self.first_component.has_common_child(other_pop.first_component) or \
               self.second_component.has_common_child(other_pop.second_component)

    def is_child(self, child):
        if child.unrolled_pattern == self.unrolled_pattern:
            return True
        if not self.second_component:
            return False
        return self.second_component.is_child(child) or self.first_component.is_child(child)

    def next_patterns(self):
        """ Gives the list of patterns that are likely to occur next. """
        if not self.first_child_parents:
            return []
        next_patterns = []
        for parent_i in self.first_child_parents:
            if parent_i.second_component:
                next_patterns.append(parent_i.second_component)
        return next_patterns

    def print_components(self):
        if self.first_component and self.second_component:
            return self.first_component.print_components() + ' ' + self.second_component.print_components()
        return ' ' + self.unrolled_pattern

    def add_parent(self, parent_pop):
        """
        :type parent_pop: Pop
        :rtype: None
        """
        self.first_child_parents.append(parent_pop)
