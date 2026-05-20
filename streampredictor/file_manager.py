from streampredictor.pop import Pop
from streampredictor import pop_manager


class FileManager():
    """
    Saves and loads the pop manager in different format.
    """

    def __init__(self, pm):
        """
        :type pm: pop_manager.PopManager
        """
        self.pop_manager = pm  # type: pop_manager.PopManager

    def save_tsv(self, filename):
        header = '\t'.join(['pattern', 'strength', 'component1', 'component2', 'parents']) + '\n'
        save_string = header
        for key, pop in sorted(iter(self.pop_manager.pattern_collection.items()), key=lambda ng: ng[1].strength):
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
        print('Saved file ' + filename)

    def load_tsv(self, filename):
        with open(filename, mode='r') as file:
            all_lines = file.readlines()
            data_lines = all_lines[1:]  # skip header
            # first
            self.get_keys_and_strength(data_lines)
            self.connect_pops(data_lines)
        print('Loaded file ' + filename
              + ' with number of patterns = ' + str(len(self.pop_manager.pattern_collection)))

    def connect_pops(self, data_lines):
        for lines in data_lines:
            elements = lines.strip('\n').split('\t')
            key = elements[0]
            current_pop = self.pop_manager.get(key)
            if elements[2] != '' and elements[3] != '':
                first_component = self.pop_manager.get(elements[2])
                second_component = self.pop_manager.get(elements[3])
                current_pop.set_components(first_component, second_component)
            # elements 4 + are parents
            if elements[4] != '':
                for parent_i in elements[4:]:
                    if parent_i != '' and parent_i in self.pop_manager.pattern_collection:
                        parent_pop = self.pop_manager.get(parent_i)
                        current_pop.first_child_parents.append(parent_pop)

    def get_keys_and_strength(self, data_lines):
        for lines in data_lines:
            elements = lines.split('\t')
            key = elements[0]
            self.pop_manager.pattern_collection[key] = Pop(key)
            self.pop_manager.pattern_collection[key].strength = int(elements[1])
