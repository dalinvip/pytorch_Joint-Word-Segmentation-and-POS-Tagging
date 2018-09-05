# @Author : bamtercelboo
# @Datetime : 2018/9/4 16:57
# @File : state.py
# @Last Modify Time : 2018/9/4 16:57
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  state.py
    FUNCTION : None
"""


class state_instance:
    """
        state_instance
    """
    def __init__(self, inst):
        self.chars = inst.chars
        self.gold = inst.gold
        self.words = []
        self.pos_id = []
        self.pos_labels = []
        self.actions = []

        self.word_hiddens = []
        self.word_cells = []

        self.all_h = []
        self.all_c = []


class state_batch_instance:
    """
        state_batch_instance
    """
    def __init__(self, features, char_features_num):
        self.chars = []
        self.gold = []
        self.words = []
        self.words_startindex = []
        self.pos_id = []
        self.pos_labels = []
        self.actions = []

        self.word_hiddens = []
        self.word_cells = []

        self.all_h = []
        self.all_c = []
        for index in range(features.batch_length):
            self.chars.append(features.inst[index].chars)
            self.gold.append(features.inst[index].gold)

            self.words.append([])
            self.pos_labels.append([])

    def show(self):
        """
        :return:
        """
        print("chars", self.chars)
        print("gold", self.gold)
        print("words", self.words)
        print("pos_id", self.pos_id)
        print("pos_labels", self.pos_labels)
        print("actions", self.actions)
        print("word_hidden", self.word_hiddens)
        print("word_cells", self.word_cells)
        print("all_h", self.all_h)
        print("all_c", self.all_c)