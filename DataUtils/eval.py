class Eval:
    """
        Eval
    """
    def __init__(self):
        self.predict_num = 0
        self.correct_num = 0
        self.gold_num = 0

        self.precision = 0
        self.recall = 0
        self.fscore = 0

    def clear(self):
        """
        :return:
        """
        self.predict_num = 0
        self.correct_num = 0
        self.gold_num = 0

        self.precision = 0
        self.recall = 0
        self.fscore = 0

    def getFscore(self):
        """
        :return:
        """
        if self.predict_num == 0:
            self.precision = 0
        else:
            self.precision = (self.correct_num / self.predict_num) * 100

        if self.gold_num == 0:
            self.recall = 0
        else:
            self.recall = (self.correct_num / self.gold_num) * 100

        if self.precision + self.recall == 0:
            self.fscore = 0
        else:
            self.fscore = 2 * (self.precision * self.recall) / (self.precision + self.recall)

        return self.precision, self.recall, self.fscore

    def acc(self):
        """
        :return:
        """
        return self.correct_num / self.gold_num
