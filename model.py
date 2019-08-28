class Model:

    def add_naiveModel(self, naiveModel):
        self.naiveModel = naiveModel

    def add_logisticModel(self, logisticModel):
        self.logisticModel = logisticModel

    def add_word_indices(self, word_indices):
        self.word_indices = word_indices

    def get_naiveModel(self):
        return self.naiveModel

    def get_logisticModel(self):
        return self.logisticModel

    def get_word_indices(self):
        return self.word_indices