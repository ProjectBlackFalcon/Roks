
available_functions = ['linear']


class LearningRate:
    def __init__(self, _from, to, epochs, function='linear'):
        assert function in available_functions
        self.__from = _from
        self.__to = to
        self.__epochs = epochs
        self.__function = function

    def epoch(self, epoch):
        if self.__function == 'linear':
            return self.__from + ((self.__to - self.__from) / self.__epochs) * epoch

