from abc import ABC, abstractmethod

class ActFunc(ABC):
    @abstractmethod
    def act_func(x):
        pass

    @abstractmethod
    def d_func(val):
        pass

    '''
    @staticmethod
    @abstractmethod
    def multi_act_func(X):
        pass
    '''
