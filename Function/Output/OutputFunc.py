from abc import ABC, abstractmethod

class OutputFunc(ABC):

    @staticmethod
    @abstractmethod
    def out_func(x):
        pass

    @staticmethod
    @abstractmethod
    def d_func(val):
        pass
