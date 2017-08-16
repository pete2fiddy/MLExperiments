from abc import ABC, abstractmethod

class CostFunc(ABC):

    @staticmethod
    @abstractmethod
    def cost_func(x, y):
        pass

    @staticmethod
    @abstractmethod
    def d_func(x, y):
        pass
