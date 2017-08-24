from abc import ABC, abstractmethod

class ActFunc(ABC):
    @abstractmethod
    def act_func(x):
        pass

    @abstractmethod
    def d_func(val):
        pass
