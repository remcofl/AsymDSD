from abc import ABC, abstractmethod


class FactoryConfig(ABC):
    @property
    @abstractmethod
    def CLS(self) -> type:
        pass

    def instantiate(self):
        return self.CLS(self)

    def __call__(self, *args, **kwargs):
        return self.instantiate(*args, **kwargs)
