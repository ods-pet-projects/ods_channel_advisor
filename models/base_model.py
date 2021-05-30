import abc


class BaseModel(abc.ABC):
    @abc.abstractmethod
    def fit(self, X_text: list, y: list):
        pass

    @abc.abstractmethod
    def predict(self, X_text: list):
        pass
