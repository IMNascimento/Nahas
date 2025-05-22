from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def train(self, X_train, y_train, X_val, y_val):
        pass

    @abstractmethod
    def save_model(self, model_path: str):
        pass

    @abstractmethod
    def load_model(self, model_path: str):
        pass

    @abstractmethod
    def predict(self, X):
        pass