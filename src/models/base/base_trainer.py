from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    @abstractmethod
    def build_model(self):
        """Monta e inicializa o modelo."""
        pass

    @abstractmethod
    def train(self, X_train, y_train, X_val, y_val):
        """Realiza o treinamento do modelo."""
        pass

    @abstractmethod
    def save_model(self, model_path: str):
        """Salva o modelo no caminho especificado."""
        pass

    @abstractmethod
    def load_model(self, model_path: str):
        """Carrega o modelo salvo do caminho especificado."""
        pass

    @abstractmethod
    def predict(self, X):
        """Faz predição utilizando o modelo carregado."""
        pass

    @abstractmethod
    def finetune(self, X_train, y_train, X_val, y_val, config: dict = None):
        """Executa fine-tuning do modelo atual."""
        pass

    @abstractmethod
    def evaluate(self, X, y, metrics: list = None) -> dict:
        """Avalia o modelo em dados de teste/validação."""
        pass

    @abstractmethod
    def validate_model(self) -> bool:
        """Valida integridade, arquitetura e compatibilidade do modelo."""
        pass

    def set_hyperparameters(self, **kwargs):
        """Atualiza hiperparâmetros dinamicamente."""
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get_hyperparameters(self) -> dict:
        """Retorna hiperparâmetros atuais como dict."""
        return {
            attr: getattr(self, attr)
            for attr in dir(self)
            if not attr.startswith("_") and not callable(getattr(self, attr))
        }

    def feature_importance(self):
        """Retorna a importância das features, se aplicável."""
        raise NotImplementedError("Feature importance não implementado para este modelo.")