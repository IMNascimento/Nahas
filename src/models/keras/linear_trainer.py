# models/keras/linear_trainer.py
"""
Backbone linear (artigo 4): um único mapa linear da janela achatada (L x C) para
os H passos do horizonte, sem ativação, sem normalização interna, sem recorrência
e sem atenção. É o controle do desenho fatorial: um modelo que não tem nenhum
mecanismo interno capaz de reescalonar a entrada, de modo que toda a tolerância
ao escalonador observada nele vem da normalização externa. Inspirado na crítica
de Zeng et al. (2023), em que um modelo linear compete com Transformers em
previsão de longo horizonte.
"""
from models.base.base_trainer import BaseTrainer
from utils.validation import DataValidator
from keras.models import Sequential, load_model
from keras.layers import Input, Flatten, Dense
from keras.regularizers import l1_l2
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping
import numpy as np
import os


class KerasLinearTrainer(BaseTrainer):
    def __init__(
        self,
        input_shape: tuple = None,
        batch_size: int = 32,
        epochs: int = 50,
        patience: int = 5,
        loss_fn: str = "mean_squared_error",
        metrics: list = None,
        optimizer: str = "adam",
        l1_reg: float = 0.0,
        l2_reg: float = 0.0,
        output_units: int = 1,
        callbacks: list = None,
        learning_rate: float = 0.001,
        **kwargs,
    ):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.loss_fn = loss_fn
        self.metrics = metrics or []
        self.optimizer = optimizer
        self.l1_reg = l1_reg or 0.0
        self.l2_reg = l2_reg or 0.0
        self.output_units = output_units
        self.callbacks = callbacks or []
        self.learning_rate = learning_rate
        self.model = None
        self.validate_args()

    def validate_args(self):
        DataValidator.validate_tuple(self.input_shape, min_length=2)
        DataValidator.validate_integer(self.batch_size, min_value=1)
        DataValidator.validate_integer(self.epochs, min_value=1)
        DataValidator.validate_integer(self.patience, min_value=1)
        DataValidator.validate_string(self.loss_fn)
        DataValidator.validate_string(self.optimizer)
        DataValidator.validate_float(float(self.l1_reg), min_value=0.0)
        DataValidator.validate_float(float(self.l2_reg), min_value=0.0)
        DataValidator.validate_integer(self.output_units, min_value=1)
        DataValidator.validate_float(self.learning_rate, min_value=0.0)

    def build_model(self) -> Sequential:
        model = Sequential([
            Input(shape=self.input_shape),
            Flatten(),
            Dense(self.output_units, kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg)),
        ])
        if self.optimizer.lower() == "adam":
            optimizer = Adam(learning_rate=self.learning_rate)
        elif self.optimizer.lower() == "sgd":
            optimizer = SGD(learning_rate=self.learning_rate)
        else:
            raise ValueError(f"Otimizador '{self.optimizer}' não suportado.")
        model.compile(optimizer=optimizer, loss=self.loss_fn, metrics=self.metrics)
        self.model = model
        return model

    def _fit(self, X_train, y_train, X_val, y_val):
        early_stopping = EarlyStopping(monitor="val_loss", patience=self.patience, restore_best_weights=True)
        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stopping] + self.callbacks,
            verbose=1,
        )
        return self.model

    def train(self, X_train, y_train, X_val, y_val):
        if self.model is None:
            self.build_model()
        return self._fit(X_train, y_train, X_val, y_val)

    def finetune(self, X_train, y_train, X_val, y_val, config: dict = None):
        if config:
            self.set_hyperparameters(**config)
            self.build_model()
        return self._fit(X_train, y_train, X_val, y_val)

    def save_model(self, model_path: str):
        directory = os.path.dirname(model_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        self.model.save(model_path)

    def load_model(self, model_path: str):
        self.model = load_model(model_path)
        return self.model

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y, metrics: list = None) -> dict:
        return self.model.evaluate(X, y, return_dict=True)

    def validate_model(self) -> bool:
        try:
            assert self.model is not None
            self.model.predict(np.zeros((1,) + tuple(self.input_shape)))
            return True
        except Exception as e:
            print(f"Erro de validação do modelo: {e}")
            return False

    def set_hyperparameters(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get_hyperparameters(self) -> dict:
        return {
            "input_shape": self.input_shape,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "patience": self.patience,
            "loss_fn": self.loss_fn,
            "metrics": self.metrics,
            "optimizer": self.optimizer,
            "l1_reg": self.l1_reg,
            "l2_reg": self.l2_reg,
            "output_units": self.output_units,
            "callbacks": self.callbacks,
            "learning_rate": self.learning_rate,
        }

    def feature_importance(self):
        """Pesos do mapa linear, reorganizados em (L, C, H): a 'importância' de cada
        posição e canal da janela para cada passo do horizonte."""
        if self.model is None:
            raise RuntimeError("Modelo não construído.")
        W = self.model.layers[-1].get_weights()[0]            # (L*C, H)
        L, C = self.input_shape[0], self.input_shape[1]
        return W.reshape(L, C, -1)
