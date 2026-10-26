# models/keras/cnn_lstm_trainer.py
"""
CNN-LSTM (artigo 6): camadas convolucionais 1D antes das recorrentes, no desenho
que se repete na literatura de previsao de preco de criptomoedas (Livieris et
al., 2021; Guo et al., 2021): Conv1D(filtros, kernel, ReLU, causal) -> MaxPool1D
-> LSTM empilhada -> Dense(H). E um dos tres pipelines reimplementados sob o
protocolo do artigo; a normalizacao externa e a mesma dos demais backbones.
"""
from models.base.base_trainer import BaseTrainer
from utils.validation import DataValidator
from keras.models import Sequential, load_model
from keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from keras.regularizers import l1_l2
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping
import numpy as np
import os


class KerasCNNLSTMTrainer(BaseTrainer):
    def __init__(
        self,
        input_shape: tuple = None,
        layers_config: list = None,
        conv_filters: int = 64,
        conv_kernel: int = 3,
        conv_layers: int = 1,
        pool_size: int = 2,
        dropout: float = 0.2,
        recurrent_dropout: float = 0.0,
        batch_size: int = 32,
        epochs: int = 50,
        patience: int = 5,
        loss_fn: str = "mean_squared_error",
        metrics: list = None,
        optimizer: str = "adam",
        l1_reg: float = 0.0,
        l2_reg: float = 0.0,
        output_units: int = 1,
        activation_functions: list = None,
        kernel_initializer: str = "glorot_uniform",
        callbacks: list = None,
        learning_rate: float = 0.001,
        **kwargs,
    ):
        self.input_shape = input_shape
        self.layers_config = layers_config or [128, 64]
        self.conv_filters = int(conv_filters)
        self.conv_kernel = int(conv_kernel)
        self.conv_layers = int(conv_layers)
        self.pool_size = int(pool_size)
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.loss_fn = loss_fn
        self.metrics = metrics or []
        self.optimizer = optimizer
        self.l1_reg = l1_reg or 0.0
        self.l2_reg = l2_reg or 0.0
        self.output_units = output_units
        self.activation_functions = activation_functions or ["tanh"] * len(self.layers_config)
        self.kernel_initializer = kernel_initializer
        self.callbacks = callbacks or []
        self.learning_rate = learning_rate
        self.model = None
        self.validate_args()

    def validate_args(self):
        DataValidator.validate_tuple(self.input_shape, min_length=2)
        DataValidator.validate_list(self.layers_config, item_type=int, min_length=1)
        DataValidator.validate_integer(self.conv_filters, min_value=1)
        DataValidator.validate_integer(self.conv_kernel, min_value=1)
        DataValidator.validate_integer(self.conv_layers, min_value=1)
        DataValidator.validate_integer(self.pool_size, min_value=1)
        DataValidator.validate_float(self.dropout, min_value=0.0, max_value=1.0)
        DataValidator.validate_float(self.recurrent_dropout, min_value=0.0, max_value=1.0)
        DataValidator.validate_integer(self.batch_size, min_value=1)
        DataValidator.validate_integer(self.epochs, min_value=1)
        DataValidator.validate_integer(self.patience, min_value=1)
        DataValidator.validate_string(self.loss_fn)
        DataValidator.validate_string(self.optimizer)
        DataValidator.validate_float(float(self.l1_reg), min_value=0.0)
        DataValidator.validate_float(float(self.l2_reg), min_value=0.0)
        DataValidator.validate_integer(self.output_units, min_value=1)
        DataValidator.validate_list(self.activation_functions, item_type=str, min_length=len(self.layers_config))
        DataValidator.validate_float(self.learning_rate, min_value=0.0)
        if int(self.input_shape[0]) // self.pool_size < 1:
            raise ValueError("pool_size maior que a janela: nenhum passo sobra para a LSTM.")

    def build_model(self) -> Sequential:
        regularizer = l1_l2(l1=self.l1_reg, l2=self.l2_reg)
        model = Sequential([Input(shape=self.input_shape)])
        for _ in range(self.conv_layers):
            model.add(Conv1D(filters=self.conv_filters, kernel_size=self.conv_kernel,
                             padding="causal", activation="relu", kernel_regularizer=regularizer))
        model.add(MaxPooling1D(pool_size=self.pool_size))
        for i, units in enumerate(self.layers_config):
            model.add(LSTM(units=units,
                           return_sequences=i != len(self.layers_config) - 1,
                           kernel_regularizer=regularizer,
                           dropout=self.dropout,
                           recurrent_dropout=self.recurrent_dropout,
                           activation=self.activation_functions[i],
                           kernel_initializer=self.kernel_initializer))
        model.add(Dropout(self.dropout))
        model.add(Dense(self.output_units))
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
            "layers_config": self.layers_config,
            "conv_filters": self.conv_filters,
            "conv_kernel": self.conv_kernel,
            "conv_layers": self.conv_layers,
            "pool_size": self.pool_size,
            "dropout": self.dropout,
            "recurrent_dropout": self.recurrent_dropout,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "patience": self.patience,
            "loss_fn": self.loss_fn,
            "metrics": self.metrics,
            "optimizer": self.optimizer,
            "l1_reg": self.l1_reg,
            "l2_reg": self.l2_reg,
            "output_units": self.output_units,
            "activation_functions": self.activation_functions,
            "kernel_initializer": self.kernel_initializer,
            "callbacks": self.callbacks,
            "learning_rate": self.learning_rate,
        }

    def feature_importance(self):
        raise NotImplementedError("Feature importance não implementado para CNN-LSTM.")
