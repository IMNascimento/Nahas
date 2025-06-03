from models.base.base_trainer import BaseTrainer
from utils.validation import DataValidator
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Bidirectional
from keras.regularizers import l1_l2
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping
import numpy as np
import os


def check_data_stats(X_train, y_train, X_val, y_val):
    arrays = [("X_train", X_train), ("y_train", y_train), ("X_val", X_val), ("y_val", y_val)]
    for nome, arr in arrays:
        arr = np.asarray(arr)
        print(f"--- {nome} ---")
        print(f"Shape: {arr.shape}")
        print(f"Min: {np.min(arr)}, Max: {np.max(arr)}, Mean: {np.mean(arr)}, Std: {np.std(arr)}")
        if np.isnan(arr).any():
            print(f"[ERRO] {nome} contém NaN")
        if np.isinf(arr).any():
            print(f"[ERRO] {nome} contém Inf")
    print("=" * 40)


class KerasLSTMTrainer(BaseTrainer):
    def __init__(
        self,
        input_shape: tuple = None,
        layers_config: list = None,
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
        bidirectional: bool = False,
        output_units: int = 1,
        activation_functions: list = None,
        kernel_initializer: str = "glorot_uniform",
        callbacks: list = None,
        learning_rate: float = 0.001, 
        **kwargs
    ):
        # Salva atributos e aceita expansão futura via kwargs
        self.input_shape = input_shape
        self.layers_config = layers_config
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.loss_fn = loss_fn
        self.metrics = metrics or []
        self.optimizer = optimizer
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.bidirectional = bidirectional
        self.output_units = output_units
        self.activation_functions = activation_functions or ["tanh"] * len(layers_config or [1])
        self.kernel_initializer = kernel_initializer
        self.callbacks = callbacks or []
        self.learning_rate = learning_rate
        self.model = None

        # Validação logo na inicialização
        self.validate_args()

    def validate_args(self):
        DataValidator.validate_tuple(self.input_shape, min_length=2)
        DataValidator.validate_list(self.layers_config, item_type=int, min_length=1)
        DataValidator.validate_float(self.dropout, min_value=0.0, max_value=1.0)
        DataValidator.validate_float(self.recurrent_dropout, min_value=0.0, max_value=1.0)
        DataValidator.validate_integer(self.batch_size, min_value=1)
        DataValidator.validate_integer(self.epochs, min_value=1)
        DataValidator.validate_integer(self.patience, min_value=1)
        DataValidator.validate_string(self.loss_fn)
        DataValidator.validate_list(self.metrics, item_type=str, allow_empty=True)
        DataValidator.validate_string(self.optimizer)
        DataValidator.validate_float(self.l1_reg, min_value=0.0)
        DataValidator.validate_float(self.l2_reg, min_value=0.0)
        DataValidator.validate_boolean(self.bidirectional)
        DataValidator.validate_integer(self.output_units, min_value=1)
        DataValidator.validate_list(self.activation_functions, item_type=str, min_length=len(self.layers_config))
        DataValidator.validate_string(self.kernel_initializer, allow_empty=True)
        DataValidator.validate_float(self.learning_rate, min_value=0.0)

    def build_model(self) -> Sequential:
        model = Sequential()
        for i, units in enumerate(self.layers_config):
            regularizer = l1_l2(l1=self.l1_reg or 0.0, l2=self.l2_reg or 0.0)
            activation = self.activation_functions[i]
            if self.bidirectional:
                model.add(Bidirectional(LSTM(
                    units=units,
                    return_sequences=i != len(self.layers_config) - 1,
                    input_shape=self.input_shape if i == 0 else None,
                    kernel_regularizer=regularizer,
                    dropout=self.dropout,
                    recurrent_dropout=self.recurrent_dropout,
                    activation=activation,
                    kernel_initializer=self.kernel_initializer
                )))
            else:
                model.add(LSTM(
                    units=units,
                    return_sequences=i != len(self.layers_config) - 1,
                    input_shape=self.input_shape if i == 0 else None,
                    kernel_regularizer=regularizer,
                    dropout=self.dropout,
                    recurrent_dropout=self.recurrent_dropout,
                    activation=activation,
                    kernel_initializer=self.kernel_initializer
                ))
        model.add(Dense(self.output_units))
        # Otimizador
        if self.optimizer.lower() == "adam":
            optimizer = Adam(learning_rate=self.learning_rate)
        elif self.optimizer.lower() == "sgd":
            optimizer = SGD(learning_rate=self.learning_rate)
        else:
            raise ValueError(f"Otimizador '{self.optimizer}' não suportado.")
        model.compile(optimizer=optimizer, loss=self.loss_fn, metrics=self.metrics)
        self.model = model
        return model

    def train(self, X_train, y_train, X_val, y_val):
        if self.model is None:
            self.build_model()

        # ===== Checagem de arrays e estatísticas =====
        check_data_stats(X_train, y_train, X_val, y_val)
        # =============================================

        early_stopping = EarlyStopping(monitor="val_loss", patience=self.patience, restore_best_weights=True)
        callbacks = [early_stopping] + self.callbacks
        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        return self.model

    def finetune(self, X_train, y_train, X_val, y_val, config: dict = None):
        """
        Executa fine-tuning do modelo já carregado.
        Se config for passado, atualiza hiperparâmetros.
        """
        if config:
            self.set_hyperparameters(**config)
            self.build_model()  # Rebuild se a arquitetura mudou
        # Treina mais epochs (continua do ponto atual)
        early_stopping = EarlyStopping(monitor="val_loss", patience=self.patience, restore_best_weights=True)
        callbacks = [early_stopping] + self.callbacks
        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        return self.model

    def save_model(self, model_path: str):
        directory = os.path.dirname(model_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.model.save(model_path)

    def load_model(self, model_path: str):
        self.model = load_model(model_path)
        return self.model

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y, metrics: list = None) -> dict:
        # Usa métricas já definidas no compile, pode customizar se quiser
        result = self.model.evaluate(X, y, return_dict=True)
        return result

    def validate_model(self) -> bool:
        try:
            assert self.model is not None
            dummy = np.zeros((1, ) + self.input_shape)
            self.model.predict(dummy)
            return True
        except Exception as e:
            print(f"Erro de validação do modelo: {e}")
            return False

    def set_hyperparameters(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get_hyperparameters(self) -> dict:
        params = {
            'input_shape': self.input_shape,
            'layers_config': self.layers_config,
            'dropout': self.dropout,
            'recurrent_dropout': self.recurrent_dropout,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'patience': self.patience,
            'loss_fn': self.loss_fn,
            'metrics': self.metrics,
            'optimizer': self.optimizer,
            'l1_reg': self.l1_reg,
            'l2_reg': self.l2_reg,
            'bidirectional': self.bidirectional,
            'output_units': self.output_units,
            'activation_functions': self.activation_functions,
            'kernel_initializer': self.kernel_initializer,
            'callbacks': self.callbacks,
            'learning_rate': self.learning_rate
        }
        return params

    def feature_importance(self):
        raise NotImplementedError("Feature importance não implementado para LSTM.")
