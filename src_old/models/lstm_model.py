from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.regularizers import l1_l2
from keras.callbacks import EarlyStopping
import os
from utils.validation import DataValidator 
from keras.optimizers import Adam

class CustomLSTMTrainer:
    """
    Classe para criar e treinar modelos LSTM personalizados.
    """

    def __init__(
        self, 
        input_shape: tuple = None,
        layers_config: list = None,
        dropout: float = None,
        recurrent_dropout: float = None,
        batch_size: int = None,
        epochs: int = None,
        patience: int = None,
        loss_fn: str = "mean_squared_error",
        metrics: list = None,
        optimizer: str = "adam",
        l1_reg: float = None,
        l2_reg: float = None,
        bidirectional: bool = False,
        output_units: int = 1,
        activation_functions: list = None,
        kernel_initializer: str = None,
        callbacks: list = None,
        learning_rate: float = 0.001
    ):
        """
        Inicializa o modelo LSTM personalizado.

        :param input_shape: Dimensões de entrada (timesteps, features).
        :param layers_config: Lista do número de unidades por camada.
        :param dropout: Taxa de dropout.
        :param recurrent_dropout: Dropout para estados ocultos.
        :param batch_size: Tamanho do lote.
        :param epochs: Número de épocas.
        :param patience: Quantidade de épocas para early stopping.
        :param loss_fn: Função de perda.
        :param metrics: Métricas adicionais.
        :param optimizer: Otimizador (ex.: "adam").
        :param l1_reg: Regularização L1.
        :param l2_reg: Regularização L2.
        :param bidirectional: Se a LSTM deve ser bidirecional.
        :param output_units: Número de saídas do modelo.
        :param activation: Função de ativação (ex.: "tanh").
        :param kernel_initializer: Inicializador de pesos.
        :param callbacks: Lista de callbacks adicionais.
        """

        # Mapeia cada parâmetro para seu validador correspondente
        validations = {
            "layers_config": (layers_config, lambda v: DataValidator.validate_list(v, item_type=int, min_length=1)),
            "dropout": (dropout, lambda v: DataValidator.validate_float(v, min_value=0.0, max_value=1.0)),
            "recurrent_dropout": (recurrent_dropout, lambda v: DataValidator.validate_float(v, min_value=0.0, max_value=1.0)),
            "batch_size": (batch_size, lambda v: DataValidator.validate_integer(v, min_value=1)),
            "epochs": (epochs, lambda v: DataValidator.validate_integer(v, min_value=1)),
            "patience": (patience, lambda v: DataValidator.validate_integer(v, min_value=1)),
            "loss_fn": (loss_fn, lambda v: DataValidator.validate_string(v)),
            "metrics": (metrics, lambda v: DataValidator.validate_list(v, item_type=str, allow_empty=True)),
            "optimizer": (optimizer, lambda v: DataValidator.validate_string(v)),
            "l1_reg": (l1_reg, lambda v: DataValidator.validate_float(v, min_value=0.0)),
            "l2_reg": (l2_reg, lambda v: DataValidator.validate_float(v, min_value=0.0)),
            "learning_rate": (learning_rate, lambda v: DataValidator.validate_float(v, min_value=0.0)),
            "bidirectional": (bidirectional, lambda v: DataValidator.validate_boolean(v)),
            "output_units": (output_units, lambda v: DataValidator.validate_integer(v, min_value=1)),
            "activation_functions": (
                activation_functions,
                lambda v: DataValidator.validate_list(v, item_type=str, min_length=len(layers_config))
                if v
                else None,
            ),
            "kernel_initializer": (kernel_initializer, lambda v: DataValidator.validate_string(v, allow_empty=True)),
            "callbacks": (callbacks, lambda v: DataValidator.validate_list(v, allow_empty=True)),
        }    

        self._input_shape = input_shape
        self._layers_config = layers_config
        self._dropout = dropout
        self._recurrent_dropout = recurrent_dropout
        self._batch_size = batch_size
        self._epochs = epochs
        self._patience = patience
        self._loss_fn = loss_fn
        self._metrics = metrics
        self._optimizer = optimizer
        self._l1_reg = l1_reg
        self._l2_reg = l2_reg
        self._bidirectional = bidirectional
        self._output_units = output_units
        self._activation = "tanh"
        self._activation_functions = activation_functions
        self._kernel_initializer = kernel_initializer
        self._callbacks = callbacks
        self._learning_rate = learning_rate

    def build_model(self) -> Sequential:
        """
        Constrói o modelo LSTM baseado nas configurações.

        :return: Modelo Keras compilado.
        """
        model = Sequential()
        for i, units in enumerate(self._layers_config):
            regularizer = None
            if self._l1_reg or self._l2_reg:
                regularizer = l1_l2(l1=self._l1_reg or 0.0, l2=self._l2_reg or 0.0)

            activation_function = (
                self._activation_functions[i] if self._activation_functions and i < len(self._activation_functions)
                else self._activation
            )


            if self._bidirectional:
                model.add(Bidirectional(LSTM(units=units, return_sequences=i != len(self._layers_config) - 1,
                                             input_shape=self._input_shape if i == 0 else None,
                                             kernel_regularizer=regularizer,
                                             dropout=self._dropout or 0.0,
                                             recurrent_dropout=self._recurrent_dropout or 0.0,
                                             activation=activation_function)))
            else:
                model.add(LSTM(units=units, return_sequences=i != len(self._layers_config) - 1,
                               input_shape=self._input_shape if i == 0 else None,
                               kernel_regularizer=regularizer,
                               dropout=self._dropout or 0.0,
                               recurrent_dropout=self._recurrent_dropout or 0.0,
                               activation=activation_function,
                               kernel_initializer=self._kernel_initializer))

        model.add(Dense(units=self._output_units))

        # Configurando o otimizador com learning rate
        if self._optimizer.lower() == "adam":
            optimizer = Adam(learning_rate=self._learning_rate)
        elif self._optimizer.lower() == "sgd":
            from keras.optimizers import SGD
            optimizer = SGD(learning_rate=self._learning_rate)
        else:
            raise ValueError(f"Otimizador '{self._optimizer}' não suportado.")

        model.compile(optimizer=optimizer, loss=self._loss_fn, metrics=self._metrics)
        return model

    def train(self, X_train, y_train, X_val, y_val) -> Sequential:
        """
        Treina o modelo.

        :param X_train: Dados de treino.
        :param y_train: Labels de treino.
        :param X_val: Dados de validação.
        :param y_val: Labels de validação.
        :return: Modelo treinado.
        """
        model = self.build_model()
        early_stopping = EarlyStopping(monitor="val_loss", patience=self._patience, restore_best_weights=True)
        callbacks = [early_stopping] + (self._callbacks or [])

        model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  epochs=self._epochs, batch_size=self._batch_size, callbacks=callbacks)
        return model

    def save_model(self, model: Sequential, model_path: str):
        """
        Salva o modelo em um arquivo.

        :param model: Modelo treinado.
        :param model_path: Caminho para salvar o modelo.
        """
        directory = os.path.dirname(model_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        model.save(model_path)

    def loading_model(self, model_path: str) -> Sequential:
        """
        Carrega um modelo salvo.

        :param model_path: Caminho do modelo salvo.
        :return: Modelo carregado.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado em {model_path}.")
        return load_model(model_path)