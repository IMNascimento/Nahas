from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import os

class CustomLSTMTrainer:
    def __init__(
        self, 
        input_shape,
        layers_config=None,
        dropout=None,
        batch_size=None,
        epochs=None,
        patience=None,
        loss_fn="mean_squared_error",  
        metrics=None                    
    ):
        """
        Inicializa o treinador LSTM personalizado.

        :param input_shape: Shape dos dados de entrada (timesteps, features).
        :param layers_config: Lista de configurações de camadas, cada item sendo o número de unidades.
        :param dropout: Taxa de dropout para as camadas LSTM.
        :param batch_size: Tamanho do lote para o treinamento.
        :param epochs: Número de épocas para o treinamento.
        :param patience: Número de épocas sem melhora na validação antes de parar.
        :param loss_fn: Pode ser uma string reconhecida pelo Keras 
                        (ex. 'mse', 'mae', 'mean_squared_error', etc.) 
                        ou uma função Python customizada (y_true, y_pred) -> escalar.
        :param metrics: Lista de métricas (strings ou funções) a serem usadas na compilação do modelo.
        """
        self.input_shape = input_shape
        self.layers_config = layers_config or [50]
        self.dropout = dropout or 0.2
        self.batch_size = batch_size or 32
        self.epochs = epochs or 50
        self.patience = patience or 5

        self.loss_fn = loss_fn
        self.metrics = metrics if metrics is not None else []  # se None, vira lista vazia

    def build_model(self):
        """
        Constrói o modelo LSTM personalizado baseado nas configurações fornecidas.

        :return: Modelo Keras compilado.
        """
        model = Sequential()

        # Adicionar as camadas LSTM configuradas
        for i, units in enumerate(self.layers_config):
            if i == 0:
                model.add(LSTM(units=units, return_sequences=True, input_shape=self.input_shape))
            elif i == len(self.layers_config) - 1:
                model.add(LSTM(units=units, return_sequences=False))
            else:
                model.add(LSTM(units=units, return_sequences=True))
            model.add(Dropout(self.dropout))

        # Camada de saída
        model.add(Dense(units=1))

        # Compilar o modelo
        # Aqui usamos self.loss_fn e self.metrics
        model.compile(
            optimizer="adam",
            loss=self.loss_fn,
            metrics=self.metrics
        )

        return model

    def train(self, X_train, y_train, X_val, y_val):
        """
        Treina o modelo LSTM com os dados fornecidos.

        :param X_train: Dados de treino.
        :param y_train: Labels de treino.
        :param X_val: Dados de validação.
        :param y_val: Labels de validação.
        :return: Modelo treinado.
        """
        print(f"Dimensões de X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"Dimensões de X_val: {X_val.shape}, y_val: {y_val.shape}")
        model = self.build_model()

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=self.patience,
            restore_best_weights=True
        )

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stopping]
        )

        return model

    def save_model(self, model, model_path="result/models/lstm_model.h5"):
        """
        Salva o modelo treinado.

        :param model: Modelo Keras treinado.
        :param model_path: Caminho para salvar o modelo.
        """
        directory = os.path.dirname(model_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        model.save(model_path)
        print(f"Modelo salvo em: {model_path}")

    def load_model(self, model_path):
        """
        Carrega um modelo salvo.

        :param model_path: Caminho do modelo salvo.
        :return: Modelo carregado.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"O modelo no caminho {model_path} não foi encontrado.")
        return load_model(model_path)