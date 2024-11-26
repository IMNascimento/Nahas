from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import os
import joblib
import numpy as np

class CustomLSTMTrainer:
    def __init__(self, input_shape, layers_config=None, dropout=None, batch_size=None, epochs=None, patience=None):
        """
        Inicializa o treinador LSTM personalizado.

        :param input_shape: Shape dos dados de entrada (timesteps, features).
        :param layers_config: Lista de configurações de camadas, cada item sendo o número de unidades.
        :param dropout: Taxa de dropout para as camadas LSTM.
        :param batch_size: Tamanho do lote para o treinamento.
        :param epochs: Número de épocas para o treinamento.
        :param patience: Número de épocas sem melhora na validação antes de parar.
        """
        self.input_shape = input_shape
        self.layers_config = layers_config or [50]  # Lista de unidades por camada, padrão [50]
        self.dropout = dropout or 0.2
        self.batch_size = batch_size or 32
        self.epochs = epochs or 50
        self.patience = patience or 5

    def build_model(self):
        """
        Constrói o modelo LSTM personalizado baseado nas configurações fornecidas.

        :return: Modelo Keras compilado.
        """
        model = Sequential()

        # Adicionar as camadas LSTM configuradas
        for i, units in enumerate(self.layers_config):
            # Se for a última camada ou única camada, não retorna sequência
            return_sequences = i < len(self.layers_config) - 1
            model.add(LSTM(units=units, return_sequences=return_sequences, input_shape=self.input_shape if i == 0 else None))
            model.add(Dropout(self.dropout))

        # Adicionar camada de saída
        model.add(Dense(units=1))  # Previsão de um passo à frente (ajuste para múltiplos passos)

        # Compilar o modelo
        model.compile(optimizer='adam', loss='mean_squared_error')

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
        model = self.build_model()

        # Configurar EarlyStopping para evitar overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)

        # Treinar o modelo
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=self.epochs, batch_size=self.batch_size, callbacks=[early_stopping])

        return model

    def save_model(self, model, model_path='result/models/lstm_model.h5'):
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