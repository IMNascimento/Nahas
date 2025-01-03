from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib
import os
import pandas as pd
from utils.validation import DataValidator 

class DataProcessor:
    """
    Classe responsável por processar dados para modelos LSTM.
    """

    def __init__(self, window_size: int, feature_range: tuple = (0, 1)):
        """
        Inicializa o processador de dados.

        :param window_size: Tamanho da janela para criar as entradas dos modelos.
        :param feature_range: Intervalo de normalização do MinMaxScaler.
        """
        DataValidator.validate_integer(window_size, min_value=1)
        DataValidator.validate_tuple(feature_range, item_types=[int, float], min_length=2, max_length=2)

        self._window_size = window_size
        self._scaler_X = MinMaxScaler(feature_range=feature_range)
        self._scaler_y = MinMaxScaler(feature_range=feature_range)

    def create_windows(self, data: pd.DataFrame, coluna_alvo: str = None, steps_ahead: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """
        Cria janelas de dados para entrada em modelos, prevendo múltiplas janelas à frente.

        :param data: DataFrame com os dados.
        :param coluna_alvo: Nome da coluna alvo que será prevista.
        :param steps_ahead: Quantidade de passos a serem previstos à frente.
        :return: Arrays X (entradas) e y (saídas com steps_ahead valores).
        """
        DataValidator.validate_integer(steps_ahead, min_value=1)
        if not isinstance(data, (pd.DataFrame, np.ndarray)):
            raise TypeError("Os dados devem ser um DataFrame ou um ndarray.")

        X, y = [], []

        if isinstance(data, pd.DataFrame):
            target_col = data[coluna_alvo].values if coluna_alvo else data.iloc[:, -1].values
            for i in range(self._window_size, len(data) - steps_ahead + 1):
                X.append(data.iloc[i - self._window_size:i].values)
                if steps_ahead == 1:
                    y.append(target_col[i + steps_ahead - 1])
                else:
                    y.append(target_col[i:i + steps_ahead])

        elif len(data.shape) == 1:
            for i in range(self._window_size, len(data) - steps_ahead + 1):
                X.append(data[i - self._window_size:i])
                if steps_ahead == 1:
                    y.append(data[i + steps_ahead - 1])
                else:
                    y.append(data[i:i + steps_ahead])

        X, y = np.array(X), np.array(y)
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)

        return X, y

    def split_data(self, X: np.ndarray, y: np.ndarray, train_size: float = 0.7, validation_size: float = 0.15) -> tuple:
        """
        Divide os dados em conjuntos de treino, validação e teste.

        :param X: Conjunto de entradas.
        :param y: Conjunto de saídas.
        :param train_size: Proporção dos dados de treino.
        :param validation_size: Proporção dos dados de validação.
        :return: Tupla com conjuntos de treino, validação e teste.
        """
        DataValidator.validate_float(train_size, min_value=0.0, max_value=1.0)
        DataValidator.validate_float(validation_size, min_value=0.0, max_value=1.0)

        train_size = int(len(X) * train_size)
        validation_size = int(len(X) * validation_size)

        X_train = X[:train_size]
        X_validation = X[train_size:train_size + validation_size]
        X_test = X[train_size + validation_size:]

        y_train = y[:train_size]
        y_validation = y[train_size:train_size + validation_size]
        y_test = y[train_size + validation_size:]

        return X_train, X_validation, X_test, y_train, y_validation, y_test

    def normalize(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Normaliza os dados de entrada e saída.

        :param X: Conjunto de entradas (3D).
        :param y: Conjunto de saídas.
        :return: Dados normalizados.
        """
        DataValidator.validate_list(list(X.shape), item_type=int, min_length=3)
        DataValidator.validate_list(list(y.shape), item_type=int, min_length=1)

        X_train_scaled = self._scaler_X.fit_transform(X.reshape(-1, X.shape[2]))
        y_train_scaled = self._scaler_y.fit_transform(y.reshape(-1, 1) if len(y.shape) == 1 else y.reshape(-1, y.shape[-1]))

        X_train_scaled = X_train_scaled.reshape(X.shape)
        return X_train_scaled, y_train_scaled

    def apply_normalization(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Aplica a normalização nos dados fornecidos.

        :param X: Conjunto de entradas (3D).
        :param y: Conjunto de saídas.
        :return: Dados normalizados.
        """
        try:
            X_normalized = self._scaler_X.transform(X.reshape(-1, X.shape[2])).reshape(X.shape)
            y_normalized = self._scaler_y.transform(y.reshape(-1, 1) if len(y.shape) == 1 else y.reshape(-1, y.shape[-1]))
            return X_normalized, y_normalized.reshape(y.shape)
        except ValueError as e:
            raise ValueError(f"Erro ao normalizar os dados: {e}")

    def inverse_transform(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        Reverte a normalização dos dados de saída.

        :param y_scaled: Array normalizado.
        :return: Array desnormalizado.
        """
        DataValidator.validate_list(list(y_scaled.shape), item_type=int, min_length=1)
        return self._scaler_y.inverse_transform(y_scaled.reshape(-1, y_scaled.shape[-1]))

    def save_scaler(self, path: str = 'result/scaler/'):
        """
        Salva os scalers ajustados.

        :param path: Caminho para salvar os scalers.
        """
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        joblib.dump(self._scaler_X, os.path.join(directory, 'scaler_X.pkl'))
        joblib.dump(self._scaler_y, os.path.join(directory, 'scaler_y.pkl'))

    def load_scaler(self, path: str = 'result/scaler/'):
        """
        Carrega os scalers salvos.

        :param path: Caminho dos scalers salvos.
        """
        self._scaler_X = joblib.load(os.path.join(path, 'scaler_X.pkl'))
        self._scaler_y = joblib.load(os.path.join(path, 'scaler_y.pkl'))
