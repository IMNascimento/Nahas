from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib
import os  
import pandas as pd

class DataProcessor:
    def __init__(self, window_size, feature_range=(0, 1)):
        """
        Inicializa o processador de dados.

        :param window_size: Tamanho da janela para criar as entradas dos modelos.
        :param feature_range: Intervalo de normalização do MinMaxScaler.
        """
        self.window_size = window_size
        self.scaler_X = MinMaxScaler(feature_range=feature_range)  # Scaler para os dados de entrada (X)
        self.scaler_y = MinMaxScaler(feature_range=feature_range)  # Scaler para a coluna alvo (y)

    def create_windows(self, data, coluna_alvo=None, steps_ahead=1):
        """
        Cria janelas de dados para entrada em modelos, prevendo múltiplas janelas à frente.

        :param data: Array com os dados de preços ou valores.
        :param coluna_alvo: Nome da coluna alvo que será prevista.
        :param steps_ahead: Quantidade de janelas a serem previstas à frente.
        :return: Arrays X (entradas) e y (saídas com steps_ahead valores).
        """
        X, y = [], []

        # Se 'data' for um DataFrame (várias colunas)
        if isinstance(data, pd.DataFrame):
            if isinstance(coluna_alvo, str):
                target_col = data[coluna_alvo].values
            else:
                target_col = data.iloc[:, -1].values
            
            # Percorre o DataFrame criando janelas e múltiplos passos à frente
            for i in range(self.window_size, len(data) - steps_ahead + 1):
                X.append(data.iloc[i - self.window_size:i].values)
                # Para `steps_ahead`, pegue apenas o último valor previsto para cada janela
                if steps_ahead == 1:
                    y.append(target_col[i + steps_ahead - 1])  # Prever um valor
                else:
                    y.append(target_col[i:i + steps_ahead])  # Prever múltiplos valores à frente

        elif len(data.shape) == 1:
            for i in range(self.window_size, len(data) - steps_ahead + 1):
                X.append(data[i - self.window_size:i])
                if steps_ahead == 1:
                    y.append(data[i + steps_ahead - 1])  # Prever um valor
                else:
                    y.append(data[i:i + steps_ahead])  # Prever múltiplos valores

        # Converte X e y para numpy arrays
        X, y = np.array(X), np.array(y)
        
        # Se `X` for 2D, ajuste para o formato 3D
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)

        return X, y

    def split_data(self, X, y, train_size=0.7, validation_size=0.15):
        """
        Divide os dados em conjunto de treino, validação e teste.

        :param X: Conjunto de entradas.
        :param y: Conjunto de saídas.
        :param train_size: Proporção dos dados de treino.
        :param validation_size: Proporção dos dados de validação.
        :return: Conjuntos de treino, validação e teste.
        """
        train_size = int(len(X) * train_size)
        validation_size = int(len(X) * validation_size)

        X_train, X_validation, X_test = X[:train_size], X[train_size:train_size + validation_size], X[train_size + validation_size:]
        y_train, y_validation, y_test = y[:train_size], y[train_size:train_size + validation_size], y[train_size + validation_size:]

        return X_train, X_validation, X_test, y_train, y_validation, y_test

    def normalize(self, X, y):
        """
        Normaliza os dados de treino (X e y).
        
        :param X: Conjunto de entradas 3D (samples, timesteps, features).
        :param y: Conjunto de saídas.
        :return: Dados normalizados.
        """
        print(f"Dimensões antes da normalização (X): {X.shape}, (y): {y.shape}")
        
        X_train_scaled = self.scaler_X.fit_transform(X.reshape(-1, X.shape[2]))
        print(f"Dimensões após normalização e flatten de X: {X_train_scaled.shape}")
        
        if len(y.shape) == 1:  # Caso seja um único valor à frente
            y_train_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1))
        else:
            y_train_scaled = self.scaler_y.fit_transform(y.reshape(-1, y.shape[-1]))

        print(f"Dimensões após normalização (y_train_scaled): {y_train_scaled.shape}")

        X_train_scaled = X_train_scaled.reshape(X.shape)
        print(f"Dimensões de X_train_scaled após reshape: {X_train_scaled.shape}")
        return X_train_scaled, y_train_scaled

    def apply_normalization(self, X, y):
        try:
            print(f"Dimensões antes da normalização em apply_normalization (X): {X.shape}, (y): {y.shape}")
            X_normalized = self.scaler_X.transform(X.reshape(-1, X.shape[2]))
            print(f"Dimensões após transform (X_normalized): {X_normalized.shape}")
            
            X_normalized = X_normalized.reshape(X.shape)
            print(f"Dimensões após reshape em apply_normalization (X_normalized): {X_normalized.shape}")
            
            if len(y.shape) == 1:
                y_normalized = self.scaler_y.transform(y.reshape(-1, 1))
            else:
                y_normalized = self.scaler_y.transform(y.reshape(-1, y.shape[-1]))
            
            print(f"Dimensões após transform (y_normalized): {y_normalized.shape}")
            return X_normalized, y_normalized.reshape(y.shape)
        except ValueError as e:
            print(f"Erro em apply_normalization: {e}")
            raise
    
    def reshape_to_original_shape(self, X_scaled, original_shape):
        """
        Redimensiona os dados normalizados de volta para o formato 3D original.
        
        :param X_scaled: Dados normalizados em 2D (samples, timesteps * features).
        :param original_shape: Forma original dos dados em 3D (samples, timesteps, features).
        :return: Dados redimensionados para o formato 3D original.
        """
        return X_scaled.reshape(original_shape[0], original_shape[1], original_shape[2])

    def inverse_transform(self, y_scaled):
        """
        Reverte a normalização dos dados de saída (y) para o formato original.

        Este método lida com arrays normalizados de uma ou duas dimensões,
        restaurando os valores ao intervalo original definido no scaler.

        :param y_scaled: Array normalizado (1D ou 2D) que será desnormalizado.
        :return: Array desnormalizado no mesmo formato dimensional de entrada.
        """
        if not isinstance(y_scaled, np.ndarray):
            raise TypeError("y_scaled deve ser um array numpy.")
        print(f"Dimensões antes de inverse_transform: {y_scaled.shape}")
        y_inversed = self.scaler_y.inverse_transform(y_scaled.reshape(-1, y_scaled.shape[-1]))
        print(f"Dimensões após inverse_transform: {y_inversed.shape}")
        return y_inversed

    def save_scaler(self, path='result/scaler/'):
        """
        Salva o scaler ajustado para uso futuro.
        """
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Diretório {directory} criado.")

        # Salvar os scalers separadamente
        joblib.dump(self.scaler_X, os.path.join(directory, 'scaler_X.pkl'))
        joblib.dump(self.scaler_y, os.path.join(directory, 'scaler_y.pkl'))
        print(f"Scalers salvos em {directory}")

    def load_scaler(self, path='result/scaler/'):
        """
        Carrega os scalers salvos para X e y.
        """
        directory = os.path.dirname(path)
        self.scaler_X = joblib.load(os.path.join(directory, 'scaler_X.pkl'))
        self.scaler_y = joblib.load(os.path.join(directory, 'scaler_y.pkl'))