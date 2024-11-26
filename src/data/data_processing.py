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
        # Ajusta o scaler_X nos dados de treino e normaliza
        X_train_scaled = self.scaler_X.fit_transform(X.reshape(-1, X.shape[2]))  # Flatten para ajustar o scaler
        
        # Normalizar o `y` de acordo com a forma
        if len(y.shape) == 1:  # Caso seja um único valor à frente
            y_train_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1))
        else:
            # Flatten `y` adequadamente para steps_ahead > 1
            y_train_scaled = self.scaler_y.fit_transform(y.reshape(-1, y.shape[-1]))

        # Redimensionar o `X_train_scaled` de volta ao seu formato original
        X_train_scaled = X_train_scaled.reshape(X.shape)

        return X_train_scaled, y_train_scaled

    def apply_normalization(self, X, y):
        """
        Aplica a normalização nos dados de validação ou teste, ajustada com base nos dados de treino.
        :param X: Dados de entrada 3D (samples, timesteps, features).
        :param y: Valores alvo 2D (samples, 1 ou mais para steps_ahead).
        :return: Dados de entrada normalizados (X_normalized) e valores alvo normalizados (y_normalized).
        """
        try:
            # Normalizar `X` com base no ajuste do treino
            X_normalized = self.scaler_X.transform(X.reshape(-1, X.shape[2]))  # Flatten para ajustar o scaler
            X_normalized = X_normalized.reshape(X.shape)  # Restaurar a forma original

            # Normalizar o `y`
            if len(y.shape) == 1:
                y_normalized = self.scaler_y.transform(y.reshape(-1, 1))
            else:
                y_normalized = self.scaler_y.transform(y.reshape(-1, y.shape[-1]))

            return X_normalized, y_normalized.reshape(y.shape)
            
        except ValueError as e:
            raise ValueError(f"O MinMaxScaler foi ajustado com {self.scaler_X.n_features_in_} features, mas o X tem {X.shape[2]} features.")
    
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
        Desfaz a normalização dos dados de saída (y) e ajusta para o formato correto.
        """
        # Verifica se o array é 1D (caso steps_ahead seja 1) e reformata para 2D
        if len(y_scaled.shape) == 1:
            y_scaled = y_scaled.reshape(-1, 1)
        
        # Desfaz a normalização
        y_inversed = self.scaler_y.inverse_transform(y_scaled)
        
        # Se o y_scaled original era 1D, retorna a array como 1D novamente
        if y_inversed.shape[1] == 1:
            return y_inversed.flatten()  # Converte de volta para 1D
        
        return y_inversed

    def save_scaler(self, path='result/scaler/scaler.pkl'):
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

    def load_scaler(self, path='result/scaler/scaler.pkl'):
        """
        Carrega os scalers salvos para X e y.
        """
        directory = os.path.dirname(path)
        self.scaler_X = joblib.load(os.path.join(directory, 'scaler_X.pkl'))
        self.scaler_y = joblib.load(os.path.join(directory, 'scaler_y.pkl'))