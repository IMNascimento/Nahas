from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np
import joblib
import os
import pandas as pd
from utils.validation import DataValidator

class DataProcessor:
    """
    Processamento universal para séries temporais com janela deslizante, remoção automática do alvo,
    suporte a steps_ahead>1 e normalização flexível.

    Returns:
        X: ndarray (n_samples, window_size, n_features)
        y: ndarray (n_samples, steps_ahead)
            * Se steps_ahead=1: shape (n_samples, 1)
            * Se steps_ahead>1: shape (n_samples, steps_ahead)
    """

    def __init__(self, window_size: int, scaler_type: str = "standard"):
        DataValidator.validate_integer(window_size, min_value=1)
        self._window_size = window_size

        # Escolha de scaler
        self._scaler_X, self._scaler_y = self._get_scaler(scaler_type), self._get_scaler(scaler_type)

    def _get_scaler(self, scaler_type: str):
        scalers = {
            "standard": StandardScaler,
            "minmax": MinMaxScaler,
            "robust": RobustScaler,
        }
        if scaler_type not in scalers:
            raise ValueError(f"Scaler '{scaler_type}' não suportado. Opções: {list(scalers.keys())}")
        return scalers[scaler_type]()

    def create_windows(self, data: pd.DataFrame, coluna_alvo: str = None, steps_ahead: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """
        Cria janelas deslizantes de features e targets para previsão de séries temporais.

        Args:
            data: DataFrame com os dados (obrigatório incluir a coluna alvo!)
            coluna_alvo: nome da coluna target
            steps_ahead: número de passos à frente para prever

        Returns:
            X: (n_samples, window_size, n_features)
            y: (n_samples, steps_ahead)   [sempre 2D]
        """
        DataValidator.validate_integer(steps_ahead, min_value=1)
        if not isinstance(data, (pd.DataFrame, np.ndarray)):
            raise TypeError("Os dados devem ser um DataFrame ou um ndarray.")

        X, y = [], []

        if isinstance(data, pd.DataFrame):
            assert coluna_alvo is not None, "coluna_alvo deve ser informada para DataFrame!"
            # Remove coluna alvo das features!
            if coluna_alvo not in data.columns:
                raise ValueError(f"Coluna alvo '{coluna_alvo}' não encontrada no DataFrame.")
            target_col = data[coluna_alvo].values
            features = data.drop(columns=[coluna_alvo])
            for i in range(self._window_size, len(data) - steps_ahead + 1):
                X.append(features.iloc[i - self._window_size:i].values)
                y.append(target_col[i + steps_ahead - 1] if steps_ahead == 1
                         else target_col[i:i + steps_ahead])
        elif isinstance(data, np.ndarray) and len(data.shape) == 1:
            for i in range(self._window_size, len(data) - steps_ahead + 1):
                X.append(data[i - self._window_size:i])
                y.append(data[i + steps_ahead - 1] if steps_ahead == 1
                         else data[i:i + steps_ahead])

        X, y = np.array(X), np.array(y)
        # Ajusta shapes para universalidade
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        print(f"[DEBUG] After windowing: X.shape={X.shape}, y.shape={y.shape}")
        return X, y

    def split_data(self, X: np.ndarray, y: np.ndarray, train_size: float = 0.7, validation_size: float = 0.15) -> tuple:
        """
        Divide X e y em conjuntos de treino, validação e teste (mantendo ordem temporal).
        """
        DataValidator.validate_float(train_size, min_value=0.0, max_value=1.0)
        DataValidator.validate_float(validation_size, min_value=0.0, max_value=1.0)

        n_total = len(X)
        n_train = int(n_total * train_size)
        n_val = int(n_total * validation_size)
        n_test = n_total - n_train - n_val

        X_train = X[:n_train]
        y_train = y[:n_train]
        X_val = X[n_train:n_train + n_val]
        y_val = y[n_train:n_train + n_val]
        X_test = X[n_train + n_val:]
        y_test = y[n_train + n_val:]

        print(f"[DEBUG] Split: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def normalize(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Normaliza X e y (apenas treino). X: (n, window, features), y: (n, steps)
        """
        DataValidator.validate_list(list(X.shape), item_type=int, min_length=3)
        DataValidator.validate_list(list(y.shape), item_type=int, min_length=1)

        X_train_scaled = self._scaler_X.fit_transform(X.reshape(-1, X.shape[2])).reshape(X.shape)
        y_train_scaled = self._scaler_y.fit_transform(y.reshape(-1, y.shape[-1])).reshape(y.shape)

        if np.isnan(X_train_scaled).any() or np.isinf(X_train_scaled).any():
            raise ValueError("Dados normalizados contêm valores inválidos (NaN ou Inf).")
        return X_train_scaled, y_train_scaled

    def apply_normalization(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Aplica scaler já treinado nos dados (validação/teste).
        """
        try:
            X_normalized = self._scaler_X.transform(X.reshape(-1, X.shape[2])).reshape(X.shape)
            y_normalized = self._scaler_y.transform(y.reshape(-1, y.shape[-1])).reshape(y.shape)
            return X_normalized, y_normalized
        except ValueError as e:
            raise ValueError(f"Erro ao normalizar os dados: {e}")

    def inverse_transform(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        Reverte a normalização de y (2D).
        """
        DataValidator.validate_list(list(y_scaled.shape), item_type=int, min_length=1)
        shape = y_scaled.shape
        inv = self._scaler_y.inverse_transform(y_scaled.reshape(-1, shape[-1])).reshape(shape)
        return inv

    def save_scaler(self, dir_path: str):
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(self._scaler_X, os.path.join(dir_path, 'scaler_X.pkl'))
        joblib.dump(self._scaler_y, os.path.join(dir_path, 'scaler_y.pkl'))

    def load_scaler(self, dir_path: str):
        self._scaler_X = joblib.load(os.path.join(dir_path, 'scaler_X.pkl'))
        self._scaler_y = joblib.load(os.path.join(dir_path, 'scaler_y.pkl'))

