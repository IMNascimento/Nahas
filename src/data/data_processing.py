from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np
import joblib
import os
import pandas as pd
from utils.validation import DataValidator

class DataProcessor:
    """
    Classe para processamento universal de dados de séries temporais, com:
      - Geração de janelas deslizantes (windowing)
      - Suporte a targets multi-passos (steps_ahead)
      - Normalização flexível (StandardScaler, MinMaxScaler ou RobustScaler)
      - Compatibilidade com DataFrame do Pandas ou ndarray do NumPy
      - Métodos para split temporal (train/val/test) preservando a ordem da série
      - Persistência de scalers para reprodução dos experimentos

    Parâmetros
    ----------
    window_size : int
        Tamanho da janela deslizante (número de amostras de histórico usadas para prever o próximo valor/step).
    scaler_type : str, default="standard"
        Tipo de normalização a ser usada. Opções:
        - "standard": z-score (média=0, std=1)
        - "minmax": intervalo [0, 1]
        - "robust": robusto a outliers (mediana/IQR)

    Métodos Principais
    ------------------
    create_windows(data, coluna_alvo, steps_ahead)
        Cria janelas deslizantes de features e targets para previsão (supervisionada) de séries temporais.
    split_data(X, y, train_size, validation_size)
        Divide as amostras em conjuntos de treino, validação e teste, mantendo a ordem temporal.
    normalize(X, y)
        Normaliza X e y usando o scaler escolhido (apenas nos dados de treino).
    apply_normalization(X, y)
        Aplica os scalers ajustados para normalizar conjuntos de validação/teste.
    inverse_transform(y_scaled)
        Reverte a normalização de y para o domínio original.
    save_scaler(dir_path)
        Salva os scalers de X e y para posterior reprodução dos experimentos.
    load_scaler(dir_path)
        Carrega os scalers previamente salvos.

    Uso Típico
    ----------
    processor = DataProcessor(window_size=96, scaler_type="standard")
    X, y = processor.create_windows(df, coluna_alvo="close", steps_ahead=1)
    X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(X, y, train_size=0.7, validation_size=0.15)
    X_train_scaled, y_train_scaled = processor.normalize(X_train, y_train)
    processor.save_scaler("meu_diretorio_scalers")
    # ...
    X_val_scaled, y_val_scaled = processor.apply_normalization(X_val, y_val)
    # ...
    y_pred_real = processor.inverse_transform(y_pred_scaled)
    """

    def __init__(self, window_size: int, scaler_type: str = "standard"):
        """
        Inicializa o DataProcessor.

        Parâmetros
        ----------
        window_size : int
            Tamanho da janela deslizante.
        scaler_type : str, default="standard"
            Tipo do scaler: "standard", "minmax", ou "robust".
        """
        DataValidator.validate_integer(window_size, min_value=1)
        self._window_size = window_size
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

        Parâmetros
        ----------
        data : pd.DataFrame ou np.ndarray
            Dados de entrada. Obrigatório conter a coluna alvo se for DataFrame.
        coluna_alvo : str
            Nome da coluna target (obrigatório se data for DataFrame).
        steps_ahead : int, default=1
            Número de passos futuros a serem previstos (output multi-step).

        Retorna
        -------
        X : np.ndarray (n_samples, window_size, n_features)
        y : np.ndarray (n_samples, steps_ahead)
        """
        DataValidator.validate_integer(steps_ahead, min_value=1)
        if not isinstance(data, (pd.DataFrame, np.ndarray)):
            raise TypeError("Os dados devem ser um DataFrame ou um ndarray.")

        X, y = [], []

        if isinstance(data, pd.DataFrame):
            assert coluna_alvo is not None, "coluna_alvo deve ser informada para DataFrame!"
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
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        print(f"[DEBUG] After windowing: X.shape={X.shape}, y.shape={y.shape}")
        return X, y

    def split_data(self, X: np.ndarray, y: np.ndarray, train_size: float = 0.7, validation_size: float = 0.15) -> tuple:
        """
        Divide X e y em conjuntos de treino, validação e teste, preservando a ordem temporal.

        Parâmetros
        ----------
        X : np.ndarray
            Features.
        y : np.ndarray
            Targets.
        train_size : float, default=0.7
            Fração dos dados para treino.
        validation_size : float, default=0.15
            Fração dos dados para validação.

        Retorna
        -------
        X_train, X_val, X_test, y_train, y_val, y_test : np.ndarray
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

        Parâmetros
        ----------
        X : np.ndarray
            Features de treino.
        y : np.ndarray
            Targets de treino.

        Retorna
        -------
        X_train_scaled, y_train_scaled : np.ndarray
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

        Parâmetros
        ----------
        X : np.ndarray
            Features.
        y : np.ndarray
            Targets.

        Retorna
        -------
        X_normalized, y_normalized : np.ndarray
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

        Parâmetros
        ----------
        y_scaled : np.ndarray
            Targets normalizados.

        Retorna
        -------
        y_original : np.ndarray
            Targets no domínio original.
        """
        DataValidator.validate_list(list(y_scaled.shape), item_type=int, min_length=1)
        shape = y_scaled.shape
        inv = self._scaler_y.inverse_transform(y_scaled.reshape(-1, shape[-1])).reshape(shape)
        return inv

    def save_scaler(self, dir_path: str):
        """
        Salva os scalers de X e y para reuso posterior.

        Parâmetros
        ----------
        dir_path : str
            Diretório onde os scalers serão salvos.
        """
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(self._scaler_X, os.path.join(dir_path, 'scaler_X.pkl'))
        joblib.dump(self._scaler_y, os.path.join(dir_path, 'scaler_y.pkl'))

    def load_scaler(self, dir_path: str):
        """
        Carrega os scalers de X e y previamente salvos.

        Parâmetros
        ----------
        dir_path : str
            Diretório de onde os scalers serão carregados.
        """
        self._scaler_X = joblib.load(os.path.join(dir_path, 'scaler_X.pkl'))
        self._scaler_y = joblib.load(os.path.join(dir_path, 'scaler_y.pkl'))


    def get_outliers_quartis(self, data, feature_names=None, print_summary=True):
        """
        Exibe/retorna os outliers de cada feature com base nos quartis (IQR).

        Parâmetros
        -----------
        data : np.ndarray ou pd.DataFrame
            Dados de entrada. Espera shape (n_samples, window, n_features) ou (n_samples, n_features).
        feature_names : list, default=None
            Nomes das features (opcional).
        print_summary : bool, default=True
            Se True, imprime o resumo dos outliers.

        Retorna
        -------
        outliers_dict : dict
            Dicionário {nome_da_feature: índices_dos_outliers}
        """
        # Suporta janela 3D (n, window, feat) ou 2D
        if isinstance(data, pd.DataFrame):
            arr = data.values
            feature_names = data.columns.tolist() if feature_names is None else feature_names
        else:
            arr = data
            if arr.ndim == 3:
                arr = arr.reshape(-1, arr.shape[2])
            elif arr.ndim == 2:
                arr = arr
            else:
                raise ValueError("Os dados precisam ser 2D ou 3D (janela deslizante)")

        n_feats = arr.shape[1]
        outliers_dict = {}
        feature_names = feature_names if feature_names is not None else [f"feat_{i}" for i in range(n_feats)]
        for i in range(n_feats):
            col = arr[:, i]
            q1 = np.percentile(col, 25)
            q3 = np.percentile(col, 75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = np.where((col < lower) | (col > upper))[0]
            outliers_dict[feature_names[i]] = outliers
            if print_summary:
                print(f"[OUTLIER] {feature_names[i]}: {len(outliers)} outliers (limites: {lower:.2f}, {upper:.2f})")
        return outliers_dict

    def normalize_with_quartis(self, data):
        """
        Normaliza os dados usando mediana (centro) e IQR (escala) — igual ao RobustScaler,
        mas implementado manualmente.

        Parâmetros
        -----------
        data : np.ndarray ou pd.DataFrame
            Dados de entrada. Espera shape (n_samples, n_features) ou (n_samples, window, n_features).

        Retorna
        -------
        data_normalized : np.ndarray
            Dados normalizados.
        """
        if isinstance(data, pd.DataFrame):
            arr = data.values
        else:
            arr = data

        if arr.ndim == 3:
            arr_2d = arr.reshape(-1, arr.shape[2])
        elif arr.ndim == 2:
            arr_2d = arr
        else:
            raise ValueError("Os dados precisam ser 2D ou 3D (janela deslizante)")

        mediana = np.median(arr_2d, axis=0)
        q1 = np.percentile(arr_2d, 25, axis=0)
        q3 = np.percentile(arr_2d, 75, axis=0)
        iqr = q3 - q1

        # Evita divisão por zero (iqr == 0)
        iqr[iqr == 0] = 1e-9

        arr_norm = (arr_2d - mediana) / iqr

        # Retorna no mesmo formato original
        if arr.ndim == 3:
            arr_norm = arr_norm.reshape(arr.shape)
        return arr_norm