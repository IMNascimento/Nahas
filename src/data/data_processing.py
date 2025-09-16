# data/data_processing.py
from __future__ import annotations

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np
import joblib
import os
import pandas as pd
from typing import Optional, Tuple, Dict, Any
from utils.validation import DataValidator


class DataProcessor:
    """
    Processamento universal para séries temporais com:
      - Geração de janelas deslizantes (windowing)
      - Suporte a multi-passos (steps_ahead)
      - Normalização global (Standard/MinMax/Robust via sklearn) e local (por janela)
      - Split temporal (train/val/test) preservando ordem
      - Persistência de scalers para reprodução
      - Utilitários para alinhar timestamps às janelas

    Parâmetros
    ----------
    window_size : int
        Tamanho da janela deslizante (n amostras passadas como entrada).
    scaler_type : {"standard","minmax","robust"}, padrão="standard"
        Tipo do scaler global para X e y.

    Observações importantes
    -----------------------
    • Normalização GLOBAL:
        - `normalize_global` faz fit nos dados de treino e retorna X/y escalados.
        - `apply_normalization_global` reusa os scalers já ajustados para val/test.
        - `inverse_transform_global` reverte y para o domínio original.

    • Normalização LOCAL:
        - `normalize_local`/`apply_normalization_local` normalizam amostra-a-amostra (janela a janela),
          sem tocar nos scalers globais. Retornam também um `ctx` com estatísticas para
          desfazer `y` via `inverse_transform_local`.

        - Modos suportados para X: "zscore" | "minmax" | "robust"
        - Modos suportados para y: "none" | "relative_last" | "zscore_target" | "minmax_target" | "robust_target"
          * Os modos de y que contêm "target" **exigem** que a feature-alvo esteja dentro de X (via `target_idx`).

    • Timestamps:
        - Use `window_timestamps(...)` para alinhar corretamente um timestamp por amostra,
          seja o timestamp do horizonte previsto ("horizon") ou o da última barra da entrada ("last_input").
        - `create_windows_and_timestamps(...)` facilita ao já retornar X, y e os timestamps alinhados.
    """

    # ======================
    # CONSTRUÇÃO / SCALERS
    # ======================
    def __init__(self, window_size: int, scaler_type: str = "standard") -> None:
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

    # ======================
    # WINDOWING
    # ======================
    def create_windows(
        self,
        data: pd.DataFrame | np.ndarray,
        coluna_alvo: Optional[str] = None,
        steps_ahead: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cria janelas deslizantes (X, y).

        Parâmetros
        ----------
        data : pd.DataFrame | np.ndarray
            Se for DataFrame, deve conter a coluna alvo.
            Se for array 1D, assume que é a série alvo (apenas univariado).
        coluna_alvo : str | None
            Obrigatória quando `data` for DataFrame.
        steps_ahead : int, default=1
            Número de passos futuros previstos.

        Retorna
        -------
        X : (n_samples, window_size, n_features)
        y : (n_samples, steps_ahead)
        """
        DataValidator.validate_integer(steps_ahead, min_value=1)
        if not isinstance(data, (pd.DataFrame, np.ndarray)):
            raise TypeError("Os dados devem ser um DataFrame ou um ndarray.")

        X, y = [], []
        ws = self._window_size

        if isinstance(data, pd.DataFrame):
            assert coluna_alvo is not None, "coluna_alvo deve ser informada para DataFrame!"
            if coluna_alvo not in data.columns:
                raise ValueError(f"Coluna alvo '{coluna_alvo}' não encontrada no DataFrame.")

            target_col = data[coluna_alvo].values
            features = data.drop(columns=[coluna_alvo]).values  # X NÃO inclui o alvo
            for i in range(ws, len(data) - steps_ahead + 1):
                X.append(features[i - ws:i])
                if steps_ahead == 1:
                    y.append(target_col[i + steps_ahead - 1])
                else:
                    y.append(target_col[i:i + steps_ahead])

        elif isinstance(data, np.ndarray) and data.ndim == 1:
            # Caso univariado puro (apenas alvo)
            arr = data
            for i in range(ws, len(arr) - steps_ahead + 1):
                X.append(arr[i - ws:i].reshape(ws, 1))
                if steps_ahead == 1:
                    y.append(arr[i + steps_ahead - 1])
                else:
                    y.append(arr[i:i + steps_ahead])
        else:
            raise ValueError("Para ndarray, apenas 1D é suportado neste método.")

        X, y = np.array(X), np.array(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        print(f"[DEBUG] After windowing: X.shape={X.shape}, y.shape={y.shape}")
        return X, y

    def create_windows_and_timestamps(
        self,
        df: pd.DataFrame,
        coluna_alvo: str,
        *,
        steps_ahead: int = 1,
        timestamp_col: str = "timestamp",
        ts_mode: str = "horizon"  # "horizon" | "last_input"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Versão que já retorna timestamps alinhados 1-para-1 com as amostras.

        • Para `ts_mode="horizon"`: timestamp do horizonte previsto (t + steps_ahead).
        • Para `ts_mode="last_input"`: timestamp da última barra da janela (t).

        IMPORTANTE: `timestamp` NÃO entra em X.
        """
        if timestamp_col not in df.columns:
            raise ValueError(f"Coluna de timestamp '{timestamp_col}' não encontrada no DataFrame.")

        # 1) guarde os timestamps para alinhamento
        ts = df[timestamp_col].values

        # 2) remova 'timestamp' do DF usado para gerar X e y
        df_feat = df.drop(columns=[timestamp_col])

        # 3) gere janelas (X NÃO conterá timestamp)
        X, y = self.create_windows(df_feat, coluna_alvo=coluna_alvo, steps_ahead=steps_ahead)

        # 4) alinhe os timestamps ao modo desejado
        ts_aligned = self.window_timestamps(ts, steps_ahead=steps_ahead, mode=ts_mode)

        # 5) garantia: X precisa ser numérico
        if not np.issubdtype(X.dtype, np.number):
            try:
                X = X.astype(float)
            except Exception as e:
                raise TypeError("As features (X) devem ser numéricas. Verifique colunas com strings/objetos.") from e

        return X, y, ts_aligned

    def window_timestamps(
        self,
        timestamps: np.ndarray,
        *,
        steps_ahead: int,
        mode: str = "horizon"  # "horizon" | "last_input"
    ) -> np.ndarray:
        """
        Retorna um vetor de timestamps alinhado com as amostras (X,y).

        n_samples = len(timestamps) - window_size - steps_ahead + 1

        - "horizon": timestamps[window_size + steps_ahead - 1 :]
        - "last_input": timestamps[window_size - 1 : len(timestamps) - steps_ahead]
        """
        ws = self._window_size
        n = len(timestamps) - ws - steps_ahead + 1
        if n <= 0:
            raise ValueError("Série muito curta para o window_size e steps_ahead informados.")

        if mode == "horizon":
            start = ws + steps_ahead - 1
            out = timestamps[start:]
        elif mode == "last_input":
            start = ws - 1
            end = len(timestamps) - steps_ahead
            out = timestamps[start:end]
        else:
            raise ValueError("mode deve ser 'horizon' ou 'last_input'.")

        if len(out) != n:
            # Defesa contra off-by-one se o usuário passar inputs estranhos
            out = np.array(out[:n])
        return np.array(out)

    # ======================
    # SPLIT TEMPORAL
    # ======================
    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_size: float = 0.7,
        validation_size: float = 0.15
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Divide X e y em conjuntos de treino, validação e teste, preservando a ordem temporal.
        """
        DataValidator.validate_float(train_size, min_value=0.0, max_value=1.0)
        DataValidator.validate_float(validation_size, min_value=0.0, max_value=1.0)

        n_total = len(X)
        n_train = int(n_total * train_size)
        n_val = int(n_total * validation_size)
        # n_test = n_total - n_train - n_val  # (apenas para leitura)

        X_train = X[:n_train]
        y_train = y[:n_train]
        X_val = X[n_train:n_train + n_val]
        y_val = y[n_train:n_train + n_val]
        X_test = X[n_train + n_val:]
        y_test = y[n_train + n_val:]

        print(f"[DEBUG] Split: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    # ======================
    # NORMALIZAÇÃO GLOBAL
    # ======================
    def normalize_global(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit+transform GLOBAL em X e y (apenas treino).
        X: (n, window, features) | y: (n, steps)
        """
        DataValidator.validate_list(list(X.shape), item_type=int, min_length=3)
        DataValidator.validate_list(list(y.shape), item_type=int, min_length=2)

        X_scaled = self._scaler_X.fit_transform(X.reshape(-1, X.shape[2])).reshape(X.shape)
        y_scaled = self._scaler_y.fit_transform(y.reshape(-1, y.shape[-1])).reshape(y.shape)

        if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
            raise ValueError("Dados normalizados (X) contêm valores inválidos (NaN/Inf).")
        if np.isnan(y_scaled).any() or np.isinf(y_scaled).any():
            raise ValueError("Dados normalizados (y) contêm valores inválidos (NaN/Inf).")
        return X_scaled, y_scaled

    def apply_normalization_global(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apenas transform GLOBAL em X e y (validação/teste), usando scalers já ajustados.
        """
        try:
            Xn = self._scaler_X.transform(X.reshape(-1, X.shape[2])).reshape(X.shape)
            yn = self._scaler_y.transform(y.reshape(-1, y.shape[-1])).reshape(y.shape)
            return Xn, yn
        except ValueError as e:
            raise ValueError(f"Erro ao normalizar os dados (global): {e}")

    def inverse_transform_global(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        Reverte a normalização GLOBAL de y (2D/3D com última dim = steps).
        """
        DataValidator.validate_list(list(y_scaled.shape), item_type=int, min_length=2)
        shape = y_scaled.shape
        inv = self._scaler_y.inverse_transform(y_scaled.reshape(-1, shape[-1])).reshape(shape)
        return inv

    # ======================
    # NORMALIZAÇÃO LOCAL
    # ======================
    def normalize_local(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        x_mode: str = "zscore",                 # "zscore" | "minmax" | "robust"
        y_mode: str = "relative_last",          # "none" | "zscore_target" | "minmax_target" | "robust_target" | "relative_last"
        target_idx: int = 0,
        eps: float = 1e-8,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Normaliza por janela (amostra-a-amostra). Não usa os scalers globais.

        IMPORTANTE:
          • Se y_mode ∈ {"relative_last","*_target"}, a feature-alvo PRECISA estar em X
            (no índice `target_idx`). Caso o seu X NÃO inclua o alvo (ex.: você o removeu
            das features), troque para y_mode="none" ou insira o alvo nas features.
        """
        DataValidator.validate_list(list(X.shape), item_type=int, min_length=3)
        DataValidator.validate_list(list(y.shape), item_type=int, min_length=2)
        n, ws, nf = X.shape

        # --- X local ---
        if x_mode == "zscore":
            X_center = X.mean(axis=1, keepdims=True)           # (n,1,features)
            X_scale  = X.std(axis=1, keepdims=True)
            Xn = (X - X_center) / (X_scale + eps)
        elif x_mode == "minmax":
            X_min   = X.min(axis=1, keepdims=True)
            X_range = X.max(axis=1, keepdims=True) - X_min
            X_center, X_scale = X_min, X_range
            Xn = (X - X_center) / (X_scale + eps)
        elif x_mode == "robust":
            q1 = np.quantile(X, 0.25, axis=1, keepdims=True)
            q3 = np.quantile(X, 0.75, axis=1, keepdims=True)
            X_center = np.median(X, axis=1, keepdims=True)
            X_scale  = (q3 - q1)                                  # IQR
            Xn = (X - X_center) / (X_scale + eps)
        else:
            raise ValueError(f"x_mode inválido: {x_mode}")

        # --- y local (com base no alvo dentro da janela) ---
        if y_mode == "none":
            yn = y.copy()
            y_ctx = {"y_mode": "none"}

        elif y_mode == "relative_last":
            if target_idx < 0 or target_idx >= nf:
                raise ValueError("Para y_mode='relative_last', o target_idx deve apontar para a feature alvo dentro de X.")
            ref = X[:, -1, [target_idx]]                          # (n,1)
            yn  = (y - ref) / (np.abs(ref) + eps)
            y_ctx = {"y_mode": "relative_last", "y_ref_last": ref, "eps": eps}

        elif y_mode in ("zscore_target", "minmax_target", "robust_target"):
            if target_idx < 0 or target_idx >= nf:
                raise ValueError(f"Para y_mode='{y_mode}', o target_idx deve apontar para a feature alvo dentro de X.")
            ywin = X[:, :, target_idx]                             # (n,window)
            if y_mode == "zscore_target":
                center = ywin.mean(axis=1, keepdims=True)
                scale  = ywin.std(axis=1, keepdims=True)
            elif y_mode == "minmax_target":
                ymin   = ywin.min(axis=1, keepdims=True)
                ymax   = ywin.max(axis=1, keepdims=True)
                center = ymin
                scale  = ymax - ymin
            else:  # robust_target
                q1 = np.quantile(ywin, 0.25, axis=1, keepdims=True)
                q3 = np.quantile(ywin, 0.75, axis=1, keepdims=True)
                center = np.median(ywin, axis=1, keepdims=True)
                scale  = (q3 - q1)
            yn = (y - center) / (scale + eps)
            y_ctx = {"y_mode": y_mode, "y_center": center, "y_scale": scale, "eps": eps}

        else:
            raise ValueError(f"y_mode inválido: {y_mode}")

        ctx = {
            "mode": "local",
            "x_mode": x_mode,
            "X_center": X_center,
            "X_scale": X_scale,
            "y": y_ctx,
            "target_idx": target_idx,
        }
        return Xn, yn, ctx

    def apply_normalization_local(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Alias de normalize_local (em normalização local, val/test não dependem de fit prévio).
        """
        return self.normalize_local(X, y, **kwargs)

    def inverse_transform_local(self, y_scaled: np.ndarray, ctx: Dict[str, Any]) -> np.ndarray:
        """
        Desfaz a normalização local de y usando o `ctx` produzido por (normalize|apply)_normalization_local.
        """
        DataValidator.validate_list(list(y_scaled.shape), item_type=int, min_length=2)
        y_info = ctx.get("y", {})
        mode = y_info.get("y_mode", "none")
        eps = y_info.get("eps", 1e-8)

        if mode == "none":
            return y_scaled

        if mode == "relative_last":
            ref = y_info["y_ref_last"]                               # (n,1)
            return y_scaled * (np.abs(ref) + eps) + ref

        # modos com center/scale
        center = y_info["y_center"]                                   # (n,1)
        scale  = y_info["y_scale"]                                    # (n,1)
        return y_scaled * (scale + eps) + center

    # ======================
    # WRAPPERS COMPATÍVEIS (mantêm chamadas antigas)
    # ======================
    def normalize(self, X: np.ndarray, y: np.ndarray, *, strategy: str = "global", **kwargs):
        """
        Wrapper compatível:
          • strategy="global" -> normalize_global (RETORNA 2 valores)
          • strategy="local" -> normalize_local (RETORNA 3 valores: Xn, yn, ctx)
        """
        if strategy == "global":
            return self.normalize_global(X, y)
        elif strategy == "local":
            return self.normalize_local(X, y, **kwargs)
        else:
            raise ValueError("strategy deve ser 'global' ou 'local'.")

    def apply_normalization(self, X: np.ndarray, y: np.ndarray, *, strategy: str = "global", **kwargs):
        """
        Wrapper compatível:
          • strategy="global" -> apply_normalization_global (RETORNA 2 valores)
          • strategy="local" -> apply_normalization_local (RETORNA 3 valores: Xn, yn, ctx)
        """
        if strategy == "global":
            return self.apply_normalization_global(X, y)
        elif strategy == "local":
            return self.apply_normalization_local(X, y, **kwargs)
        else:
            raise ValueError("strategy deve ser 'global' ou 'local'.")

    def inverse_transform(self, y_scaled: np.ndarray, *, strategy: str = "global", ctx: Optional[Dict[str, Any]] = None):
        """
        Wrapper compatível:
          • strategy="global" -> inverse_transform_global
          • strategy="local" -> inverse_transform_local (exige `ctx`)
        """
        if strategy == "global":
            return self.inverse_transform_global(y_scaled)
        elif strategy == "local":
            if ctx is None:
                raise ValueError("ctx é obrigatório para strategy='local'.")
            return self.inverse_transform_local(y_scaled, ctx)
        else:
            raise ValueError("strategy deve ser 'global' ou 'local'.")

    # ======================
    # PERSISTÊNCIA SCALERS (GLOBAL)
    # ======================
    def save_scaler(self, dir_path: str) -> None:
        """
        Salva os scalers globais de X e y para reuso posterior.
        """
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(self._scaler_X, os.path.join(dir_path, 'scaler_X.pkl'))
        joblib.dump(self._scaler_y, os.path.join(dir_path, 'scaler_y.pkl'))

    def load_scaler(self, dir_path: str) -> None:
        """
        Carrega os scalers globais de X e y previamente salvos.
        """
        self._scaler_X = joblib.load(os.path.join(dir_path, 'scaler_X.pkl'))
        self._scaler_y = joblib.load(os.path.join(dir_path, 'scaler_y.pkl'))

    # ======================
    # EXTRAS: OUTLIERS / QUARTIS
    # ======================
    def get_outliers_quartis(
        self,
        data: np.ndarray | pd.DataFrame,
        feature_names: Optional[list] = None,
        print_summary: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Retorna índices de outliers por feature com base em IQR (quartis).
        Suporta (n, window, features) ou (n, features) ou DataFrame.
        """
        if isinstance(data, pd.DataFrame):
            arr = data.values
            feature_names = data.columns.tolist() if feature_names is None else feature_names
        else:
            arr = data
            if arr.ndim == 3:
                arr = arr.reshape(-1, arr.shape[2])
            elif arr.ndim != 2:
                raise ValueError("Os dados precisam ser 2D ou 3D (janela deslizante).")

        n_feats = arr.shape[1]
        outliers_dict: Dict[str, np.ndarray] = {}
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

    def normalize_with_quartis(self, data: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Normaliza usando mediana (centro) e IQR (escala) — similar ao RobustScaler, implementado manualmente.
        Suporta (n, features) ou (n, window, features) ou DataFrame.
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
        iqr[iqr == 0] = 1e-9  # evita divisão por zero

        arr_norm = (arr_2d - mediana) / iqr

        if arr.ndim == 3:
            arr_norm = arr_norm.reshape(arr.shape)
        return arr_norm

