# data/evomsn_normalizer.py
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression


@dataclass
class EvoMSNMeta:
    periods: List[int]                  # [p1, p2, ..., pk]
    period_freq_idxs: List[int]         # índices de frequência rFFT escolhidos
    k_scales: int
    window_size: int                    # L
    horizon: int                        # H
    n_features: int                     # C
    target_idx: int                     # canal alvo em X (para pesos locais)
    eps: float
    agg: str                            # "fft" | "uniform"
    predictor_type: str                 # "linear" | "mlp"
    # preditores por escala i: ϕ̂ = f_ωi(μ_x), ξ̂ = f_θi(σ_x)
    phi_predictors: List[Optional[object]]
    xi_predictors: List[Optional[object]]


class EvoMSNNormalizer:
    """
    MSN/EvoMSN (normalização/denormalização + ensemble).

    - Seleção de periodicidades por FFT média (global).
    - Slicing com padding por período.
    - Estatísticas por fatia: μ,σ do X e ϕ,ξ do Y.
    - Predição ϕ̂(μ) e ξ̂(σ) com preditor configurável ("linear" ou "mlp").
    - Normalização por fatia → backbone → denormalização por ϕ̂,ξ̂.
    - Ensemble adaptativo com pesos por amplitude local nas freq. globais.
    """

    def __init__(
        self,
        window_size: int,
        horizon: int,
        n_features: int,
        target_idx: int = 0,
        k_scales: int = 4,
        agg: str = "fft",                  # "fft" | "uniform"
        eps: float = 1e-8,
        random_state: int = 42,
        predictor_type: str = "linear",    # <--- NOVO
        hidden: tuple[int, int] = (128, 64),
    ):
        self.L = int(window_size)
        self.H = int(horizon)
        self.C = int(n_features)
        self.target_idx = int(target_idx)
        self.k = int(k_scales)
        self.agg = agg.lower()
        assert self.agg in ("fft", "uniform")
        self.eps = float(eps)
        self.random_state = int(random_state)
        self.hidden = tuple(hidden)
        self.predictor_type = predictor_type.lower()
        assert self.predictor_type in ("linear", "mlp"), "predictor_type deve ser 'linear' ou 'mlp'."
        self.meta: Optional[EvoMSNMeta] = None

    # ---------------------- seleção de períodos (FFT global) ----------------------
    def _select_periods_fft_global(self, X_train: np.ndarray, k: int) -> Tuple[List[int], List[int]]:
        N, L, C = X_train.shape
        amps_accum = np.zeros((L // 2 + 1,), dtype=float)
        count = 0
        for n in range(N):
            Xc = X_train[n] - X_train[n].mean(axis=0, keepdims=True)
            F = np.fft.rfft(Xc, axis=0)          # (L/2+1, C)
            amps = np.abs(F)
            amps[0, :] = 0.0                     # remove DC
            amps_accum += amps.mean(axis=1)
            count += 1
        A = amps_accum / max(count, 1)

        freq_idxs = np.argsort(A)[-k:][::-1]
        freq_idxs = [fi for fi in freq_idxs if fi > 0][:k]
        if len(freq_idxs) < k:
            extras = [i for i in range(1, L // 2 + 1) if i not in freq_idxs]
            freq_idxs += extras[::-1][: (k - len(freq_idxs))]

        periods = [int(np.ceil(self.L / fi)) for fi in freq_idxs]
        periods = [max(2, min(self.L, p)) for p in periods]
        return periods, freq_idxs

    # ---------------------- padding + slicing ----------------------
    @staticmethod
    def _pad_to_multiple(arr: np.ndarray, multiple: int, axis: int = 1) -> np.ndarray:
        T = arr.shape[axis]
        r = T % multiple
        if r == 0:
            return arr
        pad_len = multiple - r
        slc = [slice(None)] * arr.ndim
        slc[axis] = slice(-1, None)
        last = arr[tuple(slc)]
        pad_block = np.repeat(last, pad_len, axis=axis)
        return np.concatenate([arr, pad_block], axis=axis)

    def _slice_X(self, X: np.ndarray, p: int) -> Tuple[np.ndarray, int]:
        Xp = self._pad_to_multiple(X, p, axis=1)
        J = Xp.shape[1] // p
        return Xp.reshape(X.shape[0], J, p, X.shape[2]), J

    def _slice_y(self, y: np.ndarray, p: int) -> Tuple[np.ndarray, int]:
        yp = self._pad_to_multiple(y, p, axis=1)
        S = yp.shape[1] // p
        return yp.reshape(y.shape[0], S, p), S

    # ---------------------- estatísticas por fatia ----------------------
    @staticmethod
    def _stats_X(Xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mu = Xs.mean(axis=2, keepdims=True)
        std = Xs.std(axis=2, keepdims=True)
        return mu, std

    @staticmethod
    def _stats_y(Ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        phi = Ys.mean(axis=2, keepdims=True)
        xi = Ys.std(axis=2, keepdims=True)
        return phi, xi

    # ---------------------- normalização por fatia ----------------------
    def _norm_X(self, Xs: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
        return (Xs - mu) / (std + self.eps)

    # ---------------------- preditores de estatística ----------------------
    def _make_predictor(self):
        if self.predictor_type == "mlp":
            return MLPRegressor(
                hidden_layer_sizes=self.hidden,
                activation="relu",
                random_state=self.random_state,
                max_iter=500,
            )
        # "linear"
        return LinearRegression()

    def _fit_stat_predictors(self, Xs: np.ndarray, Ys: np.ndarray) -> Tuple[object, object]:
        mu, std = self._stats_X(Xs)    # (N,J,1,C)
        phi, xi = self._stats_y(Ys)    # (N,S,1)

        feats_mu = mu.squeeze(2).reshape(mu.shape[0], -1)    # (N, J*C)
        feats_std = std.squeeze(2).reshape(std.shape[0], -1) # (N, J*C)

        y_phi = phi.squeeze(2)  # (N, S)
        y_xi  = xi.squeeze(2)   # (N, S)

        phi_reg = self._make_predictor()
        xi_reg  = self._make_predictor()

        phi_reg.fit(feats_mu, y_phi)
        xi_reg.fit(feats_std, y_xi)
        return phi_reg, xi_reg

    def _predict_future_stats(self, Xs: np.ndarray, phi_reg: object, xi_reg: object) -> Tuple[np.ndarray, np.ndarray]:
        mu, std = self._stats_X(Xs)  # (N,J,1,C)
        feats_mu = mu.squeeze(2).reshape(mu.shape[0], -1)
        feats_std = std.squeeze(2).reshape(std.shape[0], -1)
        phi_hat = phi_reg.predict(feats_mu)[:, :, None]     # (N,S,1)
        xi_hat  = xi_reg.predict(feats_std)[:, :, None]     # (N,S,1)
        xi_hat = np.maximum(xi_hat, 0.0)
        return phi_hat, xi_hat

    # ---------------------- pesos de ensemble ----------------------
    @staticmethod
    def _rfft_amplitudes(x_win: np.ndarray) -> np.ndarray:
        X = x_win - x_win.mean(axis=0, keepdims=True)
        F = np.fft.rfft(X, axis=0)
        A = np.abs(F)
        A[0, :] = 0.0
        return A

    def _weights_for_periods(self, target_windows: np.ndarray, freq_idxs: List[int]) -> np.ndarray:
        if target_windows.ndim == 2:
            target_windows = target_windows[..., None]  # (N,L,1)

        N, L, Cw = target_windows.shape
        w = np.zeros((N, len(freq_idxs)), dtype=float)
        for n in range(N):
            A = self._rfft_amplitudes(target_windows[n])  # (L/2+1, Cw)
            if self.target_idx < Cw:
                amps = A[freq_idxs, self.target_idx]
            else:
                amps = A[freq_idxs, :].mean(axis=1)
            w[n, :] = amps

        wsum = w.sum(axis=1, keepdims=True) + 1e-12
        return w / wsum

    # ---------------------- API pública ----------------------
    def fit(
        self,
        X_train: np.ndarray,      # (N, L, C)
        y_train: np.ndarray,      # (N, H)
        save_path: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert X_train.ndim == 3 and X_train.shape[1] == self.L
        assert y_train.ndim == 2 and y_train.shape[1] == self.H

        periods, freq_idxs = self._select_periods_fft_global(X_train, self.k)

        phi_regs, xi_regs = [], []
        X_concat, y_concat = [], []

        for p in periods:
            Xs, _ = self._slice_X(X_train, p)       # (N,J,p,C)
            Ys, _ = self._slice_y(y_train, p)       # (N,S,p)

            phi_reg, xi_reg = self._fit_stat_predictors(Xs, Ys)
            phi_regs.append(phi_reg)
            xi_regs.append(xi_reg)

            mu, std = self._stats_X(Xs)
            Xn = self._norm_X(Xs, mu, std).reshape(Xs.shape[0], -1, self.C)[:, :self.L, :]

            phi_true, xi_true = self._stats_y(Ys)
            Yn = ((Ys - phi_true) / (xi_true + self.eps)).reshape(Ys.shape[0], -1)[:, :self.H]

            X_concat.append(Xn)
            y_concat.append(Yn)

        X_train_msn = np.concatenate(X_concat, axis=0)
        y_train_msn = np.concatenate(y_concat, axis=0)

        self.meta = EvoMSNMeta(
            periods=periods,
            period_freq_idxs=freq_idxs,
            k_scales=self.k,
            window_size=self.L,
            horizon=self.H,
            n_features=self.C,
            target_idx=self.target_idx,
            eps=self.eps,
            agg=self.agg,
            predictor_type=self.predictor_type,   # <--- persiste
            phi_predictors=phi_regs,
            xi_predictors=xi_regs,
        )

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            joblib.dump(self.meta, os.path.join(save_path, "evomsn_meta.pkl"))

        return X_train_msn, y_train_msn

    def transform(
        self,
        X: np.ndarray,                        # (N, L, C)
        y: Optional[np.ndarray] = None,       # (N, H)
        target_windows: Optional[np.ndarray] = None,  # (N,L,C) ou (N,L,1)
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
        assert self.meta is not None, "Chame fit() antes de transform()."

        periods = self.meta.periods
        freq_idxs = self.meta.period_freq_idxs
        X_stack, y_stack = [], []
        ctx = {"phi_hat_stack": [], "xi_hat_stack": [], "weights_stack": [], "k": len(periods)}

        if target_windows is None:
            target_windows = X[:, :, [self.target_idx]]
        w = self._weights_for_periods(target_windows, freq_idxs) if self.meta.agg == "fft" \
            else np.ones((X.shape[0], len(periods)), dtype=float) / len(periods)

        for i, p in enumerate(periods):
            Xs, _ = self._slice_X(X, p)                 # (N,J,p,C)
            mu, std = self._stats_X(Xs)
            Xn = self._norm_X(Xs, mu, std).reshape(Xs.shape[0], -1, self.C)[:, :self.L, :]
            X_stack.append(Xn)

            phi_hat, xi_hat = self._predict_future_stats(Xs, self.meta.phi_predictors[i], self.meta.xi_predictors[i])
            phi_time = np.repeat(phi_hat, p, axis=2).reshape(X.shape[0], -1)[:, :self.H]
            xi_time  = np.repeat(xi_hat,  p, axis=2).reshape(X.shape[0], -1)[:, :self.H]

            ctx["phi_hat_stack"].append(phi_time)
            ctx["xi_hat_stack"].append(xi_time)

            if y is not None:
                Ys, _ = self._slice_y(y, p)
                phi_true, xi_true = self._stats_y(Ys)
                Yn = ((Ys - phi_true) / (xi_true + self.eps)).reshape(Ys.shape[0], -1)[:, :self.H]
                y_stack.append(Yn)

        X_out = np.concatenate(X_stack, axis=0)  # (N*k, L, C)
        y_out = np.concatenate(y_stack, axis=0) if y is not None else None

        ctx["phi_hat_stack"] = np.concatenate(ctx["phi_hat_stack"], axis=0)  # (N*k, H)
        ctx["xi_hat_stack"]  = np.concatenate(ctx["xi_hat_stack"],  axis=0)  # (N*k, H)

        # (N,k) -> (N*k,) em ordem de concatenação por escala
        ctx["weights_stack"] = w.reshape(-1,)
        return X_out, y_out, ctx

    def denorm_and_ensemble(
        self,
        y_tilde_pred_stack: np.ndarray,    # (N*k, H)
        ctx: Dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        k = ctx["k"]
        phi = ctx["phi_hat_stack"]                    # (N*k, H)
        xi  = ctx["xi_hat_stack"]                     # (N*k, H)

        y_den = y_tilde_pred_stack * (xi + self.eps) + phi  # (N*k, H)

        N = y_den.shape[0] // k
        y_per_scale = y_den.reshape(k, N, -1).transpose(1, 0, 2)  # (N,k,H)

        w = ctx["weights_stack"].reshape(N, k)
        wsum = w.sum(axis=1, keepdims=True) + 1e-12
        wnorm = w / wsum

        y_ensemble = np.sum(wnorm[:, :, None] * y_per_scale, axis=1)  # (N,H)
        return y_per_scale, y_ensemble

    @staticmethod
    def load(meta_dir: str) -> EvoMSNMeta:
        meta = joblib.load(os.path.join(meta_dir, "evomsn_meta.pkl"))
        # compatibilidade com metas antigos (sem predictor_type)
        if not hasattr(meta, "predictor_type"):
            meta.predictor_type = "linear"
        return meta

    @staticmethod
    def load_meta(meta_dir: str) -> EvoMSNMeta:
        return EvoMSNNormalizer.load(meta_dir)


class EvoMSNLikeNormalizer(EvoMSNNormalizer):
    """
    Variante 'like': não treina preditores de ϕ̂ e ξ̂.
    Usa estatística da ÚLTIMA fatia de X como proxy do futuro.
    """
    def _fit_stat_predictors(self, Xs: np.ndarray, Ys: np.ndarray) -> Tuple[object, object]:
        return None, None

    def _predict_future_stats(self, Xs: np.ndarray, phi_reg: object, xi_reg: object) -> Tuple[np.ndarray, np.ndarray]:
        mu, std = self._stats_X(Xs)                # (N,J,1,C)
        last_mu = mu[:, -1:, :, :].mean(axis=3)    # (N,1,1)
        last_std = std[:, -1:, :, :].mean(axis=3)  # (N,1,1)
        return last_mu, np.maximum(last_std, 0.0)



