# utils/conformal.py
"""
Intervalos de previsão conformais com escala local (artigo 3).

A normalização por janela já calcula, para cada amostra i e passo h, uma escala
causal s_{i,h}(W). O score de não conformidade localmente escalado é

    R_i = |y_i − ŷ_i| / s_i

e o quantil conformal q é o ⌈(n+1)(1−α)⌉-ésimo menor score no conjunto de
calibração (a validação, que o protocolo já separa). O intervalo no teste é
[ŷ − q·s, ŷ + q·s]. Com s ≡ 1 recupera-se o split conformal clássico.

Também implementa o ACI (Adaptive Conformal Inference, Gibbs & Candès 2021):
α_{t+1} = α_t + γ(α − err_t), com err_t = 1 se y_t ficou fora do intervalo.

Tudo opera sobre predições exportadas (CSV com timestamp, step, real_value,
predicted_value, scale): não há treino novo.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd


# ------------------------------------------------------------ escala por amostra
def scale_from_context(inverse_kind: str, ctx, n: int, H: int, processor=None,
                       eps: float = 1e-8) -> np.ndarray:
    """
    Devolve a escala s (n, H) usada na inversão de y, por estratégia:
      • local: ctx["y"] (anchored: (n,H); *_target: (n,1); relative_last: |ref|; none: 1)
      • global: fator em unidades do alvo por unidade normalizada do scaler de y
      • evomsn/evomsn_like: média ponderada (pesos do ensemble) de (ξ̂_k + eps)
    """
    if inverse_kind == "local":
        y = ctx.get("y", {})
        mode = y.get("y_mode", "none")
        e = float(y.get("eps", eps))
        if mode == "none":
            s = np.ones((n, 1))
        elif mode == "relative_last":
            s = np.abs(np.asarray(y["y_ref_last"], dtype=float)) + e
        else:
            s = np.asarray(y["y_scale"], dtype=float) + e
        return np.broadcast_to(s.reshape(n, -1), (n, H)).copy() if s.shape[1] in (1, H) else s[:, :H]

    if inverse_kind == "global":
        sc = getattr(processor, "_scaler_y", None)
        if sc is None:
            raise ValueError("processor com _scaler_y é obrigatório para inverse_kind='global'.")
        if hasattr(sc, "data_range_"):          # MinMaxScaler: 1 unidade = amplitude do treino
            base = np.asarray(sc.data_range_, dtype=float)
        else:                                   # Standard/Robust: scale_ = desvio ou IQR
            base = np.asarray(sc.scale_, dtype=float)
        base = base.reshape(1, -1)
        if base.shape[1] == 1:
            base = np.repeat(base, H, axis=1)
        return np.broadcast_to(base[:, :H], (n, H)).copy()

    if inverse_kind in ("evomsn", "evomsn_like"):
        k = int(ctx["k"])
        xi = np.asarray(ctx["xi_hat_stack"], dtype=float).reshape(k, n, -1).transpose(1, 0, 2)  # (n,k,H)
        w = np.asarray(ctx["weights_stack"], dtype=float).reshape(n, k)
        w = w / (w.sum(axis=1, keepdims=True) + 1e-12)
        return np.sum(w[:, :, None] * (xi + eps), axis=1)[:, :H]

    raise ValueError(f"inverse_kind desconhecido: {inverse_kind!r}")


# ------------------------------------------------------------ split conformal
def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """⌈(n+1)(1−α)⌉-ésimo menor score (Lei et al., 2018). Se exceder n, devolve +inf."""
    s = np.sort(np.asarray(scores, dtype=float).reshape(-1))
    n = len(s)
    if n == 0:
        raise ValueError("Sem scores de calibração.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha deve estar em (0, 1).")
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return float("inf")
    return float(s[k - 1])


def nonconformity(y: np.ndarray, y_hat: np.ndarray, scale: Optional[np.ndarray] = None) -> np.ndarray:
    """|y − ŷ| / s; com scale=None, s ≡ 1 (split conformal clássico)."""
    r = np.abs(np.asarray(y, dtype=float) - np.asarray(y_hat, dtype=float))
    if scale is None:
        return r
    s = np.asarray(scale, dtype=float)
    if np.any(s <= 0):
        raise ValueError("Escala deve ser estritamente positiva.")
    return r / s


def intervals(y_hat: np.ndarray, q: float, scale: Optional[np.ndarray] = None):
    """[ŷ − q s, ŷ + q s]; com scale=None, s ≡ 1."""
    yh = np.asarray(y_hat, dtype=float)
    s = np.ones_like(yh) if scale is None else np.asarray(scale, dtype=float)
    return yh - q * s, yh + q * s


def coverage_and_width(y: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                       level: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Cobertura empírica e largura média; `level` (ex.: preço) normaliza a largura."""
    y = np.asarray(y, dtype=float)
    cob = float(np.mean((y >= lo) & (y <= hi)))
    w = hi - lo
    out = {"coverage": cob, "mean_width": float(np.mean(w)), "n": int(y.size)}
    if level is not None:
        out["mean_relative_width"] = float(np.mean(w / np.abs(np.asarray(level, dtype=float))))
    return out


def interval_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> float:
    """
    Interval score de Gneiting & Raftery (2007) para o intervalo central (1−α):
    (hi − lo) + (2/α)(lo − y)·1[y < lo] + (2/α)(y − hi)·1[y > hi], em média.
    Penaliza largura e falhas de cobertura; menor é melhor.
    """
    y = np.asarray(y, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    w = hi - lo
    pen_lo = (2.0 / alpha) * (lo - y) * (y < lo)
    pen_hi = (2.0 / alpha) * (y - hi) * (y > hi)
    return float(np.mean(w + pen_lo + pen_hi))


def split_conformal(y_cal, y_hat_cal, y_test, y_hat_test, alpha: float,
                    scale_cal=None, scale_test=None, level_test=None) -> Dict[str, object]:
    """Calibra em (y_cal, ŷ_cal[, s_cal]) e avalia em teste. Devolve q, lo, hi e métricas."""
    q = conformal_quantile(nonconformity(y_cal, y_hat_cal, scale_cal), alpha)
    lo, hi = intervals(y_hat_test, q, scale_test)
    return {"q": q, "lo": lo, "hi": hi, "interval_score": interval_score(y_test, lo, hi, alpha),
            **coverage_and_width(y_test, lo, hi, level_test)}


# ------------------------------------------------------------ ACI (online)
def adaptive_conformal(y_cal, y_hat_cal, y_test, y_hat_test, alpha: float, gamma: float = 0.005,
                       scale_cal=None, scale_test=None, level_test=None) -> Dict[str, object]:
    """
    ACI de Gibbs & Candès (2021), uma série por vez (y 1D). O quantil é recomputado
    a cada passo sobre a calibração inicial mais os scores já observados no teste,
    com α_t atualizado por α_{t+1} = α_t + γ(α − err_t). α_t é truncado a (0, 1).
    """
    yc = np.asarray(y_cal, dtype=float).reshape(-1)
    yhc = np.asarray(y_hat_cal, dtype=float).reshape(-1)
    yt = np.asarray(y_test, dtype=float).reshape(-1)
    yht = np.asarray(y_hat_test, dtype=float).reshape(-1)
    sc = None if scale_cal is None else np.asarray(scale_cal, dtype=float).reshape(-1)
    st = None if scale_test is None else np.asarray(scale_test, dtype=float).reshape(-1)

    scores = list(nonconformity(yc, yhc, sc))
    alpha_t = alpha
    lo = np.empty_like(yt)
    hi = np.empty_like(yt)
    alphas = np.empty_like(yt)
    for t in range(len(yt)):
        a = min(max(alpha_t, 1e-6), 1 - 1e-6)
        q = conformal_quantile(np.asarray(scores), a)
        s_t = 1.0 if st is None else st[t]
        lo[t], hi[t] = yht[t] - q * s_t, yht[t] + q * s_t
        err = 0.0 if lo[t] <= yt[t] <= hi[t] else 1.0
        alphas[t] = alpha_t
        alpha_t = alpha_t + gamma * (alpha - err)
        scores.append(nonconformity(yt[t], yht[t], None if st is None else st[t]))
    return {"lo": lo, "hi": hi, "alpha_path": alphas, "gamma": gamma,
            "interval_score": interval_score(yt, lo, hi, alpha),
            **coverage_and_width(yt, lo, hi, level_test)}


# ------------------------------------------------------------ por regime
def coverage_by_regime(y, lo, hi, regime: np.ndarray, level=None) -> pd.DataFrame:
    """Cobertura e largura por rótulo de regime (ex.: tercil de volatilidade)."""
    reg = np.asarray(regime)
    linhas = []
    for r in sorted(set(reg.tolist())):
        m = reg == r
        lv = None if level is None else np.asarray(level)[m]
        linhas.append({"regime": r, **coverage_and_width(np.asarray(y)[m], np.asarray(lo)[m],
                                                           np.asarray(hi)[m], lv)})
    df = pd.DataFrame(linhas)
    df.attrs["coverage_gap"] = float(df["coverage"].max() - df["coverage"].min()) if len(df) else float("nan")
    return df


def volatility_terciles(prices: np.ndarray, window: int = 24) -> np.ndarray:
    """
    Tercil (0, 1, 2) da volatilidade realizada causal: desvio dos `window` log-retornos
    que terminam na barra anterior. Os primeiros `window` valores recebem o tercil da
    primeira vol disponível.
    """
    p = np.asarray(prices, dtype=float).reshape(-1)
    lr = np.diff(np.log(p))
    vol = pd.Series(lr).rolling(window).std().shift(1).to_numpy()   # causal: termina em t−1
    vol = np.concatenate([[np.nan], vol])                          # alinha com p
    first = np.nanmin(vol) if np.isfinite(vol).any() else 0.0
    vol = np.where(np.isfinite(vol), vol, first)
    q1, q2 = np.quantile(vol, [1 / 3, 2 / 3])
    return np.digitize(vol, [q1, q2])


# ------------------------------------------------------------ CSV exportado
def read_predictions_csv(path: str, step: int = 1) -> Dict[str, np.ndarray]:
    """Lê o CSV exportado pelo CSVExporter e devolve y, ŷ e, se existir, scale, para um passo."""
    df = pd.read_csv(path)
    d = df[df["step"] == step]
    out = {"timestamp": d["timestamp"].to_numpy(), "y": d["real_value"].to_numpy(float),
           "y_hat": d["predicted_value"].to_numpy(float)}
    if "scale" in d.columns:
        out["scale"] = d["scale"].to_numpy(float)
    return out
