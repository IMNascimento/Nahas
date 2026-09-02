# utils/baselines.py
"""
Baselines ingênuos, diagnósticos de mimetismo e teste de Diebold-Mariano.

Motivação
---------
Em previsão de preço a um passo, a série é fortemente autocorrelacionada e a
solução de menor esforço é reproduzir a última observação. Um modelo que faz
isso apresenta RMSE baixo e R² próximo de 1 sem nenhuma capacidade preditiva.
Este módulo fornece o piso contra o qual qualquer resultado precisa ser lido:

  • `persistence_forecast`     — repete a última observação.
  • `causal_moving_average`    — média móvel causal de k passos.
  • `constant_output_forecast` — o que o modelo preveria se emitisse uma
    constante no espaço normalizado. Sob y_mode='relative_last', a constante
    zero é EXATAMENTE a persistência: prever 0 já é o baseline.
  • `mimicry_diagnostics`      — separa mimetismo de habilidade, comparando
    retorno previsto contra retorno realizado.
  • `diebold_mariano`          — significância da diferença de erro, com
    variância de longo prazo por Newey-West.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Optional

import numpy as np


# ---------------------------------------------------------------- utilitários
def _as_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def _last_observation(target_windows: np.ndarray) -> np.ndarray:
    """Última barra da janela histórica do alvo -> (n, 1)."""
    tw = np.asarray(target_windows, dtype=float)
    if tw.ndim == 3:
        if tw.shape[2] != 1:
            raise ValueError("target_windows 3D deve ter 1 canal (n, window, 1).")
        tw = tw[:, :, 0]
    if tw.ndim != 2:
        raise ValueError("target_windows deve ser (n, window) ou (n, window, 1).")
    return tw[:, [-1]]


# ------------------------------------------------------------------- métricas
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    if yt.shape != yp.shape:
        raise ValueError(f"Shapes incompatíveis: y_true={yt.shape}, y_pred={yp.shape}")
    err = yt - yp
    mse = float(np.mean(err ** 2))
    denom = float(np.sum((yt - yt.mean()) ** 2))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1.0 - np.sum(err ** 2) / denom) if denom > 0 else float("nan"),
    }


# ------------------------------------------------------------------ baselines
def persistence_forecast(target_windows: np.ndarray, steps_ahead: int = 1) -> np.ndarray:
    """ŷ_t = y_{t-1}, repetido ao longo do horizonte. Shape (n, steps_ahead)."""
    last = _last_observation(target_windows)
    return np.repeat(last, steps_ahead, axis=1)


def causal_moving_average(
    target_windows: np.ndarray, k: int = 3, steps_ahead: int = 1
) -> np.ndarray:
    """Média das k últimas observações da janela — estritamente causal."""
    tw = np.asarray(target_windows, dtype=float)
    if tw.ndim == 3:
        tw = tw[:, :, 0]
    if k < 1 or k > tw.shape[1]:
        raise ValueError(f"k={k} inválido para janela de tamanho {tw.shape[1]}.")
    ma = tw[:, -k:].mean(axis=1, keepdims=True)
    return np.repeat(ma, steps_ahead, axis=1)


def constant_output_forecast(
    inverse_fn: Callable[[np.ndarray], np.ndarray],
    n_samples: int,
    steps_ahead: int = 1,
    value: float = 0.0,
) -> np.ndarray:
    """
    Previsão de um "modelo" que emite uma constante no espaço normalizado,
    passada pela MESMA inversão usada nos modelos treinados.

    Sob y_mode='relative_last', `value=0.0` reproduz a persistência exatamente:
    é a prova de que a rede não precisa aprender nada para atingir aquele erro.
    """
    const = np.full((n_samples, steps_ahead), float(value), dtype=float)
    return np.asarray(inverse_fn(const), dtype=float)


# ---------------------------------------------------------- mimetismo x skill
def mimicry_diagnostics(
    y_true: np.ndarray, y_pred: np.ndarray, target_windows: np.ndarray
) -> Dict[str, float]:
    """
    O teste decisivo. Em vez de comparar níveis (dominados pela autocorrelação),
    compara RETORNOS.

    corr_returns ≈ 0  -> o modelo é um emulador de persistência: todo o RMSE vem
                         do nível, não há sinal.
    corr_pred_last ≈ 1 -> a previsão é essencialmente a última observação.
    """
    yt = _as_2d(y_true)[:, [0]]
    yp = _as_2d(y_pred)[:, [0]]
    last = _last_observation(target_windows)[: len(yt)]

    n = min(len(yt), len(yp), len(last))
    yt, yp, last = yt[:n], yp[:n], last[:n]

    ret_true = (yt - last).reshape(-1)
    ret_pred = (yp - last).reshape(-1)

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.std() < 1e-12 or b.std() < 1e-12:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    nz = np.abs(ret_true) > 0
    directional = (
        float(np.mean(np.sign(ret_pred[nz]) == np.sign(ret_true[nz]))) if nz.any() else float("nan")
    )

    return {
        "corr_pred_vs_last_obs": _corr(yp.reshape(-1), last.reshape(-1)),
        "corr_returns": _corr(ret_pred, ret_true),
        "directional_accuracy": directional,
        "mean_abs_predicted_return": float(np.mean(np.abs(ret_pred))),
        "mean_abs_realized_return": float(np.mean(np.abs(ret_true))),
        "bet_size_ratio": float(
            np.mean(np.abs(ret_pred)) / (np.mean(np.abs(ret_true)) + 1e-12)
        ),
        "n": int(n),
    }


# --------------------------------------------------------- Diebold-Mariano
def _newey_west_lrv(d: np.ndarray, lag: int) -> float:
    """Variância de longo prazo com kernel de Bartlett (Newey-West)."""
    n = len(d)
    dc = d - d.mean()
    gamma0 = float(np.dot(dc, dc) / n)
    lrv = gamma0
    for j in range(1, lag + 1):
        gamma_j = float(np.dot(dc[j:], dc[:-j]) / n)
        lrv += 2.0 * (1.0 - j / (lag + 1.0)) * gamma_j
    return lrv


def diebold_mariano(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    h: int = 1,
    power: int = 2,
    lag: Optional[int] = None,
    harvey_correction: bool = True,
) -> Dict[str, float]:
    """
    Testa se o modelo A tem erro diferente do modelo B (normalmente B = persistência).

    Convenção: d_t = L(e_A) - L(e_B).
      DM > 0  -> A tem erro MAIOR que B (A é pior).
      DM < 0  -> A é melhor.
    |DM| > 1.96 rejeita a hipótese de igualdade a 5%.

    `lag=None` usa a regra automática floor(4*(n/100)^(2/9)), com piso h-1,
    respeitando a dependência serial de previsões em janela deslizante.
    """
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    ea = yt - np.asarray(y_pred_a, dtype=float).reshape(-1)
    eb = yt - np.asarray(y_pred_b, dtype=float).reshape(-1)

    d = np.abs(ea) ** power - np.abs(eb) ** power
    n = len(d)
    if n < 10:
        raise ValueError(f"Amostra pequena demais para DM: n={n}.")

    if lag is None:
        lag = max(h - 1, int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lag = max(0, min(int(lag), n - 2))

    lrv = _newey_west_lrv(d, lag)
    if lrv <= 0:
        return {"dm_stat": float("nan"), "p_value": float("nan"), "mean_loss_diff": float(d.mean()),
                "lag": lag, "n": n, "harvey_correction": harvey_correction}

    dm = float(d.mean() / math.sqrt(lrv / n))

    if harvey_correction:
        k = math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        dm *= k

    p = math.erfc(abs(dm) / math.sqrt(2.0))  # bicaudal, aproximação normal
    return {
        "dm_stat": dm,
        "p_value": float(p),
        "mean_loss_diff": float(d.mean()),
        "lag": lag,
        "n": n,
        "harvey_correction": harvey_correction,
    }


# ------------------------------------------------------------------ wrapper
def evaluate_against_baselines(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_windows: np.ndarray,
    steps_ahead: int = 1,
    ma_k: int = 3,
    inverse_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Dict[str, Dict]:
    """
    Avalia o modelo contra persistência, média móvel causal e — quando
    `inverse_fn` é fornecida — contra a saída constante zero no espaço
    normalizado. Retorna métricas, diagnóstico de mimetismo e testes DM.
    """
    yt = _as_2d(y_true)
    yp = _as_2d(y_pred)
    n = min(len(yt), len(yp), len(np.asarray(target_windows)))
    yt, yp = yt[:n], yp[:n]
    tw = np.asarray(target_windows)[:n]

    persist = persistence_forecast(tw, steps_ahead)[:, : yt.shape[1]]
    ma = causal_moving_average(tw, ma_k, steps_ahead)[:, : yt.shape[1]]

    out: Dict[str, Dict] = {
        "model": regression_metrics(yt, yp),
        "persistence": regression_metrics(yt, persist),
        f"moving_average_{ma_k}": regression_metrics(yt, ma),
        "mimicry": mimicry_diagnostics(yt, yp, tw),
        "dm_model_vs_persistence": diebold_mariano(
            yt[:, 0], yp[:, 0], persist[:, 0], h=steps_ahead
        ),
        "dm_model_vs_moving_average": diebold_mariano(
            yt[:, 0], yp[:, 0], ma[:, 0], h=steps_ahead
        ),
    }

    if inverse_fn is not None:
        const = constant_output_forecast(inverse_fn, n, steps_ahead, value=0.0)[:, : yt.shape[1]]
        out["constant_zero_normalized"] = regression_metrics(yt, const)
        out["constant_zero_equals_persistence"] = {
            "max_abs_diff": float(np.max(np.abs(const - persist))),
        }

    rmse_m = out["model"]["rmse"]
    rmse_p = out["persistence"]["rmse"]
    out["summary"] = {
        "excess_rmse_over_persistence_pct": float(100.0 * (rmse_m - rmse_p) / rmse_p)
        if rmse_p > 0 else float("nan"),
        "beats_persistence": bool(rmse_m < rmse_p),
    }
    return out
