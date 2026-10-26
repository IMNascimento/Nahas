# utils/cusum.py
"""
Monitoramento de deterioração do modelo por CUSUM sobre resíduos padronizados
(artigo 5).

Estatística monitorada: z_t = (y_t − ŷ_t) / s_t, em que s_t é a escala que a
normalização usou para inverter a previsão (sob normalização ancorada, a
volatilidade local vezes o nível). O resíduo é adimensional e comparável ao
longo do tempo; em dólares, qualquer limiar fixo dispararia por mudança de
escala do preço, não por deterioração do modelo.

CUSUM bilateral (Page, 1954), com drift k e limiar h:
    S⁺_t = max(0, S⁺_{t−1} + z_t − k)
    S⁻_t = max(0, S⁻_{t−1} − z_t − k)
    alarme quando max(S⁺_t, S⁻_t) > h; após o alarme as somas voltam a zero.

O limiar h é calibrado por simulação sobre os resíduos de validação para um
ARL0 alvo (comprimento médio de corrida sem alarme falso), com bootstrap em
blocos para respeitar a dependência serial.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def standardized_residuals(y: np.ndarray, y_hat: np.ndarray, scale: Optional[np.ndarray] = None) -> np.ndarray:
    """z = (y − ŷ) / s, 1D. Com scale=None, s ≡ 1 (resíduo em unidades do alvo)."""
    e = np.asarray(y, dtype=float).reshape(-1) - np.asarray(y_hat, dtype=float).reshape(-1)
    if scale is None:
        return e
    s = np.asarray(scale, dtype=float).reshape(-1)
    if np.any(s <= 0):
        raise ValueError("Escala deve ser estritamente positiva.")
    return e / s


def cusum_two_sided(z: np.ndarray, k: float = 0.5, h: float = 5.0, reset: bool = True) -> Dict[str, object]:
    """
    CUSUM bilateral. Devolve as trajetórias S⁺ e S⁻, os índices de alarme e
    os comprimentos de corrida entre alarmes. `reset=False` mantém as somas
    após o alarme (útil para visualização).
    """
    z = np.asarray(z, dtype=float).reshape(-1)
    n = len(z)
    sp = np.zeros(n)
    sm = np.zeros(n)
    alarms: List[int] = []
    cp = cm = 0.0
    last = -1
    runs: List[int] = []
    for t in range(n):
        cp = max(0.0, cp + z[t] - k)
        cm = max(0.0, cm - z[t] - k)
        sp[t], sm[t] = cp, cm
        if max(cp, cm) > h:
            alarms.append(t)
            runs.append(t - last)
            last = t
            if reset:
                cp = cm = 0.0
    return {"s_plus": sp, "s_minus": sm, "alarms": alarms, "run_lengths": runs,
            "k": float(k), "h": float(h), "n": n}


def _block_bootstrap(z: np.ndarray, n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Reamostra blocos contíguos de `z` até completar `n` observações."""
    m = len(z)
    if block >= m:
        block = max(1, m // 2)
    starts = rng.integers(0, m - block + 1, size=int(np.ceil(n / block)))
    return np.concatenate([z[s:s + block] for s in starts])[:n]


def _bootstrap_series(z_ref: np.ndarray, n_sim: int, horizon: int, block: int,
                      rng: np.random.Generator) -> np.ndarray:
    """`n_sim` series simuladas (bootstrap em blocos) de `horizon` observacoes cada."""
    return np.stack([_block_bootstrap(z_ref, horizon, block, rng) for _ in range(n_sim)])


def _arl_on_series(series: np.ndarray, k: float, h: float) -> float:
    """ARL0 sobre series ja simuladas; sem alarme conta `horizon` (censura a direita)."""
    horizon = series.shape[1]
    comps = []
    for z in series:
        out = cusum_two_sided(z, k, h, reset=False)
        comps.append(out["alarms"][0] + 1 if out["alarms"] else horizon)
    return float(np.mean(comps))


def average_run_length(z_ref: np.ndarray, k: float, h: float, *, n_sim: int = 200,
                       horizon: int = 20000, block: int = 24,
                       rng: Optional[np.random.Generator] = None) -> float:
    """
    ARL0 estimado: comprimento médio até o primeiro alarme em séries simuladas
    por bootstrap em blocos dos resíduos de referência (sem mudança). Séries
    sem alarme contam `horizon` (censura à direita, estimativa conservadora).
    """
    rng = rng or np.random.default_rng(0)
    z_ref = np.asarray(z_ref, dtype=float).reshape(-1)
    return _arl_on_series(_bootstrap_series(z_ref, n_sim, horizon, block, rng), k, h)


def calibrate_threshold(z_ref: np.ndarray, *, k: float = 0.5, target_arl0: float = 720.0,
                        h_min: float = 0.5, h_max: float = 50.0, tol: float = 0.05,
                        n_sim: int = 200, block: int = 24,
                        rng: Optional[np.random.Generator] = None, max_iter: int = 25) -> Dict[str, float]:
    """
    Bisseção em h para que ARL0(h) ≈ target_arl0 (ex.: 720 horas = 30 dias).
    Os resíduos de referência são centrados e padronizados antes da simulação,
    e a mesma transformação deve ser aplicada em produção (devolvida em `mu`, `sd`).
    As mesmas séries simuladas são usadas em todas as avaliações de h (números
    aleatórios comuns), o que torna ARL0(h) monótona e a bisseção estável.
    `converged` diz se o ARL0 final ficou dentro de `tol` do alvo.
    """
    rng = rng or np.random.default_rng(0)
    z = np.asarray(z_ref, dtype=float).reshape(-1)
    mu, sd = float(z.mean()), float(z.std())
    if sd <= 0:
        raise ValueError("Resíduos de referência sem dispersão.")
    zc = (z - mu) / sd
    horizon = int(max(5 * target_arl0, 2000))
    series = _bootstrap_series(zc, n_sim, horizon, block, rng)
    lo, hi = h_min, h_max
    arl_lo = _arl_on_series(series, k, lo)
    arl_hi = _arl_on_series(series, k, hi)
    if arl_hi < target_arl0:
        return {"h": hi, "arl0": arl_hi, "k": k, "mu": mu, "sd": sd, "converged": False}
    if arl_lo > target_arl0:
        return {"h": lo, "arl0": arl_lo, "k": k, "mu": mu, "sd": sd, "converged": False}
    h, arl = hi, arl_hi
    for _ in range(max_iter):
        h = 0.5 * (lo + hi)
        arl = _arl_on_series(series, k, h)
        if abs(arl - target_arl0) / target_arl0 < tol:
            break
        if arl < target_arl0:
            lo = h
        else:
            hi = h
    converged = bool(abs(arl - target_arl0) / target_arl0 < tol)
    if not converged:
        # ARL0(h) deu um salto entre lo e hi (poucos residuos de referencia): fica
        # com o menor h que atinge o alvo, o lado conservador (menos alarmes falsos).
        h, arl = hi, _arl_on_series(series, k, hi)
    return {"h": float(h), "arl0": float(arl), "k": float(k), "mu": mu, "sd": sd, "converged": converged}


class CusumMonitor:
    """Estado incremental do CUSUM para uso em fluxo (uma observação por vez)."""

    def __init__(self, k: float, h: float, mu: float = 0.0, sd: float = 1.0):
        self.k, self.h, self.mu, self.sd = float(k), float(h), float(mu), float(sd)
        self.s_plus = self.s_minus = 0.0
        self.t = 0
        self.alarms: List[int] = []

    def update(self, z: float) -> bool:
        zc = (float(z) - self.mu) / self.sd
        self.s_plus = max(0.0, self.s_plus + zc - self.k)
        self.s_minus = max(0.0, self.s_minus - zc - self.k)
        self.t += 1
        if max(self.s_plus, self.s_minus) > self.h:
            self.alarms.append(self.t - 1)
            self.s_plus = self.s_minus = 0.0
            return True
        return False

    def update_many(self, zs) -> List[int]:
        """Atualiza com uma sequência; devolve os índices locais que dispararam."""
        return [i for i, z in enumerate(np.asarray(zs, dtype=float).reshape(-1)) if self.update(z)]
