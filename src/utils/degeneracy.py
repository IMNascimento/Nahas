# utils/degeneracy.py
"""
Diagnósticos da degenerescência de horizonte curto em normalizações por fatia
(EvoMSN, SAN): quando H < período, a dispersão prevista do alvo é ~0 e a saída
da rede some na desnormalização. Dois diagnósticos, ambos computáveis a partir
de predições exportadas:

  • `backbone_invariance`  — D: diferença absoluta média entre as predições de
    dois backbones sob a MESMA normalização, em fração do nível médio do alvo.
    Normalização sadia: D ~ 1e-3 a 1e-2. Degenerada: D ~ eps (1e-8 ou menos).
  • `contribution_ratio`   — r: fração da variância da previsão que vem do termo
    da rede, Var[ỹ (ξ̂ + eps)] / Var[ŷ]. Sadia: r ~ 1. Degenerada: r ~ eps².

`degeneracy_summary` junta os dois ao relatório que o EvoMSNNormalizer já gera
no fit (períodos, escalas afetadas, desvio do alvo normalizado por escala).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def backbone_invariance(y_a: np.ndarray, y_b: np.ndarray, y_ref: np.ndarray) -> float:
    """D = mean|ŷ_a − ŷ_b| / mean|y_ref|. Shapes iguais; y_ref é o alvo real."""
    a = np.asarray(y_a, dtype=float).reshape(-1)
    b = np.asarray(y_b, dtype=float).reshape(-1)
    ref = np.asarray(y_ref, dtype=float).reshape(-1)
    if a.shape != b.shape or a.shape != ref.shape:
        raise ValueError(f"Shapes incompatíveis: {a.shape}, {b.shape}, {ref.shape}.")
    nivel = float(np.mean(np.abs(ref)))
    if nivel == 0.0:
        raise ValueError("Nível médio do alvo é zero; D indefinido.")
    return float(np.mean(np.abs(a - b)) / nivel)


def backbone_invariance_from_csv(path_a: str, path_b: str) -> Dict[str, float]:
    """
    Lê dois CSVs exportados pelo CSVExporter (timestamp, step, real_value,
    predicted_value), alinha por (timestamp, step) e devolve D e o n alinhado.
    """
    cols = ["timestamp", "step", "real_value", "predicted_value"]
    a = pd.read_csv(path_a, usecols=cols)
    b = pd.read_csv(path_b, usecols=cols)
    m = a.merge(b, on=["timestamp", "step"], suffixes=("_a", "_b"))
    if m.empty:
        raise ValueError("Nenhum (timestamp, step) em comum entre os dois CSVs.")
    if not np.allclose(m["real_value_a"], m["real_value_b"]):
        raise ValueError("Os CSVs têm valores reais diferentes: não são o mesmo teste.")
    return {
        "D": backbone_invariance(m["predicted_value_a"], m["predicted_value_b"], m["real_value_a"]),
        "n": int(len(m)),
    }


def contribution_ratio(
    y_tilde: np.ndarray, xi_hat: np.ndarray, y_hat: np.ndarray, eps: float
) -> float:
    """r = Var[ỹ (ξ̂ + eps)] / Var[ŷ]. Entradas com o mesmo shape (N, H) ou (N*k, H)."""
    termo = np.asarray(y_tilde, dtype=float) * (np.asarray(xi_hat, dtype=float) + eps)
    var_hat = float(np.var(np.asarray(y_hat, dtype=float)))
    if var_hat == 0.0:
        return float("nan")
    return float(np.var(termo) / var_hat)


def degeneracy_summary(
    msn,
    y_tilde_stack: np.ndarray,
    ctx: Dict,
    y_ensemble: np.ndarray,
    *,
    y_other_backbone: Optional[np.ndarray] = None,
    y_true: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """
    Resume a degenerescência de uma execução EvoMSN/EvoMSN-like:
      - relatório do fit (períodos, política, escalas afetadas, std do alvo normalizado)
      - r por escala e r do ensemble
      - D, se as predições de outro backbone forem fornecidas
    """
    k = int(ctx["k"])
    xi = np.asarray(ctx["xi_hat_stack"], dtype=float)          # (N*k, H)
    yt = np.asarray(y_tilde_stack, dtype=float).reshape(xi.shape)
    y_per_scale, _ = msn.denorm_and_ensemble(yt, ctx)            # (N, k, H)
    N = y_per_scale.shape[0]
    eps = float(msn.eps)

    r_por_escala = []
    for j in range(k):
        yt_j = yt.reshape(k, N, -1)[j]
        xi_j = xi.reshape(k, N, -1)[j]
        r_por_escala.append(contribution_ratio(yt_j, xi_j, y_per_scale[:, j, :], eps))

    termo_total = (yt * (xi + eps)).reshape(k, N, -1).transpose(1, 0, 2)   # (N,k,H)
    w = np.asarray(ctx["weights_stack"], dtype=float).reshape(N, k)
    w = w / (w.sum(axis=1, keepdims=True) + 1e-12)
    termo_ens = np.sum(w[:, :, None] * termo_total, axis=1)                   # (N,H)
    var_ens = float(np.var(np.asarray(y_ensemble, dtype=float)))
    r_ens = float(np.var(termo_ens) / var_ens) if var_ens > 0 else float("nan")

    out: Dict[str, object] = {
        "fit_report": dict(getattr(msn, "degeneracy_report", {}) or {}),
        "periods": list(getattr(msn.meta, "periods", [])) if getattr(msn, "meta", None) else [],
        "horizon": int(msn.H),
        "eps": eps,
        "xi_hat_mean_per_scale": [float(np.mean(xi.reshape(k, N, -1)[j])) for j in range(k)],
        "contribution_ratio_per_scale": r_por_escala,
        "contribution_ratio_ensemble": r_ens,
    }
    if y_other_backbone is not None and y_true is not None:
        out["backbone_invariance_D"] = backbone_invariance(y_ensemble, y_other_backbone, y_true)
    return out
