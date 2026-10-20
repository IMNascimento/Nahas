"""
Testes dos diagnósticos de degenerescência (artigo 2).

Rodar de dentro de `src/`:
    ./venv/bin/python -m pytest tests/test_degeneracy_diagnostics.py -q
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.evomsn_normalizer import EvoMSNNormalizer                       # noqa: E402
from utils.degeneracy import (                                             # noqa: E402
    backbone_invariance,
    backbone_invariance_from_csv,
    contribution_ratio,
    degeneracy_summary,
)

RNG = np.random.default_rng(3)


def _dados(n=300, L=48, H=1, C=2):
    t = np.arange(n + L + H)
    serie = 100 + 5 * np.sin(2 * np.pi * t / 24) + np.cumsum(RNG.normal(0, 0.3, len(t)))
    X = np.stack([np.stack([serie[i:i + L], serie[i:i + L] * 1.01], axis=1) for i in range(n)])
    y = np.stack([serie[i + L:i + L + H] for i in range(n)])
    tw = X[:, :, [0]]
    return X, y, tw


def _msn(policy, H):
    return EvoMSNNormalizer(window_size=48, horizon=H, n_features=2, target_idx=0,
                            k_scales=2, predictor_type="linear",
                            short_horizon_policy=policy)


def test_D_separa_sadio_de_degenerado():
    y = 100 + RNG.normal(0, 1, (500, 1))
    assert backbone_invariance(y, y + 1e-8, y) < 1e-9
    assert backbone_invariance(y, y + RNG.normal(0, 0.5, y.shape), y) > 1e-3


def test_r_e_um_quando_a_rede_manda_e_eps2_quando_nao():
    yt = RNG.normal(0, 1, (400, 1))
    phi = 100 + RNG.normal(0, 0.01, (400, 1))
    xi_sadio = np.full((400, 1), 2.0)
    eps = 1e-8
    r_sadio = contribution_ratio(yt, xi_sadio, yt * (xi_sadio + eps) + phi, eps)
    r_degen = contribution_ratio(yt, np.zeros_like(xi_sadio), yt * eps + phi, eps)
    assert 0.9 < r_sadio < 1.1
    assert r_degen < 1e-12


def test_legacy_em_H1_tem_r_nulo_e_window_stats_devolve_a_rede():
    X, y, tw = _dados(H=1)
    r = {}
    for policy in ("legacy", "window_stats"):
        msn = _msn(policy, 1)
        msn.fit(X, y, target_windows=tw)
        Xs, _, ctx = msn.transform(X, None, target_windows=tw)
        y_tilde = RNG.normal(0, 1, (Xs.shape[0], 1))          # "rede" aleatória
        _, y_hat = msn.denorm_and_ensemble(y_tilde, ctx)
        s = degeneracy_summary(msn, y_tilde, ctx, y_hat)
        r[policy] = s["contribution_ratio_ensemble"]
        assert len(s["contribution_ratio_per_scale"]) == 2
        assert s["horizon"] == 1
    assert r["legacy"] < 1e-10, r
    assert r["window_stats"] > 1e-3, r


def test_D_entre_dois_backbones_sob_legacy_e_eps():
    X, y, tw = _dados(H=1)
    msn = _msn("legacy", 1)
    msn.fit(X, y, target_windows=tw)
    Xs, _, ctx = msn.transform(X, None, target_windows=tw)
    _, ya = msn.denorm_and_ensemble(RNG.normal(0, 1, (Xs.shape[0], 1)), ctx)
    _, yb = msn.denorm_and_ensemble(RNG.normal(0, 1, (Xs.shape[0], 1)), ctx)
    assert backbone_invariance(ya, yb, y) < 1e-8


def test_leitura_de_csv_alinha_por_timestamp_e_step(tmp_path):
    ts = pd.date_range("2024-01-01", periods=50, freq="h").astype(str)
    real = 100 + np.arange(50.0)
    a = pd.DataFrame({"timestamp": ts, "step": 1, "real_value": real, "predicted_value": real + 0.5})
    b = pd.DataFrame({"timestamp": ts, "step": 1, "real_value": real, "predicted_value": real - 0.5})
    pa, pb = tmp_path / "a.csv", tmp_path / "b.csv"
    a.to_csv(pa, index=False)
    b.iloc[::-1].to_csv(pb, index=False)                     # ordem invertida
    out = backbone_invariance_from_csv(str(pa), str(pb))
    assert out["n"] == 50
    assert abs(out["D"] - 1.0 / np.mean(real)) < 1e-12


def test_csv_de_testes_diferentes_e_recusado(tmp_path):
    ts = pd.date_range("2024-01-01", periods=10, freq="h").astype(str)
    a = pd.DataFrame({"timestamp": ts, "step": 1, "real_value": 1.0, "predicted_value": 1.0})
    b = pd.DataFrame({"timestamp": ts, "step": 1, "real_value": 2.0, "predicted_value": 1.0})
    a.to_csv(tmp_path / "a.csv", index=False)
    b.to_csv(tmp_path / "b.csv", index=False)
    with pytest.raises(ValueError):
        backbone_invariance_from_csv(str(tmp_path / "a.csv"), str(tmp_path / "b.csv"))
