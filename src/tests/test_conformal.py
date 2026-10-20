"""
Testes dos intervalos conformais com escala local (artigo 3).

Rodar de dentro de `src/`:
    ./venv/bin/python -m pytest tests/test_conformal.py -q
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_processing import DataProcessor                      # noqa: E402
from utils.conformal import (                                       # noqa: E402
    adaptive_conformal,
    conformal_quantile,
    coverage_by_regime,
    interval_score,
    read_predictions_csv,
    scale_from_context,
    split_conformal,
    volatility_terciles,
)
from utils.csv_exporter import CSVExporter                          # noqa: E402

RNG = np.random.default_rng(5)


def test_quantil_conformal_segue_a_regra_do_teto():
    s = np.arange(1, 11, dtype=float)                 # n = 10
    assert conformal_quantile(s, 0.10) == 10.0        # ceil(11*0.9) = 10
    assert conformal_quantile(s, 0.50) == 6.0         # ceil(11*0.5) = 6
    assert conformal_quantile(s, 0.05) == float("inf")  # ceil(11*0.95) = 11 > n


def test_cobertura_marginal_sob_permutabilidade():
    """Scores i.i.d.: cobertura empírica no teste ≈ 1−α (lei dos grandes números)."""
    n_cal, n_test, alpha = 2000, 20000, 0.10
    cob = []
    for seed in range(8):                              # media sobre calibracoes: a garantia e em esperanca
        rng = np.random.default_rng(seed)
        s_cal = np.exp(rng.normal(0, 0.5, n_cal))      # escalas heterocedásticas
        s_test = np.exp(rng.normal(0, 0.5, n_test))
        y_cal = s_cal * rng.standard_t(4, n_cal)
        y_test = s_test * rng.standard_t(4, n_test)
        r = split_conformal(y_cal, np.zeros(n_cal), y_test, np.zeros(n_test), alpha,
                            scale_cal=s_cal, scale_test=s_test)
        cob.append(r["coverage"])
        assert 0.85 < r["coverage"] < 0.95
    assert abs(np.mean(cob) - 0.90) < 0.01


def test_escala_local_tem_cobertura_uniforme_e_global_nao():
    """Dois regimes de dispersão: só o score escalado cobre igual nos dois."""
    n = 6000
    regime = np.repeat([0, 1], n // 2)
    s = np.where(regime == 0, 1.0, 5.0)
    y = s * RNG.normal(0, 1, n)
    y_hat = np.zeros(n)
    cal = slice(0, n, 2)
    tst = slice(1, n, 2)
    glob = split_conformal(y[cal], y_hat[cal], y[tst], y_hat[tst], 0.10)
    loc = split_conformal(y[cal], y_hat[cal], y[tst], y_hat[tst], 0.10,
                          scale_cal=s[cal], scale_test=s[tst])
    gap_g = coverage_by_regime(y[tst], glob["lo"], glob["hi"], regime[tst]).attrs["coverage_gap"]
    gap_l = coverage_by_regime(y[tst], loc["lo"], loc["hi"], regime[tst]).attrs["coverage_gap"]
    assert gap_g > 0.15, gap_g            # global: sobra em calmaria, falta em agitação
    assert gap_l < 0.04, gap_l            # local: quase uniforme
    assert loc["mean_width"] < glob["mean_width"]


def test_aci_corrige_a_cobertura_sob_mudanca_de_regime():
    n_cal, n_test = 500, 3000
    y_cal = RNG.normal(0, 1, n_cal)
    y_test = RNG.normal(0, 3, n_test)                 # teste 3x mais disperso que a calibração
    zeros_c, zeros_t = np.zeros(n_cal), np.zeros(n_test)
    fixo = split_conformal(y_cal, zeros_c, y_test, zeros_t, 0.10)
    aci = adaptive_conformal(y_cal, zeros_c, y_test, zeros_t, 0.10, gamma=0.01)
    assert fixo["coverage"] < 0.70
    assert abs(aci["coverage"] - 0.90) < 0.04
    assert aci["alpha_path"].shape == (n_test,)


def test_escala_do_contexto_local_anchored_e_global(tmp_path):
    ws, H, n = 24, 3, 50
    serie = 100 + np.cumsum(RNG.normal(0, 1, n + ws + H))
    tw = np.stack([serie[i:i + ws] for i in range(n)])[:, :, None]
    X = np.repeat(tw, 2, axis=2)
    y = np.stack([serie[i + ws:i + ws + H] for i in range(n)])
    p = DataProcessor(window_size=ws)
    _, _, ctx = p.normalize_local(X, y, x_mode="zscore", y_mode="zscore_target",
                                  target_idx=0, target_windows=tw)
    s = scale_from_context("local", ctx, n, H)                     # escala (n,1) replicada em H
    assert s.shape == (n, H) and np.all(s > 0)
    assert np.allclose(s[:, 0], tw[:, :, 0].std(axis=1) + 1e-8)

    # Escala por passo (n,H), como no modo ancorado da branch do artigo 1:
    # s_h = |ref| sigma sqrt(h). Aqui o ctx e montado a mao para nao depender daquela branch.
    ref = np.abs(tw[:, -1, 0])[:, None]
    sc = ref * 0.01 * np.sqrt(np.arange(1, H + 1))[None, :]
    ctx_h = {"y": {"y_mode": "anchored", "y_center": ref, "y_scale": sc, "eps": 1e-8}}
    sh = scale_from_context("local", ctx_h, n, H)
    assert sh.shape == (n, H)
    assert np.allclose(sh[:, 2] / sh[:, 0], np.sqrt(3), rtol=1e-6)

    pg = DataProcessor(window_size=ws, scaler_type="minmax")
    pg.normalize_global(X, y)
    sg = scale_from_context("global", None, n, H, pg)
    assert sg.shape == (n, H)
    assert np.allclose(sg, y.max() - y.min())        # 1 unidade normalizada = amplitude do treino


def test_tercis_de_volatilidade_sao_causais_e_balanceados():
    p = 100 * np.exp(np.cumsum(RNG.normal(0, 0.01, 3000)))
    t = volatility_terciles(p, window=24)
    assert t.shape == p.shape
    counts = np.bincount(t, minlength=3)
    assert counts.min() > 0.3 * len(p) * 0.9
    p2 = p.copy()
    p2[-1] *= 10                                      # mudar a última barra
    assert np.array_equal(volatility_terciles(p2, window=24)[:-1], t[:-1])  # não altera o passado


def test_csv_exportado_carrega_a_escala(tmp_path):
    n, H = 20, 2
    ts = pd.date_range("2024-01-01", periods=n, freq="h").astype(str).to_numpy()
    y = RNG.normal(100, 1, (n, H))
    yh = y + 0.1
    s = np.abs(RNG.normal(2, 0.1, (n, H)))
    path = tmp_path / "r.csv"
    CSVExporter().save_predictions_to_csv(ts, y, yh, str(path), steps_ahead=H,
                                          extra_columns={"scale": s})
    d = read_predictions_csv(str(path), step=2)
    assert d["y"].shape == (n,)
    assert np.allclose(d["scale"], s[:, 1])
    with pytest.raises(ValueError):
        CSVExporter().save_predictions_to_csv(ts, y, yh, str(path), steps_ahead=H,
                                              extra_columns={"scale": s[:5]})


def test_interval_score_penaliza_falha_e_largura():
    y = np.array([0.0, 0.0, 0.0])
    assert interval_score(y, np.array([-1, -1, -1.0]), np.array([1, 1, 1.0]), 0.1) == 2.0
    pior = interval_score(y, np.array([1, -1, -1.0]), np.array([2, 1, 1.0]), 0.1)   # 1a obs fora
    assert pior > 2.0
    assert interval_score(y, -np.ones(3), np.ones(3), 0.1) < interval_score(y, -2 * np.ones(3), 2 * np.ones(3), 0.1)
