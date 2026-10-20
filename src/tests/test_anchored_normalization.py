"""
Testes da normalização ancorada (artigo 1) e do runner de origem rolante.

Rodar de dentro de `src/`:
    ./venv/bin/python -m pytest tests/test_anchored_normalization.py -q
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_processing import DataProcessor                      # noqa: E402
from services.model_service import ModelService                     # noqa: E402
from utils.baselines import constant_output_forecast, persistence_forecast  # noqa: E402
from optimization.rolling_origin import build_origin_configs        # noqa: E402

RNG = np.random.default_rng(11)
WS = 48


def _gbm(n, sigma, s0=100.0):
    """Passeio geométrico com volatilidade `sigma` por barra."""
    r = RNG.normal(0.0, sigma, n)
    return s0 * np.exp(np.cumsum(r))


def _windows(series, ws, H):
    n = len(series) - ws - H + 1
    tw = np.stack([series[i:i + ws] for i in range(n)])[:, :, None]      # (n, ws, 1)
    y = np.stack([series[i + ws:i + ws + H] for i in range(n)])          # (n, H)
    X = np.repeat(tw, 2, axis=2)                                          # 2 canais iguais
    return X, y, tw


def _norm(X, y, tw, H):
    p = DataProcessor(window_size=WS)
    return p, *p.normalize_local(X, y, x_mode="zscore", y_mode="anchored",
                                 target_idx=0, target_windows=tw)


def test_saida_zero_e_exatamente_persistencia():
    """Proposição 1: a previsão implícita do modo ancorado é a última observação."""
    s = _gbm(600, 0.01)
    X, y, tw = _windows(s, WS, 1)
    p, _, _, ctx = _norm(X, y, tw, 1)
    const = constant_output_forecast(lambda a: p.inverse_transform_local(a, ctx), len(y), 1, 0.0)
    assert np.allclose(const, persistence_forecast(tw, 1))


def test_inversa_recupera_o_alvo():
    for H in (1, 6, 24):
        s = _gbm(800, 0.02)
        X, y, tw = _windows(s, WS, H)
        p, _, yn, ctx = _norm(X, y, tw, H)
        assert yn.shape == y.shape
        assert np.allclose(p.inverse_transform_local(yn, ctx), y, rtol=1e-9, atol=1e-6)


def test_escala_cresce_com_raiz_do_horizonte():
    """Proposição 2 aplicada ao desenho: s_h = |a| σ sqrt(h)."""
    s = _gbm(800, 0.015)
    X, y, tw = _windows(s, WS, 24)
    _, _, _, ctx = _norm(X, y, tw, 24)
    scale = ctx["y"]["y_scale"]                                            # (n, 24)
    razao = scale / scale[:, [0]]
    assert np.allclose(razao, np.sqrt(np.arange(1, 25))[None, :])
    assert np.allclose(scale[:, 0], np.abs(ctx["y"]["y_center"][:, 0]) * ctx["y"]["y_sigma"][:, 0])


def test_residuo_normalizado_e_homocedastico_entre_regimes():
    """
    Em dois regimes de volatilidade (0,5% e 3% por barra), a dispersão do alvo
    ancorado deve ser parecida nos dois; em relative_last ela cresce ~6x.
    """
    calmo = _gbm(1500, 0.005)
    agitado = _gbm(1500, 0.03, s0=calmo[-1])
    out = {}
    for mode in ("anchored", "relative_last"):
        disp = []
        for serie in (calmo, agitado):
            X, y, tw = _windows(serie, WS, 1)
            p = DataProcessor(window_size=WS)
            _, yn, _ = p.normalize_local(X, y, x_mode="zscore", y_mode=mode,
                                         target_idx=0, target_windows=tw)
            disp.append(float(yn.std()))
        out[mode] = disp[1] / disp[0]
    assert out["relative_last"] > 4.0, out
    assert 0.6 < out["anchored"] < 1.7, out


def test_janela_sem_movimento_nao_explode():
    s = np.full(300, 50.0)
    X, y, tw = _windows(s, WS, 3)
    p, _, yn, ctx = _norm(X, y, tw, 3)
    assert np.all(np.isfinite(yn))
    assert np.allclose(p.inverse_transform_local(yn, ctx), y)


def test_modo_exige_canal_do_alvo():
    assert ModelService._y_mode_needs_target_in_X("anchored")
    assert ModelService._resolve_include_target_channel("legacy", norm_strategy="local", y_mode="anchored")


def test_modo_invalido_continua_falhando():
    s = _gbm(300, 0.01)
    X, y, tw = _windows(s, WS, 1)
    with pytest.raises(ValueError):
        DataProcessor(window_size=WS).normalize_local(X, y, x_mode="zscore", y_mode="ancorado",
                                                      target_idx=0, target_windows=tw)


# ------------------------------------------------------------ origem rolante
_CFG = {
    "start_date": "2017-08-18 00:00:00",
    "end_date": "2024-11-11 23:00:00",
    "interval": "1h",
    "train_size": 0.7,
    "validation_split": 0.15,
    "run_id": "abc",
    "normalization": {"strategy": "local", "x_mode": "zscore", "y_mode": "anchored"},
}


def test_origens_tem_teste_de_mesmo_tamanho_e_nao_se_sobrepoem():
    cfgs = build_origin_configs(_CFG, n_origins=5, test_hours=24 * 90)
    assert len(cfgs) == 5
    fins = [c["end_date"] for c in cfgs]
    assert fins == sorted(fins) and fins[-1] == _CFG["end_date"]
    for a, b in zip(cfgs, cfgs[1:]):
        assert b["rolling_origin"]["test_start_expected"] > a["end_date"]
    for c in cfgs:
        n = c["rolling_origin"]["n_bars_expected"]
        assert abs((1 - c["train_size"] - c["validation_split"]) * n - 24 * 90) < 1.0
        assert c["run_id"].startswith("abc_o")
    assert _CFG["train_size"] == 0.7, "config original nao pode ser alterado"


def test_origem_curta_demais_e_recusada():
    with pytest.raises(ValueError):
        build_origin_configs({**_CFG, "start_date": "2024-10-01 00:00:00"}, n_origins=3, test_hours=24 * 30)
