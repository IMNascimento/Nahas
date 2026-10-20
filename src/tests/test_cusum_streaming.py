"""
Testes do CUSUM sobre resíduos padronizados e da simulação de fluxo (artigo 5).

Rodar de dentro de `src/`:
    ./venv/bin/python -m pytest tests/test_cusum_streaming.py -q
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_processing import DataProcessor                      # noqa: E402
from optimization.streaming import POLICIES, predict_block, simulate_all, simulate_policy  # noqa: E402
from utils.cusum import (                                           # noqa: E402
    CusumMonitor,
    average_run_length,
    calibrate_threshold,
    cusum_two_sided,
    standardized_residuals,
)

RNG = np.random.default_rng(21)


def test_residuo_padronizado_e_adimensional():
    y = np.array([100.0, 200.0])
    yh = np.array([99.0, 198.0])
    s = np.array([1.0, 2.0])
    assert np.allclose(standardized_residuals(y, yh, s), [1.0, 1.0])
    with pytest.raises(ValueError):
        standardized_residuals(y, yh, np.array([1.0, 0.0]))


def test_cusum_detecta_mudanca_de_media_e_nao_dispara_sem_ela():
    z0 = RNG.normal(0, 1, 3000)
    out0 = cusum_two_sided(z0, k=0.5, h=8.0)
    assert len(out0["alarms"]) <= 2
    z1 = np.concatenate([RNG.normal(0, 1, 1000), RNG.normal(1.5, 1, 500)])
    out1 = cusum_two_sided(z1, k=0.5, h=8.0)
    assert out1["alarms"] and 1000 <= out1["alarms"][0] < 1060     # detecta logo após a mudança
    z2 = np.concatenate([RNG.normal(0, 1, 1000), RNG.normal(-1.5, 1, 500)])
    assert cusum_two_sided(z2, k=0.5, h=8.0)["alarms"][0] >= 1000   # bilateral


def test_calibracao_atinge_o_arl0_alvo():
    z_ref = RNG.standard_t(5, 4000)
    cal = calibrate_threshold(z_ref, k=0.5, target_arl0=720, n_sim=60, tol=0.15,
                              rng=np.random.default_rng(1))
    assert cal["converged"]
    arl = average_run_length((z_ref - cal["mu"]) / cal["sd"], 0.5, cal["h"], n_sim=60,
                             horizon=4000, rng=np.random.default_rng(2))
    assert 0.6 * 720 < arl < 1.6 * 720, (cal, arl)


def test_monitor_incremental_equivale_ao_lote():
    z = np.concatenate([RNG.normal(0, 1, 800), RNG.normal(2, 1, 200)])
    lote = cusum_two_sided(z, k=0.5, h=6.0)["alarms"]
    mon = CusumMonitor(k=0.5, h=6.0)
    inc = mon.update_many(z)
    assert inc == lote


# ------------------------------------------------------------ simulação de fluxo
class _Bloco:
    def __init__(self, i, nivel, ruido):
        self.label = f"m{i}"
        self.nivel = nivel
        self.ruido = ruido


def _cenario():
    """Modelo com viés fixo; a partir do bloco 6 o regime muda e o viés dobra."""
    rng = np.random.default_rng(4)
    blocos = [_Bloco(i, 100.0, 1.0) for i in range(12)]

    def predict_fn(state, b):
        n = 300
        y = b.nivel + rng.normal(0, b.ruido, n)
        vies = state["vies"] * (2.5 if int(b.label[1:]) >= 6 else 1.0)
        return {"y": y, "y_hat": y - vies + rng.normal(0, 0.3, n), "y_last": y + rng.normal(0, 1.4, n),
                "scale": np.full(n, 1.0)}

    def finetune_fn(state, b):
        return {"vies": state["vies"] * 0.3}, 0.5          # retreino reduz o viés

    return blocos, predict_fn, finetune_fn, {"vies": 0.5}


def test_politicas_tem_os_efeitos_esperados():
    blocos, pf, ff, s0 = _cenario()
    res = simulate_all(blocos, pf, ff, s0, period=3,
                       monitor_factory=lambda: CusumMonitor(k=0.5, h=20.0))
    assert set(res) == set(POLICIES)
    assert res["A"]["n_retrain"] == 0
    assert res["B"]["n_retrain"] == 3                      # blocos 3, 6, 9 (nunca no último)
    assert res["C"]["n_retrain"] >= 1
    assert res["C"]["rmse_mean"] < res["A"]["rmse_mean"]
    assert res["B"]["rmse_mean"] < res["A"]["rmse_mean"]
    assert all(r["theil_u"] > 0 for r in res["A"]["blocks"])


def test_rollback_quando_o_ajuste_piora():
    blocos, pf, _, s0 = _cenario()
    chamadas = {"n": 0}

    def ff_ruim(state, b):
        chamadas["n"] += 1
        return {"vies": 5.0}, 0.5 if chamadas["n"] == 1 else 10.0   # segunda tentativa piora muito
    res = simulate_policy("B", blocos, pf, ff_ruim, s0, period=3)
    assert res["n_retrain"] == 3 and res["n_rollback"] == 2
    assert [r["rolled_back"] for r in res["blocks"] if r["retrained"]] == [False, True, True]


def test_politica_c_exige_monitor():
    blocos, pf, ff, s0 = _cenario()
    with pytest.raises(ValueError):
        simulate_policy("C", blocos, pf, ff, s0)


# ------------------------------------------------------------ adaptador Nahas
class _ModeloZero:
    """Rede que emite zero: sob relative_last a previsão é a persistência."""
    def predict(self, X, verbose=0):
        return np.zeros((len(X), 1))


def test_predict_block_local_zero_e_persistencia():
    ws, n = 24, 120
    serie = 100 + np.cumsum(RNG.normal(0, 1, n + ws))
    df = pd.DataFrame({"close": serie, "open": serie + 0.1, "volume": RNG.uniform(1, 2, n + ws)})
    cfg = {"window_size": ws, "steps_ahead": 1, "target_column": "close",
           "relevant_columns": ["close", "open", "volume"],
           "normalization": {"strategy": "local", "x_mode": "zscore", "y_mode": "relative_last"},
           "include_target_channel": "equalized"}
    out = predict_block(_ModeloZero(), cfg, df, DataProcessor(window_size=ws), inverse_kind="local")
    assert out["y"].shape == (n,) and out["y_hat"].shape == (n,) and out["scale"].shape == (n,)
    assert np.allclose(out["y_hat"], out["y_last"])
    assert np.allclose(out["y"], serie[ws:])


def test_predict_block_global_usa_o_scaler_salvo(tmp_path):
    ws, n = 24, 120
    serie = 100 + np.cumsum(RNG.normal(0, 1, n + ws))
    df = pd.DataFrame({"close": serie, "open": serie + 0.1, "volume": RNG.uniform(1, 2, n + ws)})
    cfg = {"window_size": ws, "steps_ahead": 1, "target_column": "close",
           "relevant_columns": ["close", "open", "volume"],
           "normalization": {"strategy": "global", "scaler_type": "standard"},
           "include_target_channel": "equalized"}
    p = DataProcessor(window_size=ws, scaler_type="standard")
    X, y = p.create_windows(df[["open", "volume", "close"]], coluna_alvo="close", steps_ahead=1)
    tw = np.stack([serie[i:i + ws] for i in range(n)])[:, :, None]
    p.normalize_global(np.concatenate([X, tw], axis=2), y)
    p.save_scaler(str(tmp_path))
    out = predict_block(_ModeloZero(), cfg, df, DataProcessor(window_size=ws, scaler_type="standard"),
                        inverse_kind="global", scaler_dir=str(tmp_path))
    assert np.allclose(out["y_hat"], y.mean())            # zero normalizado = média do treino
    assert np.allclose(out["scale"], y.std())
