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


# ------------------------------------------------------------ simulacao hora a hora (Algoritmo 1)
from optimization.streaming import (   # noqa: E402
    STREAM_POLICIES, block_bounds, finetune_slices, first_alarm, simulate_hourly,
)


def test_blocos_e_fatias_do_finetuning():
    assert block_bounds(1500, 720) == [(0, 720), (720, 1440), (1440, 1500)]
    sl = finetune_slices(2000, days=90, val_days=7, embargo=96)
    assert sl["val"] == (2001 - 168, 2001)
    assert sl["train"] == (2001 - 2160, 2001 - 168 - 96)
    sl0 = finetune_slices(10, days=30, val_days=3, embargo=96)       # pode voltar antes do inicio da simulacao
    assert sl0["train"][0] < 0 and sl0["train"][1] < sl0["val"][0]
    with pytest.raises(ValueError):
        finetune_slices(10, days=1, val_days=0, embargo=96)
    assert set(STREAM_POLICIES) == {"A", "B", "C", "D", "E", "B_anch", "E_per"}


def test_first_alarm_para_no_primeiro_disparo():
    mon = CusumMonitor(k=0.5, h=3.0)
    z = np.array([0.0, 0.0, 2.0, 2.0, 2.0, 5.0, 5.0])
    i = first_alarm(mon, z)
    assert i == 4 and mon.s_plus == 0.0 and mon.alarms == [4]      # 1.5, 3.0 (nao excede), 4.5 > h


def _cenario_horario(n=3000, shift_at=1500):
    """Vies constante ate `shift_at`; depois triplica. Fine-tuning reduz o vies a 30%."""
    rng = np.random.default_rng(4)

    def predict_fn(state, a, b):
        idx = np.arange(a, b)
        y = 100 + rng.normal(0, 1, len(idx))
        bias = state["b"] * np.where(idx >= shift_at, 3.0, 1.0)
        return {"y": y, "y_hat": y - bias + rng.normal(0, 0.3, len(idx)),
                "y_last": y + rng.normal(0, 1.4, len(idx)), "scale": np.ones(len(idx))}

    chamadas = []

    def finetune_fn(state, t):
        chamadas.append(t)
        pior = len(chamadas) % 2 == 0                    # toda segunda tentativa e pior (rollback)
        return {"b": state["b"] * 0.3}, (0.9 if pior else 0.4), 0.6, 2.0

    return predict_fn, finetune_fn, chamadas, n


def test_simulacao_horaria_periodica_e_cusum():
    pf, ff, chamadas, n = _cenario_horario()
    r0 = simulate_hourly("none", n, pf, ff, {"b": 0.5}, monitor=CusumMonitor(0.5, 20.0))
    assert r0["n_retrain"] == 0 and r0["n_alarms"] > 0            # monitor passivo registra alarmes
    assert not np.isnan(r0["hours"]["y_hat"]).any() and r0["hours"]["block"][-1] == (n - 1) // 720

    pf, ff, chamadas, n = _cenario_horario()
    rp = simulate_hourly("periodic", n, pf, ff, {"b": 0.5}, period=720, block=720,
                         replay_fn=lambda s, a, b: pf(s, a, b)["y_hat"])
    assert chamadas == [719, 1439, 2159, 2879]                     # 720 h desde o ultimo update, nunca na ultima hora
    assert rp["n_retrain"] == 4 and rp["n_accepted"] == 2 and rp["n_rejected"] == 2
    rejeitados = [e for e in rp["events"] if e["type"] == "finetune" and not e["accepted"]]
    assert all("replay" in e and 0 < e["replay"]["to"] - e["replay"]["from"] <= 720 for e in rejeitados)
    assert rejeitados[0]["replay"]["to"] - rejeitados[0]["replay"]["from"] == 720 and rejeitados[-1]["replay"]["to"] == n
    assert all(np.isfinite(e["replay"]["rmse_rejected"]) for e in rejeitados)
    assert rp["hours"]["model_version"][-1] == 2

    pf, ff, chamadas, n = _cenario_horario()
    rc = simulate_hourly("cusum", n, pf, ff, {"b": 0.5}, monitor=CusumMonitor(0.5, 20.0), period=720)
    assert rc["n_retrain"] >= 1 and 1500 <= chamadas[0] < 1560     # detecta logo apos a mudanca
    assert rc["hours"]["alarm"][chamadas[0]]                        # a hora do alarme e a do fine-tuning
    assert rc["hours"]["model_version"][chamadas[0]] == 0 and rc["hours"]["model_version"][chamadas[0] + 1] == 1

    pf, ff, chamadas, n = _cenario_horario()
    rcp = simulate_hourly("cusum+periodic", n, pf, ff, {"b": 0.5}, monitor=CusumMonitor(0.5, 20.0), period=720)
    assert 719 in chamadas and rcp["n_retrain"] >= 4

    with pytest.raises(ValueError):
        simulate_hourly("cusum", 100, pf, ff, {"b": 0.5})


def test_rollback_sem_tolerancia_exige_melhora_estrita():
    pf, _, _, n = _cenario_horario(n=1500)
    r = simulate_hourly("periodic", n, pf, lambda s, t: (s, 0.6, 0.6, 1.0), {"b": 0.5}, period=720)
    assert r["n_retrain"] == 2 and r["n_accepted"] == 0
    r2 = simulate_hourly("periodic", n, pf, lambda s, t: (s, 0.6, 0.6, 1.0), {"b": 0.5}, period=720,
                         rollback_tolerance_pct=5.0)
    assert r2["n_accepted"] == 2
