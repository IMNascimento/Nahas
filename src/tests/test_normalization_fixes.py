"""
Testes das correções de confundimento entre estratégias de normalização.

Rodar de dentro de `src/`:
    ./venv/bin/python -m pytest tests/test_normalization_fixes.py -q
    ./venv/bin/python tests/test_normalization_fixes.py      # sem pytest
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_processing import DataProcessor            # noqa: E402
from services.model_service import ModelService           # noqa: E402
from utils.baselines import (                             # noqa: E402
    persistence_forecast,
    constant_output_forecast,
    diebold_mariano,
    mimicry_diagnostics,
)

WS, STEPS, N = 12, 1, 400
RNG = np.random.default_rng(7)


def _fake_ohlcv(n=N):
    close = 100 + np.cumsum(RNG.normal(0, 1, n))
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
        "close": close,
        "open": close + RNG.normal(0, .1, n),
        "high": close + np.abs(RNG.normal(0, .3, n)),
        "low": close - np.abs(RNG.normal(0, .3, n)),
        "volume": RNG.uniform(1, 10, n),
    })


def test_windowing_sem_vazamento():
    """X cobre [j, j+ws-1]; y é target[j+ws]. tw termina na barra anterior ao alvo."""
    df = _fake_ohlcv()
    p = DataProcessor(window_size=WS)
    data = df.drop(columns=["timestamp"])
    X, y = p.create_windows(data, coluna_alvo="close", steps_ahead=STEPS)
    tw = ModelService._build_target_windows(data["close"].values, WS, STEPS)

    assert len(X) == len(y) == len(tw)
    close = data["close"].values
    for j in (0, 5, len(X) - 1):
        assert y[j, 0] == close[j + WS], "alvo desalinhado"
        assert tw[j, -1, 0] == close[j + WS - 1], "tw invade o alvo"
        assert tw[j, -1, 0] != y[j, 0]
    print("OK  janelamento sem vazamento")


def test_canal_do_alvo_equaliza_estrategias():
    """O confundimento: antes, só o braço 'local' recebia o canal do alvo."""
    r = ModelService._resolve_include_target_channel
    # legacy: assimétrico (o bug)
    assert r("legacy", norm_strategy="local", y_mode="relative_last") is True
    assert r("legacy", norm_strategy="global", y_mode="relative_last") is False
    assert r("legacy", norm_strategy="evomsn", y_mode="relative_last") is False
    # equalized: mesma entrada para todos
    for s in ("global", "local", "evomsn"):
        assert r("equalized", norm_strategy=s, y_mode="relative_last") is True
        assert r("never", norm_strategy=s, y_mode="relative_last") is False
    print("OK  canal do alvo equalizado entre estrategias")


def test_target_windows_reproduz_caminho_antigo():
    """Com tw == canal de X, o caminho novo é numericamente idêntico ao antigo."""
    df = _fake_ohlcv()
    p = DataProcessor(window_size=WS)
    data = df.drop(columns=["timestamp"])
    X, y = p.create_windows(data, coluna_alvo="close", steps_ahead=STEPS)
    tw = ModelService._build_target_windows(data["close"].values, WS, STEPS)
    Xc = np.concatenate([X, tw], axis=2)

    for y_mode in ("relative_last", "zscore_target", "minmax_target", "robust_target"):
        _, y_old, _ = p.normalize_local(Xc, y, y_mode=y_mode, target_idx=Xc.shape[2] - 1)
        _, y_new, _ = p.normalize_local(X, y, y_mode=y_mode, target_windows=tw)
        assert np.allclose(y_old, y_new, atol=1e-12), f"divergiu em {y_mode}"
    print("OK  target_windows reproduz o caminho antigo (retrocompatível)")


def test_zero_normalizado_e_exatamente_persistencia():
    """Sob relative_last, prever 0 no espaço normalizado É a persistência."""
    df = _fake_ohlcv()
    p = DataProcessor(window_size=WS)
    data = df.drop(columns=["timestamp"])
    X, y = p.create_windows(data, coluna_alvo="close", steps_ahead=STEPS)
    tw = ModelService._build_target_windows(data["close"].values, WS, STEPS)

    _, _, ctx = p.normalize_local(X, y, y_mode="relative_last", target_windows=tw)
    const = constant_output_forecast(
        lambda a: p.inverse_transform_local(a, ctx), len(X), STEPS, value=0.0
    )
    persist = persistence_forecast(tw, STEPS)
    assert np.allclose(const, persist, atol=1e-9), "zero normalizado != persistencia"
    print(f"OK  saida constante zero == persistencia (max diff "
          f"{np.max(np.abs(const - persist)):.2e})")


def test_embargo_cria_lacuna():
    df = _fake_ohlcv()
    p = DataProcessor(window_size=WS)
    n = len(df) - WS - STEPS + 1
    emb = WS + STEPS - 1

    s0 = p.split_indices(n, 0.7, 0.15, embargo=0)
    assert s0["train"].stop == s0["val"].start, "sem embargo deveria ser contiguo"

    s1 = p.split_indices(n, 0.7, 0.15, embargo=emb)
    assert s1["val"].start - s1["train"].stop == emb
    assert s1["test"].start - s1["val"].stop == emb
    assert s1["test"] == s0["test"], "o teste nao deve encolher"
    print(f"OK  embargo={emb} cria lacuna entre segmentos, teste intacto")


def test_diagnosticos_separam_mimetismo_de_skill():
    df = _fake_ohlcv()
    p = DataProcessor(window_size=WS)
    data = df.drop(columns=["timestamp"])
    X, y = p.create_windows(data, coluna_alvo="close", steps_ahead=STEPS)
    tw = ModelService._build_target_windows(data["close"].values, WS, STEPS)
    last = tw[:, -1, 0].reshape(-1, 1)

    # emulador de persistência puro
    mim = mimicry_diagnostics(y, last, tw)
    assert abs(mim["corr_pred_vs_last_obs"] - 1.0) < 1e-9
    assert np.isnan(mim["corr_returns"]), "retorno previsto constante -> corr indefinida"

    # modelo com sinal genuíno (50% do retorno real)
    skilled = last + 0.5 * (y - last)
    mim2 = mimicry_diagnostics(y, skilled, tw)
    assert mim2["corr_returns"] > 0.99
    assert mim2["directional_accuracy"] > 0.99

    dm = diebold_mariano(y[:, 0], last[:, 0], last[:, 0])
    assert abs(dm["mean_loss_diff"]) < 1e-12, "DM contra si mesmo deve dar diferenca nula"
    print("OK  diagnosticos separam mimetismo de habilidade")




# ---------------------------------------------------------------- EvoMSN
from data.evomsn_normalizer import EvoMSNNormalizer, EvoMSNLikeNormalizer  # noqa: E402


def _evomsn_fixture(H=1, L=48, C=4, n=180):
    close = 100 + np.cumsum(RNG.normal(0, 1, n + L + H))
    X = np.stack([
        np.stack([close[i:i + L] + RNG.normal(0, .2, L) for _ in range(C)], axis=1)
        for i in range(n)
    ])
    y = np.stack([close[i + L:i + L + H] for i in range(n)])
    return X, y, X[:, :, [0]]


def _sensibilidade(msn, X, tw, H):
    """Quanto a saída da rede realmente move a previsão final."""
    _, _, ctx = msn.transform(X, None, target_windows=tw)
    a = RNG.normal(0, 1, (ctx["xi_hat_stack"].shape[0], H))
    _, ea = msn.denorm_and_ensemble(a, ctx)
    _, eb = msn.denorm_and_ensemble(a + 1.0, ctx)
    return float(np.mean(np.abs(eb - ea)))


def test_evomsn_degenera_sob_horizonte_unitario():
    """Com H=1 a fatia do alvo é constante: std=0, e a rede é anulada."""
    X, y, tw = _evomsn_fixture(H=1)
    msn = EvoMSNNormalizer(window_size=X.shape[1], horizon=1, n_features=X.shape[2],
                           k_scales=3, short_horizon_policy="legacy")
    _, yn = msn.fit(X, y, target_windows=tw)

    assert all(msn.degeneracy_report["degenerate_scales"]), "deveria detectar degenerescencia"
    assert yn.std() < 1e-4, f"alvo do backbone deveria ser ~constante, std={yn.std():.2e}"
    sens = _sensibilidade(msn, X, tw, 1)
    assert sens < 1e-6, f"a saida da rede nao deveria importar em legacy, sens={sens:.2e}"
    print(f"OK  legacy degenera: std_alvo={yn.std():.2e}, sensibilidade={sens:.2e}")


def test_evomsn_window_stats_devolve_a_rede():
    """Com estatísticas da janela, o alvo volta a ter escala e a rede volta a importar."""
    X, y, tw = _evomsn_fixture(H=1)
    msn = EvoMSNNormalizer(window_size=X.shape[1], horizon=1, n_features=X.shape[2],
                           k_scales=3, short_horizon_policy="window_stats")
    _, yn = msn.fit(X, y, target_windows=tw)

    assert all(msn.degeneracy_report["using_window_stats"])
    assert yn.std() > 0.1, f"alvo do backbone deveria ter escala util, std={yn.std():.2e}"
    sens = _sensibilidade(msn, X, tw, 1)
    assert sens > 0.1 * float(y.std()), f"a saida da rede deveria mover a previsao, sens={sens:.2e}"
    print(f"OK  window_stats: std_alvo={yn.std():.3f}, sensibilidade={sens:.3f} "
          f"(std do alvo real={float(y.std()):.3f})")


def test_evomsn_policy_error_falha_alto():
    X, y, tw = _evomsn_fixture(H=1)
    msn = EvoMSNNormalizer(window_size=X.shape[1], horizon=1, n_features=X.shape[2],
                           k_scales=3, short_horizon_policy="error")
    try:
        msn.fit(X, y, target_windows=tw)
    except ValueError as e:
        assert "menor que os per" in str(e)
        print("OK  policy='error' recusa horizonte invalido")
        return
    raise AssertionError("deveria ter levantado ValueError")


def test_evomsn_horizonte_longo_inalterado():
    """H >= p: nenhuma escala degenera, a política não muda nada."""
    L, H = 8, 8
    X, y, tw = _evomsn_fixture(H=H, L=L, C=3, n=120)
    out = {}
    for pol in ("legacy", "window_stats"):
        msn = EvoMSNNormalizer(window_size=L, horizon=H, n_features=X.shape[2],
                               k_scales=2, short_horizon_policy=pol)
        _, yn = msn.fit(X, y, target_windows=tw)
        assert not any(msn.degeneracy_report["degenerate_scales"]), \
            f"p={msn.meta.periods} nao deveria degenerar com H={H}"
        out[pol] = yn
    assert np.allclose(out["legacy"], out["window_stats"]), \
        "sem degenerescencia as politicas devem coincidir"
    print("OK  horizonte longo: politicas coincidem, nada muda")


def test_evomsn_like_nao_mistura_canais():
    """A variante 'like' usava media sobre todos os canais (volume com preco)."""
    L, H, n = 24, 1, 120
    close = 1000 + np.cumsum(RNG.normal(0, 5, n + L + H))
    volume = RNG.uniform(1, 10, n + L + H)          # escala 1e0 vs preco 1e3
    X = np.stack([np.stack([close[i:i + L], volume[i:i + L]], axis=1) for i in range(n)])
    y = close[L:L + n].reshape(-1, 1)
    tw = X[:, :, [0]]

    msn = EvoMSNLikeNormalizer(window_size=L, horizon=H, n_features=2,
                               target_idx=0, k_scales=2, short_horizon_policy="legacy")
    msn.fit(X, y, target_windows=tw)
    _, _, ctx = msn.transform(X, None, target_windows=tw)
    phi = ctx["phi_hat_stack"]
    # phi deve estar na escala do preco, nao puxado para baixo pelo volume
    assert phi.mean() > 0.5 * close.mean(), \
        f"phi={phi.mean():.1f} fora da escala do preco ({close.mean():.1f}) -> canais misturados"
    print(f"OK  'like' usa o canal do alvo: phi={phi.mean():.1f} vs preco={close.mean():.1f}")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print(f"\n{len(fns)} testes passaram.")
