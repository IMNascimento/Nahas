"""
Testes do artigo 6 (habilidade vs mimetismo): backbone CNN-LSTM, entrada
univariada e teste de saida constante no caminho EvoMSN.

Rodar de dentro de `src/`:
    ./venv/bin/python -m pytest tests/test_artigo6_mimetismo.py -q
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.trainer_factory import TrainerFactory          # noqa: E402
from services.model_service import ModelService              # noqa: E402
from data.data_processing import DataProcessor               # noqa: E402
from utils.baselines import evaluate_against_baselines       # noqa: E402

RNG = np.random.default_rng(6)
L, C, H, N = 16, 3, 2, 64


def _dados():
    X = RNG.normal(0, 1, (N, L, C)).astype("float32")
    y = X[:, -1, :H] + 0.1 * RNG.normal(0, 1, (N, H)).astype("float32")
    return X, y


def _nomes(model):
    return [layer.__class__.__name__ for layer in model.layers]


def test_fabrica_resolve_o_cnn_lstm():
    assert TrainerFactory.get_trainer("keras", "cnn_lstm").__name__ == "KerasCNNLSTMTrainer"


def test_cnn_lstm_treina_preve_salva_e_carrega(tmp_path):
    X, y = _dados()
    T = TrainerFactory.get_trainer("keras", "cnn_lstm")
    # dropout=0: no Keras 3.7 a LSTM aplica a mascara de dropout tambem na inferencia
    # (caminho cuDNN/backend.lstm ignora `training`), o que tornaria predict() aleatorio.
    tr = T(input_shape=(L, C), layers_config=[8, 4], conv_filters=4, conv_kernel=3, pool_size=2,
           dropout=0.0, epochs=1, patience=1, batch_size=16, output_units=H)
    model = tr.train(X, y, X, y)
    nomes = _nomes(model)
    assert nomes[:2] == ["Conv1D", "MaxPooling1D"] and nomes.count("LSTM") == 2 and nomes[-1] == "Dense"
    assert tr.predict(X).shape == (N, H)
    hp = tr.get_hyperparameters()
    assert hp["conv_filters"] == 4 and hp["conv_kernel"] == 3 and hp["pool_size"] == 2 and hp["conv_layers"] == 1
    p = str(tmp_path / "m.keras")
    tr.save_model(p)
    m2 = tr.load_model(p)
    assert np.allclose(m2.predict(X, verbose=0), model.predict(X, verbose=0))


def test_cnn_lstm_aceita_os_kwargs_da_grade_e_as_chaves_sao_mapeadas():
    mapa = ModelService._grid_key_mapping(None)   # nao usa self
    for k, v in {"CONV_FILTERS": "conv_filters", "CONV_KERNEL": "conv_kernel",
                 "CONV_LAYERS": "conv_layers", "POOL_SIZE": "pool_size"}.items():
        assert mapa[k] == v
    # kwargs de outros backbones (LSTM/Transformer) nao derrubam o construtor
    T = TrainerFactory.get_trainer("keras", "cnn_lstm")
    tr = T(input_shape=(L, C), layers_config=[4], epochs=1, patience=1, output_units=H,
           num_layers=2, embed_dim=32, use_layernorm=True, layer_norm=False, bidirectional=False)
    assert tr.build_model() is not None


def test_cnn_lstm_rejeita_pool_maior_que_a_janela():
    import pytest
    T = TrainerFactory.get_trainer("keras", "cnn_lstm")
    with pytest.raises(ValueError):
        T(input_shape=(L, C), layers_config=[4], pool_size=L + 1, epochs=1, patience=1)


def test_entrada_univariada_gera_um_canal_com_o_alvo():
    """relevant_columns = [close] + canal do alvo 'equalized' => X com exatamente 1 canal."""
    n = 200
    df = pd.DataFrame({"close": 100 + np.cumsum(RNG.normal(0, 1, n)),
                       "timestamp": pd.date_range("2024-01-01", periods=n, freq="h")})
    proc = DataProcessor(window_size=8, scaler_type="minmax")
    X, y, ts = proc.create_windows_and_timestamps(df, coluna_alvo="close", steps_ahead=1)
    assert X.shape[2] == 0                                   # X nao inclui o alvo
    tw = ModelService._build_target_windows(df["close"].values, 8, 1)
    assert ModelService._resolve_include_target_channel("equalized", norm_strategy="global", y_mode="none")
    X = np.concatenate([X, tw], axis=2)
    assert X.shape == (len(y), 8, 1) and np.allclose(X[:, -1, 0], df["close"].values[7:-1])
    # scaler global MinMax ajusta e inverte no canal unico


def test_saida_constante_registrada_para_global_e_local():
    tw = np.abs(RNG.normal(100, 5, (N, L, 1)))
    y = tw[:, -1, 0:1] + RNG.normal(0, 1, (N, 1))
    yhat = y + RNG.normal(0, 1, (N, 1))
    # ancorada: zero normalizado = ultimo valor => igual a persistencia
    rep = evaluate_against_baselines(y, yhat, tw, 1, inverse_fn=lambda a: a + tw[:, -1, :])
    assert rep["constant_zero_normalized"]["equals_persistence"] is True
    # global minmax: zero normalizado = minimo do treino => diferente da persistencia
    rep = evaluate_against_baselines(y, yhat, tw, 1, inverse_fn=lambda a: a * 50.0 + 80.0)
    assert rep["constant_zero_normalized"]["equals_persistence"] is False
    assert rep["constant_zero_normalized"]["rmse"] > rep["persistence"]["rmse"]
    for k in ("mimicry", "dm_model_vs_persistence", "constant_zero_vs_persistence"):
        assert k in rep


def test_saida_constante_no_evomsn_e_o_ensemble_dos_niveis_previstos():
    """Com saida zero, a inversao do EvoMSN devolve o ensemble ponderado de phi_hat."""
    class _Msn:
        eps = 1e-8
        def denorm_and_ensemble(self, stack, ctx):
            k = ctx["k"]; phi = ctx["phi_hat_stack"]; xi = ctx["xi_hat_stack"]
            y_den = stack * (xi + self.eps) + phi
            Nn = y_den.shape[0] // k
            per = y_den.reshape(k, Nn, -1).transpose(1, 0, 2)
            w = ctx["weights_stack"].reshape(Nn, k); w = w / w.sum(axis=1, keepdims=True)
            return per, np.sum(w[:, :, None] * per, axis=1)
    k, Nn, Hh = 3, 10, 2
    phi = RNG.normal(100, 1, (k * Nn, Hh)); xi = np.abs(RNG.normal(1, 0.1, (k * Nn, Hh)))
    w = np.abs(RNG.normal(1, 0.1, (k * Nn,)))
    ctx = {"k": k, "phi_hat_stack": phi, "xi_hat_stack": xi, "weights_stack": w}
    out = ModelService._evomsn_inverse_constant(np.zeros((4, Hh)), (_Msn(), ctx))
    esperado = _Msn().denorm_and_ensemble(np.zeros((k * Nn, Hh)), ctx)[1][-4:]
    assert out.shape == (4, Hh) and np.allclose(out, esperado)
