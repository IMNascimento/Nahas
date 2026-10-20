"""
Testes do backbone linear e das flags de ablação (artigo 4).

Rodar de dentro de `src/`:
    ./venv/bin/python -m pytest tests/test_ablations_linear.py -q
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.trainer_factory import TrainerFactory                       # noqa: E402
from models.keras.transformer_trainer import sinusoidal_positional_encoding  # noqa: E402

RNG = np.random.default_rng(9)
L, C, H, N = 16, 3, 2, 64


def _dados():
    X = RNG.normal(0, 1, (N, L, C)).astype("float32")
    y = X[:, -1, :H] + 0.1 * RNG.normal(0, 1, (N, H)).astype("float32")
    return X, y


def _nomes(model):
    return [layer.__class__.__name__ for layer in model.layers]


def test_fabrica_resolve_o_backbone_linear():
    T = TrainerFactory.get_trainer("keras", "linear")
    assert T.__name__ == "KerasLinearTrainer"


def test_linear_treina_preve_e_tem_apenas_um_mapa_linear():
    X, y = _dados()
    T = TrainerFactory.get_trainer("keras", "linear")
    tr = T(input_shape=(L, C), epochs=2, patience=1, batch_size=16, output_units=H)
    model = tr.train(X, y, X, y)
    assert tr.predict(X).shape == (N, H)
    densas = [ly for ly in model.layers if ly.__class__.__name__ == "Dense"]
    assert len(densas) == 1 and densas[0].activation.__name__ == "linear"
    assert "LayerNormalization" not in _nomes(model)
    assert tr.feature_importance().shape == (L, C, H)
    assert tr.validate_model()


def test_transformer_sem_layernorm_remove_a_normalizacao_interna():
    T = TrainerFactory.get_trainer("keras", "transformer")
    com = T(input_shape=(L, C), num_layers=1, embed_dim=8, num_heads=2, ff_dim=8,
            epochs=1, patience=1, output_units=H, use_layernorm=True).build_model()
    sem = T(input_shape=(L, C), num_layers=1, embed_dim=8, num_heads=2, ff_dim=8,
            epochs=1, patience=1, output_units=H, use_layernorm=False).build_model()
    assert _nomes(com).count("LayerNormalization") == 2
    assert _nomes(sem).count("LayerNormalization") == 0
    assert _nomes(sem).count("Add") == 2                      # residuais preservadas


def test_transformer_com_codificacao_posicional_senoidal():
    T = TrainerFactory.get_trainer("keras", "transformer")
    tr = T(input_shape=(L, C), num_layers=1, embed_dim=8, num_heads=2, ff_dim=8,
           epochs=1, patience=1, output_units=H, positional_encoding="sinusoidal")
    model = tr.build_model()
    assert any(layer.name == "positional_encoding" for layer in model.layers)
    X, _ = _dados()
    assert model.predict(X, verbose=0).shape == (N, H)
    pe = sinusoidal_positional_encoding(L, 8)
    assert pe.shape == (L, 8) and np.allclose(pe[0, 0::2], 0.0) and np.allclose(pe[0, 1::2], 1.0)
    with pytest.raises(ValueError):
        T(input_shape=(L, C), output_units=H, positional_encoding="learned")


def test_transformer_padrao_reproduz_a_variante_anterior():
    """Sem flags: LayerNorm ligado e sem codificação posicional, como no estudo anterior."""
    T = TrainerFactory.get_trainer("keras", "transformer")
    tr = T(input_shape=(L, C), num_layers=2, embed_dim=8, num_heads=2, ff_dim=8, output_units=H)
    hp = tr.get_hyperparameters()
    assert hp["use_layernorm"] is True and hp["positional_encoding"] == "none"
    model = tr.build_model()
    assert _nomes(model).count("LayerNormalization") == 4
    assert not any(layer.name == "positional_encoding" for layer in model.layers)


def test_lstm_com_layer_norm():
    T = TrainerFactory.get_trainer("keras", "lstm")
    base = T(input_shape=(L, C), layers_config=[8, 4], output_units=H).build_model()
    ln = T(input_shape=(L, C), layers_config=[8, 4], output_units=H, layer_norm=True).build_model()
    assert _nomes(base).count("LayerNormalization") == 0
    assert _nomes(ln).count("LayerNormalization") == 2
    X, _ = _dados()
    assert ln.predict(X, verbose=0).shape == (N, H)


def test_config_de_treino_aceita_as_flags_como_kwargs():
    """O ModelService instancia o trainer com **config; as flags precisam passar sem erro."""
    cfg = dict(input_shape=(L, C), epochs=1, patience=1, batch_size=8, output_units=H,
               window_size=L, seed=1, normalization={"strategy": "global"},
               use_layernorm=False, positional_encoding="sinusoidal", layer_norm=True,
               layers_config=[4], num_layers=1, embed_dim=8, num_heads=2, ff_dim=8)
    for mt in ("lstm", "transformer", "linear"):
        TrainerFactory.get_trainer("keras", mt)(**cfg).build_model()
