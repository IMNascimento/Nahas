"""
Testes da captura diaria de candles da Binance.

Nao acessam rede nem banco: o cliente HTTP e o repositorio sao falsos.

    cd src && ./venv/bin/python -m pytest tests/test_captura_diaria.py -q
"""
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.captura_diaria import (  # noqa: E402
    CapturaConfig,
    buscar_klines,
    capturar,
    captura_ativa,
    klines_para_linhas,
    proxima_execucao,
)

HORA_MS = 3_600_000


def _kline(abertura_ms, close="100.5"):
    """Formato da API /api/v3/klines: [abertura, o, h, l, c, v, fechamento, ...]."""
    return [abertura_ms, "100.0", "101.0", "99.0", close, "12.5", abertura_ms + HORA_MS - 1, "0", 0, "0", "0", "0"]


class HttpFalso:
    """Serve candles de 1h entre `inicio_ms` e `fim_ms`, respeitando `limit`."""

    def __init__(self, inicio_ms, fim_ms):
        self.inicio_ms, self.fim_ms, self.chamadas = inicio_ms, fim_ms, []

    def __call__(self, url, params):
        self.chamadas.append(dict(params))
        t = max(params["startTime"], self.inicio_ms)
        fim = min(params["endTime"], self.fim_ms)
        lote = []
        while t <= fim and len(lote) < params["limit"]:
            lote.append(_kline(t))
            t += HORA_MS
        return lote


class RepositorioFalso:
    def __init__(self, ultimo=None):
        self.ultimo, self.gravadas = ultimo, []

    def ultimo_timestamp(self, simbolo, intervalo, source, exchange):
        return self.ultimo

    def gravar(self, linhas):
        self.gravadas.extend(linhas)
        return len(linhas)


def test_buscar_klines_pagina_ate_o_fim():
    inicio = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    fim = inicio + 2500 * HORA_MS - 1
    http = HttpFalso(inicio, fim)

    klines = buscar_klines("BTCUSDT", "1h", inicio, fim, http_get=http, limite=1000)

    assert len(klines) == 2500
    assert len(http.chamadas) == 3
    assert [k[0] for k in klines] == sorted({k[0] for k in klines})


def test_klines_para_linhas_descarta_candle_ainda_aberto():
    agora = datetime(2024, 1, 1, 10, 30, tzinfo=timezone.utc)
    base = int(datetime(2024, 1, 1, 8, tzinfo=timezone.utc).timestamp() * 1000)
    klines = [_kline(base), _kline(base + HORA_MS), _kline(base + 2 * HORA_MS)]  # 08h, 09h, 10h (aberto)

    linhas = klines_para_linhas(klines, "ETHUSDT", "1h", "binance", "BINANCE", agora=agora)

    assert [l["timestamp"] for l in linhas] == [datetime(2024, 1, 1, 8), datetime(2024, 1, 1, 9)]
    assert linhas[0]["close"] == 100.5 and linhas[0]["currency"] == "USDT"
    assert linhas[0]["timestamp"].tzinfo is None  # UTC naive, como o restante do banco


def test_capturar_retoma_do_ultimo_candle_salvo():
    agora = datetime(2024, 1, 2, 0, 5, tzinfo=timezone.utc)
    repo = RepositorioFalso(ultimo=datetime(2024, 1, 1, 21))
    http = HttpFalso(0, int(agora.timestamp() * 1000))
    cfg = CapturaConfig(simbolos=("BTCUSDT",), intervalos=("1h",))

    total = capturar(cfg, repositorio=repo, http_get=http, agora=agora)

    assert http.chamadas[0]["startTime"] == int(datetime(2024, 1, 1, 22, tzinfo=timezone.utc).timestamp() * 1000)
    assert [l["timestamp"].hour for l in repo.gravadas] == [22, 23]  # 00h de 02/01 ainda esta aberto
    assert total == {("BTCUSDT", "1h"): 2}


def test_capturar_banco_vazio_comeca_na_listagem_da_binance():
    agora = datetime(2017, 8, 17, 6, 0, tzinfo=timezone.utc)
    http = HttpFalso(int(datetime(2017, 8, 17, 4, tzinfo=timezone.utc).timestamp() * 1000), int(agora.timestamp() * 1000))
    cfg = CapturaConfig(simbolos=("BTCUSDT",), intervalos=("1h",))

    capturar(cfg, repositorio=RepositorioFalso(), http_get=http, agora=agora)

    assert http.chamadas[0]["startTime"] == 0


def test_captura_ativa_le_o_env_a_cada_chamada(tmp_path):
    env = tmp_path / ".env"
    env.write_text("CAPTURA_DIARIA_ATIVA=false\n")
    assert captura_ativa(env) is False

    env.write_text("CAPTURA_DIARIA_ATIVA=true\n")
    assert captura_ativa(env) is True


def test_proxima_execucao_no_mesmo_dia_ou_no_seguinte():
    cfg = CapturaConfig(horario="23:55", fuso="America/Sao_Paulo")

    antes = datetime(2024, 3, 10, 12, 0, tzinfo=timezone.utc)   # 09:00 em Sao Paulo
    depois = datetime(2024, 3, 11, 3, 0, tzinfo=timezone.utc)   # 00:00 do dia 11 em Sao Paulo

    assert proxima_execucao(cfg, antes) == datetime(2024, 3, 11, 2, 55, tzinfo=timezone.utc)
    assert proxima_execucao(cfg, depois) == datetime(2024, 3, 12, 2, 55, tzinfo=timezone.utc)
