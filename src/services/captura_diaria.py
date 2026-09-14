"""
Captura diaria de candles da Binance para a tabela price_history.

Busca, para cada simbolo e intervalo, os candles fechados desde o ultimo
registro salvo (ou desde a listagem na Binance, se a tabela estiver vazia) e
grava com upsert. Usa o endpoint publico /api/v3/klines, sem chave de API.

Ativacao
--------
A captura e controlada pelo .env, relido a cada ciclo (nao precisa reiniciar):

    CAPTURA_DIARIA_ATIVA=true          # false desliga
    CAPTURA_SIMBOLOS=BTCUSDT,ETHUSDT
    CAPTURA_INTERVALOS=1h
    CAPTURA_HORARIO=23:55              # hora local de execucao
    CAPTURA_FUSO=America/Sao_Paulo

Uso
---
    cd src
    ./venv/bin/python -m services.captura_diaria --agora     # uma vez, se ativa
    ./venv/bin/python -m services.captura_diaria --forcar    # uma vez, ignora a chave
    ./venv/bin/python -m services.captura_diaria --daemon    # todo dia no horario
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable
from zoneinfo import ZoneInfo

URL_KLINES = "https://api.binance.com/api/v3/klines"
SOURCE = "binance"      # mesmo rotulo usado nos configs de treino
EXCHANGE = "BINANCE"
ENV_PADRAO = Path(__file__).resolve().parent.parent / ".env"

DURACAO_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
MOEDAS_COTACAO = ("USDT", "FDUSD", "BUSD", "USD", "BRL")


@dataclass(frozen=True)
class CapturaConfig:
    """Parametros da captura. `from_env` le os valores do .env."""

    simbolos: tuple = ("BTCUSDT", "ETHUSDT")
    intervalos: tuple = ("1h",)
    horario: str = "23:55"
    fuso: str = "America/Sao_Paulo"
    source: str = SOURCE
    exchange: str = EXCHANGE

    @classmethod
    def from_env(cls, env_path: Path = ENV_PADRAO) -> "CapturaConfig":
        valores = _ler_env(env_path)
        lista = lambda chave, padrao: tuple(s.strip() for s in valores.get(chave, padrao).split(",") if s.strip())
        return cls(
            simbolos=lista("CAPTURA_SIMBOLOS", "BTCUSDT,ETHUSDT"),
            intervalos=lista("CAPTURA_INTERVALOS", "1h"),
            horario=valores.get("CAPTURA_HORARIO", cls.horario),
            fuso=valores.get("CAPTURA_FUSO", cls.fuso),
        )


def _ler_env(env_path: Path) -> dict:
    """Le o .env direto do arquivo, para que mudancas valham sem reiniciar o processo."""
    from dotenv import dotenv_values

    valores = dict(dotenv_values(env_path)) if Path(env_path).exists() else {}
    return {k: (v or "").strip() for k, v in valores.items()}


def captura_ativa(env_path: Path = ENV_PADRAO) -> bool:
    """Retorna True se CAPTURA_DIARIA_ATIVA estiver ligada no .env."""
    return _ler_env(env_path).get("CAPTURA_DIARIA_ATIVA", "false").lower() in ("1", "true", "sim", "yes")


def _http_get_json(url: str, params: dict) -> list:
    requisicao = f"{url}?{urllib.parse.urlencode(params)}"
    for tentativa in range(5):
        try:
            with urllib.request.urlopen(requisicao, timeout=30) as resposta:
                return json.load(resposta)
        except Exception:
            if tentativa == 4:
                raise
            time.sleep(2 ** tentativa)
    return []


def buscar_klines(simbolo: str, intervalo: str, inicio_ms: int, fim_ms: int,
                  http_get: Callable[[str, dict], list] = _http_get_json, limite: int = 1000) -> list:
    """Busca todos os candles entre `inicio_ms` e `fim_ms`, paginando de `limite` em `limite`."""
    klines, cursor = [], inicio_ms
    while cursor <= fim_ms:
        lote = http_get(URL_KLINES, {"symbol": simbolo, "interval": intervalo,
                                     "startTime": cursor, "endTime": fim_ms, "limit": limite})
        if not lote:
            break
        klines.extend(lote)
        cursor = lote[-1][0] + DURACAO_MS[intervalo]
        if len(lote) < limite:
            break
    return klines


def _moeda(simbolo: str) -> str:
    return next((m for m in MOEDAS_COTACAO if simbolo.endswith(m)), "USDT")


def klines_para_linhas(klines: Iterable[list], simbolo: str, intervalo: str, source: str, exchange: str,
                       agora: datetime | None = None) -> list:
    """Converte klines em linhas de price_history, mantendo apenas candles ja fechados."""
    agora_ms = int((agora or datetime.now(timezone.utc)).timestamp() * 1000)
    linhas = []
    for k in klines:
        if k[6] >= agora_ms:  # horario de fechamento ainda no futuro: candle aberto
            continue
        linhas.append({
            "symbol": simbolo,
            "timestamp": datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc).replace(tzinfo=None),
            "open": float(k[1]), "high": float(k[2]), "low": float(k[3]), "close": float(k[4]), "volume": float(k[5]),
            "interval": intervalo, "source": source, "currency": _moeda(simbolo), "exchange": exchange,
        })
    return linhas


class RepositorioPriceHistory:
    """Acesso a tabela price_history via peewee (importado so quando usado)."""

    def __init__(self, tamanho_lote: int = 1000):
        from database.model_nocapital import PriceHistory
        self.modelo, self.tamanho_lote = PriceHistory, tamanho_lote

    def ultimo_timestamp(self, simbolo, intervalo, source, exchange):
        from peewee import fn
        m = self.modelo
        return (m.select(fn.MAX(m.timestamp))
                 .where((m.symbol == simbolo) & (m.interval == intervalo) & (m.source == source) & (m.exchange == exchange))
                 .scalar())

    def gravar(self, linhas):
        from uuid import uuid4
        m = self.modelo
        with m._meta.database.atomic():
            for i in range(0, len(linhas), self.tamanho_lote):
                lote = [{"id": str(uuid4()), **linha} for linha in linhas[i:i + self.tamanho_lote]]
                # preserve gera "col = VALUES(col)" no ON DUPLICATE KEY UPDATE do MySQL
                (m.insert_many(lote)
                  .on_conflict(preserve=[m.open, m.high, m.low, m.close, m.volume])
                  .execute())
        return len(linhas)


def capturar(cfg: CapturaConfig, repositorio=None, http_get=_http_get_json, agora: datetime | None = None) -> dict:
    """Captura os candles pendentes de cada simbolo e intervalo. Retorna {(simbolo, intervalo): gravados}."""
    repositorio = repositorio or RepositorioPriceHistory()
    agora = agora or datetime.now(timezone.utc)
    fim_ms = int(agora.timestamp() * 1000)
    resultado = {}
    for simbolo in cfg.simbolos:
        for intervalo in cfg.intervalos:
            ultimo = repositorio.ultimo_timestamp(simbolo, intervalo, cfg.source, cfg.exchange)
            inicio_ms = 0 if ultimo is None else int(ultimo.replace(tzinfo=timezone.utc).timestamp() * 1000) + DURACAO_MS[intervalo]
            klines = buscar_klines(simbolo, intervalo, inicio_ms, fim_ms, http_get=http_get)
            linhas = klines_para_linhas(klines, simbolo, intervalo, cfg.source, cfg.exchange, agora=agora)
            resultado[(simbolo, intervalo)] = repositorio.gravar(linhas) if linhas else 0
    return resultado


def proxima_execucao(cfg: CapturaConfig, agora: datetime) -> datetime:
    """Proximo instante (UTC) em que o horario configurado ocorre no fuso configurado."""
    fuso = ZoneInfo(cfg.fuso)
    hora, minuto = (int(x) for x in cfg.horario.split(":"))
    local = agora.astimezone(fuso)
    alvo = local.replace(hour=hora, minute=minuto, second=0, microsecond=0)
    if alvo <= local:
        alvo = (local + timedelta(days=1)).replace(hour=hora, minute=minuto, second=0, microsecond=0)
    return alvo.astimezone(timezone.utc)


def _registrar(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC] {msg}", flush=True)


def executar_uma_vez(env_path: Path = ENV_PADRAO, forcar: bool = False) -> None:
    if not forcar and not captura_ativa(env_path):
        _registrar("captura desativada (CAPTURA_DIARIA_ATIVA=false); nada feito")
        return
    resultado = capturar(CapturaConfig.from_env(env_path))
    for (simbolo, intervalo), n in resultado.items():
        _registrar(f"{simbolo} {intervalo}: {n} candles gravados")


def daemon(env_path: Path = ENV_PADRAO) -> None:
    _registrar("agendador da captura diaria iniciado")
    while True:
        cfg = CapturaConfig.from_env(env_path)
        alvo = proxima_execucao(cfg, datetime.now(timezone.utc))
        _registrar(f"proxima execucao: {alvo:%Y-%m-%d %H:%M} UTC ({cfg.horario} {cfg.fuso})")
        while datetime.now(timezone.utc) < alvo:
            time.sleep(min(300, max(1, (alvo - datetime.now(timezone.utc)).total_seconds())))
        try:
            executar_uma_vez(env_path)
        except Exception as erro:  # o agendador nao pode morrer por uma falha de rede
            _registrar(f"falha na captura: {erro}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Captura diaria de candles da Binance para price_history.")
    modo = ap.add_mutually_exclusive_group(required=True)
    modo.add_argument("--agora", action="store_true", help="executa uma vez, se a captura estiver ativa")
    modo.add_argument("--forcar", action="store_true", help="executa uma vez, ignorando CAPTURA_DIARIA_ATIVA")
    modo.add_argument("--daemon", action="store_true", help="executa todo dia no horario configurado")
    ap.add_argument("--env", default=str(ENV_PADRAO), help="caminho do .env")
    args = ap.parse_args()

    if args.daemon:
        daemon(Path(args.env))
    else:
        executar_uma_vez(Path(args.env), forcar=args.forcar)


if __name__ == "__main__":
    main()
