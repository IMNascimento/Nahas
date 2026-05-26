from datetime import datetime, timedelta, timezone
from uuid import uuid4
from typing import Iterable, Optional, Dict

import pandas as pd
from peewee import fn

from database.model_base import db
from database.model_nocapital import PriceHistory  # ajuste o import para onde está sua model
from services.binance import BinanceData  # ajuste o import para sua classe acima

# Mapeia seus rótulos de intervalo para o que a Binance espera
# (você pode adicionar outros: '1m','5m','15m','4h','1d', etc.)
BINANCE_INTERVALS: Dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}

# Deltas para pular o último candle salvo e começar no próximo
INTERVAL_DELTAS: Dict[str, timedelta] = {
    "1m": timedelta(minutes=1),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
    "1d": timedelta(days=1),
}

EXCHANGE_DEFAULT = "BINANCE"
SOURCE_DEFAULT = "binance_api"

def _currency_from_symbol(symbol: str) -> str:
    # Heurística simples: tenta extrair sufixos comuns
    for quote in ("USDT", "USD", "BRL", "BUSD", "FDUSD"):
        if symbol.endswith(quote):
            return quote
    return "USDT"

def _last_timestamp_utc(symbol: str, interval: str,
                        exchange: str = EXCHANGE_DEFAULT,
                        source: str = SOURCE_DEFAULT) -> Optional[datetime]:
    q = (PriceHistory
         .select(fn.MAX(PriceHistory.timestamp).alias("last_ts"))
         .where(
             (PriceHistory.symbol == symbol) &
             (PriceHistory.interval == interval) &
             (PriceHistory.exchange == exchange) &
             (PriceHistory.source == source)
         ))
    row = q.dicts().first()
    last_ts = row["last_ts"] if row else None
    return last_ts

def _naive_utc(dt: pd.Timestamp | datetime) -> datetime:
    """Converte para datetime UTC *naive* (sem tzinfo), consistente com a maioria dos DateTimeField."""
    if isinstance(dt, pd.Timestamp):
        if dt.tzinfo is None:
            # assume UTC
            return dt.to_pydatetime().replace(tzinfo=None)
        return dt.tz_convert("UTC").to_pydatetime().replace(tzinfo=None)
    # datetime
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)

def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Garante tipos numéricos
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Timestamp já vem como datetime pela sua BinanceData, mas setamos UTC explícito
    if "timestamp" in df.columns:
        # pandas já converteu a ms -> Timestamp (naive). Marcamos como UTC para normalizar:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df

def fetch_and_upsert_pair(symbol: str, interval: str,
                          exchange: str = EXCHANGE_DEFAULT,
                          source: str = SOURCE_DEFAULT) -> int:
    """
    Busca na Binance os candles faltantes para (symbol, interval) e faz upsert no banco.
    Retorna a quantidade de candles inseridos/atualizados.
    """
    if interval not in BINANCE_INTERVALS:
        raise ValueError(f"Intervalo não suportado: {interval}")

    binance_iv = BINANCE_INTERVALS[interval]
    currency = _currency_from_symbol(symbol)

    # Determina o ponto inicial
    last_ts = _last_timestamp_utc(symbol, interval, exchange, source)

    if last_ts:
        start_dt = last_ts + INTERVAL_DELTAS[interval]
        start_str = start_dt.strftime("%d %b, %Y %H:%M:%S")
    else:
        # "Desde o início" da Binance:
        # Usar uma data bem antiga garante cobertura (BTC/USDT ~2017, ETH/USDT ~2017)
        start_str = "1 Jan, 2015 00:00:00"

    # Coleta
    binance = BinanceData()
    df = binance.get_historical_data(
        symbol=symbol,
        start_str=start_str,
        interval=binance_iv,
        end_str=None  # até "agora"
    )

    if df.empty:
        return 0

    df = _prepare_dataframe(df)

    # Upsert em transação; chunk para não estourar memória/packet
    total = 0
    with db.atomic():
        for _, row in df.iterrows():
            ts = _naive_utc(row["timestamp"])

            data = {
                "id": str(uuid4()),
                "symbol": symbol,
                "timestamp": ts,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "interval": interval,
                "source": source,
                "currency": currency,
                "exchange": exchange,
                # caso queira salvar extras no futuro (número de trades etc.), pode preencher aqui:
                "extra_data": None,
            }

            # Usa sua chave única (symbol, interval, exchange, source, timestamp)
            PriceHistory.upsert_row(data)
            total += 1

    return total

def sync_binance_price_history(
    symbols: Iterable[str] = ("BTCUSDT", "ETHUSDT"),
    intervals: Iterable[str] = ("1m", "1h", "1d"),
    exchange: str = EXCHANGE_DEFAULT,
    source: str = SOURCE_DEFAULT,
) -> dict:
    """
    Sincroniza múltiplos símbolos/intervalos.
    Retorna um dicionário { (symbol, interval): qtd_upserts }.
    """
    results: Dict[tuple, int] = {}
    for symbol in symbols:
        for interval in intervals:
            try:
                count = fetch_and_upsert_pair(symbol, interval, exchange, source)
                results[(symbol, interval)] = count
            except Exception as e:
                # Log simples; adapte para seu logger
                print(f"[ERRO] {symbol} {interval}: {e}")
                results[(symbol, interval)] = -1
    return results

if __name__ == "__main__":
    # Exemplo rápido de execução manual:
    out = sync_binance_price_history(
        symbols=("BTCUSDT", "ETHUSDT"),
        intervals=("1m", "1h"),  # ajuste como quiser
    )
    print(out)
