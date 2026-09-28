from peewee import (
    CharField, DateTimeField, FloatField, TextField
)
import pandas as pd
import json

from database.model_base import BaseModel

# JSONField para MySQL 5.7+
try:
    from playhouse.mysql_ext import JSONField  # type: ignore
    _JSONField = JSONField
except Exception:
    _JSONField = TextField  # fallback: armazena JSON serializado

class PriceHistory(BaseModel):
    id = CharField(36, primary_key=True)            # varchar(36) PK (UUID em string)
    symbol = CharField(20, index=True)
    timestamp = DateTimeField(index=True)
    open = FloatField()
    high = FloatField()
    low = FloatField()
    close = FloatField()
    volume = FloatField()

    interval = CharField(10, index=True)            # ex: '1m','1h','1d'
    source = CharField(20, null=True, index=True)   # ex: 'binance_api'
    currency = CharField(10, null=True, index=True) # ex: 'USDT','USD','BRL'
    exchange = CharField(20, null=True, index=True) # ex: 'BINANCE','BYBIT'

    extra_data = _JSONField(null=True)

    class Meta:
        table_name = 'price_history'
        # Índice único para evitar duplicidade de candles por origem
        indexes = (
            (('symbol', 'interval', 'exchange', 'source', 'timestamp'), True),
        )

    # -------------------------
    # Helpers internos
    # -------------------------
    @staticmethod
    def _apply_filters(q, symbol=None, interval=None, source=None, exchange=None):
        if symbol:
            q = q.where(PriceHistory.symbol == symbol)
        if interval:
            q = q.where(PriceHistory.interval == interval)
        if source:
            q = q.where(PriceHistory.source == source)
        if exchange:
            q = q.where(PriceHistory.exchange == exchange)
        return q

    @staticmethod
    def _df(query, columns=None):
        rows = list(query.dicts())
        if not rows:
            print("Nenhum dado encontrado no banco.")
            return pd.DataFrame()

        # Normaliza extra_data quando for TextField (fallback)
        for r in rows:
            if 'extra_data' in r and isinstance(r['extra_data'], str):
                try:
                    r['extra_data'] = json.loads(r['extra_data'])
                except Exception:
                    pass

        df = pd.DataFrame(rows)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        if columns:
            cols = [c for c in columns if c in df.columns]
            return df[cols]
        return df

    # -------------------------
    # Consultas em DataFrame
    # -------------------------
    @classmethod
    def get_to_date(cls, end_date, symbol=None, interval=None, source=None, exchange=None, with_extra=False):
        q = (cls
             .select()
             .where(cls.timestamp <= end_date)
             .order_by(cls.timestamp))
        q = cls._apply_filters(q, symbol, interval, source, exchange)
        base_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        meta_cols = ["symbol", "interval", "source", "currency", "exchange"]
        cols = base_cols + meta_cols + (["extra_data"] if with_extra else [])
        return cls._df(q, columns=cols)

    @classmethod
    def get_from_date(cls, start_date, symbol=None, interval=None, source=None, exchange=None, only_ohlcv=True):
        q = (cls
             .select()
             .where(cls.timestamp >= start_date)
             .order_by(cls.timestamp))
        q = cls._apply_filters(q, symbol, interval, source, exchange)
        if only_ohlcv:
            return cls._df(q, columns=["open", "high", "low", "close", "volume"])
        return cls._df(q)

    @classmethod
    def get_between_dates(cls, start_date, end_date, symbol=None, interval=None, source=None, exchange=None, with_meta=False):
        q = (cls
             .select()
             .where((cls.timestamp >= start_date) & (cls.timestamp <= end_date))
             .order_by(cls.timestamp))
        q = cls._apply_filters(q, symbol, interval, source, exchange)
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        if with_meta:
            cols += ["symbol", "interval", "source", "currency", "exchange"]
        return cls._df(q, columns=cols)

    # -------------------------
    # Upsert (MySQL: ON DUPLICATE KEY UPDATE)
    # -------------------------
    @classmethod
    def upsert_row(cls, data: dict):
        """
        Insere/atualiza linha garantindo unicidade por
        (symbol, interval, exchange, source, timestamp).

        Requer MySQL 5.7+ (ou MariaDB com suporte) para JSON e ON DUPLICATE KEY.
        """
        # Serializa extra_data no fallback TextField
        if 'extra_data' in data and isinstance(cls.extra_data, TextField) and isinstance(data['extra_data'], (dict, list)):
            data = {**data, 'extra_data': json.dumps(data['extra_data'])}

        update_map = {
            cls.open: data.get('open'),
            cls.high: data.get('high'),
            cls.low: data.get('low'),
            cls.close: data.get('close'),
            cls.volume: data.get('volume'),
            cls.currency: data.get('currency'),
            cls.extra_data: data.get('extra_data'),
        }

        return (cls
                .insert(data)
                .on_conflict(  # em MySQL, Peewee gera ON DUPLICATE KEY UPDATE
                    update=update_map
                )
                .execute())