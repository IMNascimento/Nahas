from binance.client import Client
from datetime import datetime
import pandas as pd

class BinanceData:
    """
    Classe responsável pela conexão com a API da Binance e coleta de dados históricos de criptomoedas.
    """

    def __init__(self, api_key: str, api_secret: str):
        """
        Inicializa a classe carregando as credenciais da API da Binance do arquivo .env.
        """

        self.__api_key = api_key
        self.__api_secret = api_secret

        # Inicializa o cliente da Binance com as credenciais
        self.client = Client(self.__api_key, self.__api_secret)

    def get_historical_data(self, symbol: str, start_str: str, interval: str, end_str: str = None) -> pd.DataFrame:
        """
        Obtém dados históricos da criptomoeda em um intervalo de tempo específico.

        :param symbol: Par de criptomoedas, ex: 'BTCUSDT'.
        :param start_str: Data de início (em string), ex: '1 Jan, 2020'.
        :param interval: Intervalo de tempo, ex: Client.KLINE_INTERVAL_1HOUR para 1 hora.
        :param end_str: Data final (opcional), ex: '1 Jan, 2023'.
        :return: Um DataFrame contendo os dados históricos.
        """
        # Coleta os dados históricos de candles (klines)
        klines = self.client.get_historical_klines(symbol, interval, start_str, end_str)

        # Criar um DataFrame a partir dos dados coletados
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 
            'close_time', 'quote_asset_volume', 'number_of_trades', 
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])

        # Converter o timestamp para um formato de data legível
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Retornar apenas as colunas mais importantes
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]