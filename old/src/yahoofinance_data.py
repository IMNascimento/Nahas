import yfinance as yf
import pandas as pd

class YahooFinanceData:
    """
    Classe responsável por coletar dados históricos de ativos do Yahoo Finance.
    """

    def get_historical_data(self, symbol: str, start_date: str, end_date: str, interval: str = '1h') -> pd.DataFrame:
        """
        Obtém dados históricos de um ativo no Yahoo Finance em um intervalo de tempo específico.

        :param symbol: Símbolo do ativo no Yahoo Finance, ex: 'BTC-USD' para Bitcoin em USD.
        :param start_date: Data de início no formato 'YYYY-MM-DD'.
        :param end_date: Data de término no formato 'YYYY-MM-DD'.
        :param interval: Intervalo de tempo (ex: '1h' para 1 hora, '1d' para 1 dia).
        :return: Um DataFrame contendo os dados históricos.
        """
        # Coletar dados históricos usando yfinance
        data = yf.download(symbol, start=start_date, end=end_date, interval=interval)

        # Verifica se dados foram coletados
        if data.empty:
            raise ValueError(f"Nenhum dado encontrado para {symbol} no período especificado.")

        return data

