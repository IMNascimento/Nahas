import MetaTrader5 as mt5
from datetime import datetime
from config.settings import Settings
import pandas as pd


class MetaTraderData:
    """
    Classe responsável pela conexão com MetaTrader 5 e coleta de dados de mercado (ticks e candles).
    """

    def __init__(self):
        """
        Inicializa a classe com as credenciais de login e configura o servidor MetaTrader 5.
        
        :param login: O ID de login na plataforma MetaTrader 5.
        :param password: A senha associada ao login.
        :param server: O endereço do servidor MetaTrader 5.
        """
        self.__login = Settings.USER_MT5
        self.__password = Settings.PASSWORD_MT5
        self.__server = Settings.SERVER_MT5
        self.__is_initialized = False

    def initialize(self):
        """
        Inicializa a conexão com o MetaTrader 5 usando as credenciais fornecidas.
        
        :raises RuntimeError: Se a conexão já estiver inicializada.
        """
        if not self.__is_initialized:
            mt5.initialize(login=self.__login, password=self.__password, server=self.__server)
            self.__is_initialized = True
            print("MetaTrader 5 initialized.")
        else:
            print("MetaTrader 5 is already initialized.")

    def shutdown(self):
        """
        Finaliza a conexão com o MetaTrader 5.
        """
        if self.__is_initialized:
            mt5.shutdown()
            self.__is_initialized = False
            print("MetaTrader 5 shutdown.")
        else:
            print("MetaTrader 5 is not initialized.")

    def get_ticks(self, symbol: str, start_date: datetime, end_date: datetime, tick_type=mt5.COPY_TICKS_TRADE) -> pd.DataFrame:
        """
        Obtém dados de ticks para o símbolo especificado em um intervalo de datas.

        :param symbol: O símbolo do ativo (por exemplo, "ITSA4").
        :param start_date: Data de início da coleta de dados.
        :param end_date: Data de término da coleta de dados.
        :param tick_type: Tipo de tick a ser copiado (padrão é mt5.COPY_TICKS_TRADE).
        :return: Um DataFrame do pandas contendo os dados de ticks.
        :raises ValueError: Se falhar ao obter os ticks para o símbolo especificado.
        """
        self.__check_initialization()

        ticks = mt5.copy_ticks_range(symbol, start_date, end_date, tick_type)
        if ticks is None:
            raise ValueError(f"Failed to get ticks for {symbol}.")
        return pd.DataFrame(ticks)

    def get_candles(self, symbol: str, timeframe: int, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Obtém dados de candles (barras) para o símbolo especificado em um intervalo de datas.

        :param symbol: O símbolo do ativo (por exemplo, "ITSA4").
        :param timeframe: O período do candle, por exemplo, mt5.TIMEFRAME_D1 para diário.
        :param start_date: Data de início da coleta de dados.
        :param end_date: Data de término da coleta de dados.
        :return: Um DataFrame do pandas contendo os dados de candles.
        :raises ValueError: Se falhar ao obter os candles para o símbolo especificado.
        """
        self.__check_initialization()

        candles = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
        if candles is None:
            raise ValueError(f"Failed to get candles for {symbol}.")
        return pd.DataFrame(candles)

    def __check_initialization(self):
        """
        Verifica se a conexão com o MetaTrader 5 foi inicializada.

        :raises RuntimeError: Se a conexão com o MetaTrader 5 não estiver inicializada.
        """
        if not self.__is_initialized:
            raise RuntimeError("MetaTrader 5 is not initialized. Call 'initialize()' first.")

