import pandas as pd
import numpy as np

class TechnicalIndicators:
    """
    Classe que implementa métodos estáticos para calcular diversos indicadores técnicos.
    """

    @staticmethod
    def sma(series: pd.Series, period: int = 14) -> pd.Series:
        """
        Calcula a Média Móvel Simples (SMA).
        
        :param series: Série temporal dos preços.
        :param period: Período para o cálculo da SMA. Padrão é 14.
        :return: Série contendo os valores da SMA.
        """
        return series.rolling(window=period).mean()

    @staticmethod
    def ema(series: pd.Series, period: int = 14) -> pd.Series:
        """
        Calcula a Média Móvel Exponencial (EMA).
        
        :param series: Série temporal dos preços.
        :param period: Período para o cálculo da EMA. Padrão é 14.
        :return: Série contendo os valores da EMA.
        """
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def envelopes(series: pd.Series, period: int = 14, percent: float = 3.0) -> pd.DataFrame:
        """
        Calcula Envelopes de preço ao redor da SMA.
        
        :param series: Série temporal dos preços.
        :param period: Período para o cálculo da SMA. Padrão é 14.
        :param percent: Percentual para calcular os envelopes superior e inferior. Padrão é 3%.
        :return: DataFrame com colunas de upper e lower envelopes.
        """
        sma = TechnicalIndicators.sma(series, period)
        upper_envelope = sma * (1 + percent / 100)
        lower_envelope = sma * (1 - percent / 100)
        return pd.DataFrame({
            f'Upper_Envelope_{period}_{percent}%': upper_envelope,
            f'Lower_Envelope_{period}_{percent}%': lower_envelope
        })

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """
        Calcula o Índice de Força Relativa (RSI).
        
        :param series: Série temporal dos preços.
        :param period: Período para o cálculo do RSI. Padrão é 14.
        :return: Série contendo os valores do RSI.
        """
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(series: pd.Series, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9) -> pd.DataFrame:
        """
        Calcula o MACD (Moving Average Convergence Divergence).
        
        :param series: Série temporal dos preços.
        :param fastperiod: Período rápido para a EMA. Padrão é 12.
        :param slowperiod: Período lento para a EMA. Padrão é 26.
        :param signalperiod: Período do sinal do MACD. Padrão é 9.
        :return: DataFrame contendo MACD, Signal e Histograma.
        """
        fast_ema = TechnicalIndicators.ema(series, period=fastperiod)
        slow_ema = TechnicalIndicators.ema(series, period=slowperiod)
        macd = fast_ema - slow_ema
        signal = macd.ewm(span=signalperiod, adjust=False).mean()
        hist = macd - signal

        return pd.DataFrame({
            'MACD': macd,
            'MACD_Signal': signal,
            'MACD_Hist': hist
        })