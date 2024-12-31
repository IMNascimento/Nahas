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
    

    @staticmethod
    def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """
        Calcula as Bandas de Bollinger.
        
        :param series: Série temporal dos preços.
        :param period: Período para o cálculo da SMA. Padrão é 20.
        :param std_dev: Multiplicador do desvio padrão. Padrão é 2.0.
        :return: DataFrame com colunas para as bandas superior, inferior e SMA.
        """
        sma = TechnicalIndicators.sma(series, period)
        std = series.rolling(window=period).std()
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        return pd.DataFrame({
            f'Bollinger_Upper_{period}': upper_band,
            f'Bollinger_Lower_{period}': lower_band,
            f'Bollinger_Middle_{period}': sma
        })
    

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calcula o Average True Range (ATR).
        
        :param high: Série dos preços máximos.
        :param low: Série dos preços mínimos.
        :param close: Série dos preços de fechamento.
        :param period: Período para o cálculo do ATR. Padrão é 14.
        :return: Série contendo os valores do ATR.
        """
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
        """
        Calcula o Average Directional Index (ADX) e os Indicadores Direcionais (+DI e -DI).
        
        :param high: Série dos preços máximos.
        :param low: Série dos preços mínimos.
        :param close: Série dos preços de fechamento.
        :param period: Período para o cálculo do ADX. Padrão é 14.
        :return: DataFrame com colunas para +DI, -DI e ADX.
        """
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        dm_plus = high.diff()
        dm_minus = low.diff()

        dm_plus[dm_plus < 0] = 0
        dm_minus[dm_minus > 0] = 0
        dm_minus = dm_minus.abs()

        tr = true_range.rolling(window=period).sum()
        di_plus = 100 * (dm_plus.rolling(window=period).sum() / tr)
        di_minus = 100 * (dm_minus.rolling(window=period).sum() / tr)

        dx = 100 * (abs(di_plus - di_minus) / (di_plus + di_minus))
        adx = dx.rolling(window=period).mean()

        return pd.DataFrame({
            '+DI': di_plus,
            '-DI': di_minus,
            'ADX': adx
        })
    
    @staticmethod
    def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
        """
        Calcula o Stochastic Oscillator.
        
        :param high: Série dos preços máximos.
        :param low: Série dos preços mínimos.
        :param close: Série dos preços de fechamento.
        :param period: Período para o cálculo do Stochastic. Padrão é 14.
        :return: DataFrame com %K e %D.
        """
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=3).mean()  # Média móvel de 3 períodos para suavizar
        
        return pd.DataFrame({
            '%K': k,
            '%D': d
        })
    
    @staticmethod
    def fibonacci_retracement(high: pd.Series, low: pd.Series) -> pd.DataFrame:
        """
        Calcula os níveis de retração de Fibonacci para um intervalo de preços.
        
        :param high: Série dos preços máximos.
        :param low: Série dos preços mínimos.
        :return: DataFrame com os níveis de retração de Fibonacci.
        """
        diff = high - low
        retracements = {
            'Fibo_0.0%': low,
            'Fibo_23.6%': high - 0.236 * diff,
            'Fibo_38.2%': high - 0.382 * diff,
            'Fibo_50.0%': high - 0.500 * diff,
            'Fibo_61.8%': high - 0.618 * diff,
            'Fibo_100.0%': high
        }
        return pd.DataFrame(retracements)
    
    @staticmethod
    def fibonacci_projection(start: pd.Series, end: pd.Series, retracement: pd.Series) -> pd.DataFrame:
        """
        Calcula os níveis de projeção de Fibonacci.
        
        :param start: Série com os valores iniciais (A).
        :param end: Série com os valores finais (B).
        :param retracement: Série com os valores de retração (C).
        :return: DataFrame com os níveis de projeção de Fibonacci.
        """
        difference = end - start
        projections = {
            'Proj_Fibo_0.0%': retracement,
            'Proj_Fibo_61.8%': retracement + difference * 0.618,
            'Proj_Fibo_100.0%': retracement + difference,
            'Proj_Fibo_161.8%': retracement + difference * 1.618,
            'Proj_Fibo_261.8%': retracement + difference * 2.618,
            'Proj_Fibo_423.6%': retracement + difference * 4.236,
        }
        return pd.DataFrame(projections)