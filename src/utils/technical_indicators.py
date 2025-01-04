import pandas as pd
import numpy as np

class TechnicalIndicators:
    """
    Classe que implementa métodos estáticos para calcular diversos indicadores técnicos.
    """

    @staticmethod
    def process_indicators(data: pd.DataFrame, indicators_to_apply: dict) -> pd.DataFrame:
        """
        Aplica os indicadores financeiros solicitados ao DataFrame.

        :param data: DataFrame contendo os dados históricos.
        :param indicators_to_apply: Dicionário estruturado com indicadores e seus parâmetros.
                                    Exemplo: {
                                        "sma": [{"period": 14}],
                                        "macd": [{}],
                                        "bollinger_bands": [{"period": 20, "std_dev": 2.0}]
                                    }
        :return: DataFrame com os indicadores calculados adicionados.
        """
        data = data.copy()

        # Mapeamento de indicadores e suas dependências de colunas
        indicator_mapping = {
            "sma": {"func": TechnicalIndicators.sma, "columns": ["close"]},
            "ema": {"func": TechnicalIndicators.ema, "columns": ["close"]},
            "envelopes": {"func": TechnicalIndicators.envelopes, "columns": ["close"]},
            "rsi": {"func": TechnicalIndicators.rsi, "columns": ["close"]},
            "macd": {"func": TechnicalIndicators.macd, "columns": ["close"]},
            "bollinger_bands": {"func": TechnicalIndicators.bollinger_bands, "columns": ["close"]},
            "atr": {"func": TechnicalIndicators.atr, "columns": ["high", "low", "close"]},
            "adx": {"func": TechnicalIndicators.adx, "columns": ["high", "low", "close"]},
            "stochastic": {"func": TechnicalIndicators.stochastic_oscillator, "columns": ["high", "low", "close"]},
            "fibonacci_retracement": {"func": TechnicalIndicators.fibonacci_retracement, "columns": ["high", "low"]},
            "fibonacci_projection": {"func": TechnicalIndicators.fibonacci_projection,"columns": {"open","close","low"}},
        }

        for indicator_name, params_list in indicators_to_apply.items():
            if indicator_name not in indicator_mapping:
                print(f"Indicador '{indicator_name}' não suportado.")
                continue

            indicator_info = indicator_mapping[indicator_name]
            func = indicator_info["func"]
            required_columns = indicator_info["columns"]

            # Validar a presença de colunas necessárias
            if not all(col in data.columns for col in required_columns):
                print(f"Colunas insuficientes para calcular '{indicator_name}'. Necessárias: {required_columns}")
                continue

            for params in params_list:
                try:
                    # Extraia os dados necessários para o cálculo
                    args = [data[col] for col in required_columns]

                    # Calcule o indicador
                    result = func(*args, **params)

                    # Adicione o resultado ao DataFrame
                    if isinstance(result, pd.DataFrame):
                        # Usa o nome das colunas geradas pelo indicador
                        data = pd.concat([data, result], axis=1)
                    else:
                        # Nome do indicador
                        param_suffix = "_".join([f"{k}_{v}" for k, v in params.items()])
                        column_name = f"{indicator_name}_{param_suffix}" if param_suffix else indicator_name
                        data[column_name] = result

                except Exception as e:
                    print(f"Erro ao calcular o indicador '{indicator_name}' com parâmetros {params}: {e}")

        # Substituir NaN por média (colunas originais) ou mínimo (colunas de indicador)
        for col in data.columns:
            if data[col].isna().any():
                if col in ['open', 'high', 'low', 'close', 'volume']:
                    # Para colunas originais
                    data[col].fillna(data[col].mean(), inplace=True)
                else:
                    # Para colunas de indicadores
                    data[col].fillna(data[col].min(), inplace=True)

        return data


    @staticmethod
    def sma(series: pd.Series, period: int = 14) -> pd.Series:
        """
        Calcula a Média Móvel Simples (SMA).
        
        :param series: Série temporal dos preços.
        :param period: Período para o cálculo da SMA. Padrão é 14.
        :return: DataFrame contendo a SMA com uma coluna nomeada.
        """
        sma = series.rolling(window=period).mean()
        return pd.DataFrame({'SMA': sma})

    @staticmethod
    def ema(series: pd.Series, period: int = 14) -> pd.Series:
        """
        Calcula a Média Móvel Exponencial (EMA).
        
        :param series: Série temporal dos preços.
        :param period: Período para o cálculo da EMA. Padrão é 14.
        :return: DataFrame contendo a EMA com uma coluna nomeada.
        """
        ema = series.ewm(span=period, adjust=False).mean()
        return pd.DataFrame({'EMA': ema})

    @staticmethod
    def envelopes(series: pd.Series, period: int = 14, percent: float = 3.0) -> pd.DataFrame:
        """
        Calcula Envelopes de preço ao redor da SMA.
        
        :param series: Série temporal dos preços.
        :param period: Período para o cálculo da SMA. Padrão é 14.
        :param percent: Percentual para calcular os envelopes superior e inferior. Padrão é 3%.
        :return: DataFrame com colunas de upper e lower envelopes.
        """
        sma_df = TechnicalIndicators.sma(series, period)
        sma = sma_df['SMA']
        upper_envelope = sma * (1 + percent / 100)
        lower_envelope = sma * (1 - percent / 100)
        return pd.DataFrame({
            'Upper_Envelope': upper_envelope,
            'Lower_Envelope': lower_envelope
        }, index=series.index)
   
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
        rsi = 100 - (100 / (1 + rs))
        return pd.DataFrame({'RSI': rsi})

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
        fast_ema_df = TechnicalIndicators.ema(series, period=fastperiod)
        slow_ema_df = TechnicalIndicators.ema(series, period=slowperiod)
        fast_ema = fast_ema_df['EMA']  
        slow_ema = slow_ema_df['EMA']  
        
        macd = fast_ema - slow_ema
        signal = macd.ewm(span=signalperiod, adjust=False).mean()
        hist = macd - signal
        return pd.DataFrame({
            'MACD': macd,
            'MACD_Signal': signal,
            'MACD_Hist': hist
        }, index=series.index)

    @staticmethod
    def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """
        Calcula as Bandas de Bollinger.
        
        :param series: Série temporal dos preços.
        :param period: Período para o cálculo da SMA. Padrão é 20.
        :param std_dev: Multiplicador do desvio padrão. Padrão é 2.0.
        :return: DataFrame com colunas para as bandas superior, inferior e SMA.
        """
        sma_df = TechnicalIndicators.sma(series, period)
        sma = sma_df['SMA']
        std = series.rolling(window=period).std()
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        return pd.DataFrame({
            'Bollinger_Upper': upper_band,
            'Bollinger_Lower': lower_band,
            'Bollinger_Middle': sma
        }, index=series.index)
    

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
        
        return pd.DataFrame({'ATR': atr})
    
    
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
    def fibonacci_projection(open: pd.Series, close: pd.Series, low: pd.Series) -> pd.DataFrame:
        """
        Calcula os níveis de projeção de Fibonacci.
        
        :param open: Série com os valores iniciais (A).
        :param close: Série com os valores finais (B).
        :param low: Série com os valores de retração (C).
        :return: DataFrame com os níveis de projeção de Fibonacci.
        """
        difference = close - open
        projections = {
            'Proj_Fibo_0.0%': low,
            'Proj_Fibo_61.8%': low + difference * 0.618,
            'Proj_Fibo_100.0%': low + difference,
            'Proj_Fibo_161.8%': low + difference * 1.618,
            'Proj_Fibo_261.8%': low + difference * 2.618,
            'Proj_Fibo_423.6%': low + difference * 4.236,
        }
        return pd.DataFrame(projections)