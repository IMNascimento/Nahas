import pandas as pd
from utils.technical_indicators import TechnicalIndicators
from utils.plotter import Plotter
from models.db.model_binance import HourlyQuote

def main():
    # 1. Carregar os dados do banco de dados
    print("Carregando dados do banco de dados...")
    start_date = "2023-01-01 00:00:00"
    end_date = "2024-09-14 23:59:59"
    data = HourlyQuote.get_to_date(end_date)

    if data.empty:
        print("Nenhum dado foi carregado. Encerrando...")
        return

    # 2. Calcular os indicadores financeiros
    print("Calculando indicadores financeiros...")
    data['SMA_9'] = TechnicalIndicators.sma(data['close'], period=9)
    data['SMA_14'] = TechnicalIndicators.sma(data['close'], period=14)
    data['SMA_21'] = TechnicalIndicators.sma(data['close'], period=21)
    data['EMA_9'] = TechnicalIndicators.ema(data['close'], period=9)
    data['EMA_14'] = TechnicalIndicators.ema(data['close'], period=14)
    data['EMA_21'] = TechnicalIndicators.ema(data['close'], period=21)
    data['RSI_14'] = TechnicalIndicators.rsi(data['close'], period=14)
    macd = TechnicalIndicators.macd(data['close'])
    data['MACD'] = macd['MACD']
    data['MACD_Signal'] = macd['MACD_Signal']
    data['MACD_Hist'] = macd['MACD_Hist']
    bollinger = TechnicalIndicators.bollinger_bands(data['close'])
    data['Bollinger_Upper'] = bollinger[f'Bollinger_Upper_20']
    data['Bollinger_Lower'] = bollinger[f'Bollinger_Lower_20']
    data['ATR_14'] = TechnicalIndicators.atr(data['high'], data['low'], data['close'], period=14)


    # Adicionar retração de Fibonacci
    fibo_retracements = TechnicalIndicators.fibonacci_retracement(data['high'], data['low'])
    data = pd.concat([data, fibo_retracements], axis=1)

    # Adicionar projeção de Fibonacci
    # Usando valores de exemplo para start, end e retracement
    fibo_projections = TechnicalIndicators.fibonacci_projection(
        start=data['low'], 
        end=data['high'], 
        retracement=data['close']
    )
    data = pd.concat([data, fibo_projections], axis=1)
    # 3. Remover valores NaN gerados durante o cálculo dos indicadores
    data = data.dropna()

    # 4. Selecionar as colunas para a matriz de correlação
    #columns_to_correlate = [
    #    "open", "high", "low", "close", "volume",
    #    "SMA_14", "EMA_14","SMA_9", "EMA_9","SMA_21", "EMA_21", "RSI_14", "MACD", "MACD_Signal",
    #    "MACD_Hist", "Bollinger_Upper", "Bollinger_Lower", "ATR_14"
    #]
    #columns_to_correlate = [
    #    "open", "high", "low", "close", "volume",
    #    "Fibo_23.6%", "Fibo_38.2%", "Fibo_50.0%", "Fibo_61.8%", "Fibo_100.0%",
    #    "Proj_Fibo_61.8%", "Proj_Fibo_100.0%", "Proj_Fibo_161.8%"
    #]

    columns_to_correlate = [
        "open", "high", "low", "close", "volume",
       "SMA_14", "EMA_14","SMA_9", "EMA_9","SMA_21", "EMA_21", "RSI_14", "MACD", "MACD_Signal",
        "MACD_Hist", "Bollinger_Upper", "Bollinger_Lower", "ATR_14",
        "Fibo_23.6%", "Fibo_38.2%", "Fibo_50.0%", "Fibo_61.8%", "Fibo_100.0%",
        "Proj_Fibo_61.8%", "Proj_Fibo_100.0%", "Proj_Fibo_161.8%"
    ]


    # Garantir que todas as colunas existem no DataFrame
    available_columns = [col for col in columns_to_correlate if col in data.columns]
    print("Colunas disponíveis para correlação:", available_columns)

    # 5. Plotar a matriz de correlação
    print("Gerando matriz de correlação...")
    plotter = Plotter()
    plotter.plot_correlation_matrix(
        df=data,
        columns=available_columns,
        save_path="correlation_matrix.png"
    )
    print("Matriz de correlação salva como 'correlation_matrix.png'.")

if __name__ == "__main__":
    main()