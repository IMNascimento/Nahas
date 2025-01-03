from models.db.model_binance import HourlyQuote
from keras.models import load_model
from models.lstm_model import CustomLSTMTrainer
from data.data_processing import DataProcessor
from utils.technical_indicators import TechnicalIndicators
from config.settings import Settings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os


def save_results_to_csv(results_df):
    """
    Salva os resultados em um arquivo CSV com o nome baseado no modelo.
    """
    model_name = os.path.basename(Settings.LOAD_MODEL).replace(".h5", "")  # Extrai o nome do modelo sem extensão
    output_path = f"result/csv/testes/{model_name}_results.csv"  # Define o caminho de saída
    # Cria o diretório, se necessário
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)  # Salva o DataFrame em CSV
    print(f"Resultados salvos em: {output_path}")

# Função para testar o modelo
def test_model():
    """
    Testa o modelo treinado com os dados do banco e armazena as previsões.
    """
    # Carregar o modelo e scaler
    model = CustomLSTMTrainer()
    model.loading_model(Settings.LOAD_MODEL)
    processor = DataProcessor(window_size=Settings.WINDOW_SIZE)
    processor.load_scaler()

    # Carregar os dados do banco
    data = HourlyQuote.get_from_date(Settings.START_DATE)
    if data is None or len(data) < Settings.WINDOW_SIZE:
        print("Dados insuficientes para realizar a previsão.")
        return
    data = TechnicalIndicators.process_indicators(data, Settings.INDICATORS_APPLY)
    
    # Converter timestamps para numérico (se necessário)
    if 'timestamp' in data.columns:
        data['timestamp'] = data['timestamp'].apply(lambda x: x.timestamp())

    # Preparar o array para armazenar os resultados
    results = []

    # Iterar pelas janelas de dados para fazer as previsões
    for i in range(len(data) - Settings.WINDOW_SIZE):
        # Criar uma janela de 48 horas
        window = data.iloc[i:i + Settings.WINDOW_SIZE]
        real_value = data.iloc[i + Settings.WINDOW_SIZE]['close']  # Valor real para a próxima hora

        # Normalizar e preparar os dados para o modelo
        features = window[Settings.RELEVANT_COLUMNS].values  # Apenas features relevantes
        print(f"TESTE: Dimensões da janela antes da previsão: {features.shape}")
        features_scaled = processor.scaler_X.transform(features)
        print(f"TESTE: Dimensões após normalização: {features_scaled.shape}")
        X_input = features_scaled.reshape(1, features_scaled.shape[0], features_scaled.shape[1])
        print(f"TESTE: Dimensões de entrada para o modelo: {X_input.shape}")
        # Fazer a previsão
        predicted_scaled = model.predict(X_input)
        
        print(f"TESTE: Dimensões do resultado da previsão escalada: {predicted_scaled.shape}")
        # Verificar o formato de saída e desnormalizar corretamente
        print(f"Formato de predicted_scaled: {predicted_scaled.shape}")

        predicted_value = processor.inverse_transform(predicted_scaled)
        print(f"TESTE: Valor previsto desnormalizado: {predicted_value}")

        results.append({
            'Valor_Real': real_value,
            'Valor_Predito': predicted_value
        })

    # Retornar os resultados como um DataFrame
    return pd.DataFrame(results)

# Executar o teste
results_df = test_model()

# Exibir os resultados
if results_df is not None:
    print(results_df.head())  # Exibir os primeiros resultados
    print(results_df.tail())  # Exibir os últimos resultados

    # Salvar os resultados com base no nome do modelo
    save_results_to_csv(results_df)

