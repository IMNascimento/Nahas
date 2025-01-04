import os
import sys
src_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(src_root) if src_root not in sys.path else None
from models.db.model_binance import HourlyQuote
from models.lstm_model import CustomLSTMTrainer
from data.data_processing import DataProcessor
from utils.technical_indicators import TechnicalIndicators
from config.settings import Settings
import pandas as pd


def save_results_to_csv(results_df):
    """
    Salva os resultados em um arquivo CSV com o nome baseado no modelo.
    """
    model_name = os.path.basename(Settings.LOAD_MODEL).replace(".h5", "")  # Extrai o nome do modelo sem extensão
    output_path = f"result/testes/{model_name}_results.csv"  # Define o caminho de saída
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
    trainer = CustomLSTMTrainer()
    model = trainer.loading_model(Settings.LOAD_MODEL)
    processor = DataProcessor(window_size=Settings.WINDOW_SIZE)

    # Carregar os dados do banco
    data = HourlyQuote.get_from_date(Settings.START_DATE)
    if data is None or len(data) < Settings.WINDOW_SIZE:
        print("Dados insuficientes para realizar a previsão.")
        return
    data = TechnicalIndicators.process_indicators(data, Settings.INDICATORS_APPLY)
    
    # Converter timestamps para numérico (se necessário)
    if 'timestamp' in data.columns:
        data['timestamp'] = data['timestamp'].apply(lambda x: x.timestamp())


    # Normalizar os dados com janela deslizante
    features = data[Settings.RELEVANT_COLUMNS].values
    features_normalized = processor.normalize_sliding_window(features)
    # Preparar o array para armazenar os resultados
    results = []
    for i in range(len(features_normalized) - Settings.WINDOW_SIZE):
        # Criar uma janela deslizante
        window_normalized = features_normalized[i:i + Settings.WINDOW_SIZE]
        real_value = data.iloc[i + Settings.WINDOW_SIZE]['close']

        # Preparar os dados para entrada no modelo
        X_input = window_normalized.reshape(1, window_normalized.shape[0], window_normalized.shape[1])

        # Fazer a previsão
        predicted_scaled = model.predict(X_input)

        # Reverter a normalização usando estatísticas da janela
        mean = features[i:i + Settings.WINDOW_SIZE].mean(axis=0)
        std = features[i:i + Settings.WINDOW_SIZE].std(axis=0) + 1e-8
        predicted_value = (predicted_scaled * std[-1]) + mean[-1]

        # Adicionar aos resultados
        results.append({
            'Valor_Real': real_value,
            'Valor_Predito': predicted_value[0][0]
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

