from src.utils.logger import Logger
import logging
from src.utils.csv_handler import CSVHandler
import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from keras.callbacks import EarlyStopping

Logger.initialize(log_file='logs/app.log', level=logging.INFO)

# Função para salvar os resultados das métricas em um arquivo TXT
def salvar_metrica_em_txt(batch_size, units, dropout, window_size, mse, mae, mape, r2, rmse, file_name='resultados_metrica.txt', descricao=''):
    with open(file_name, 'a') as f:
        f.write(f'{descricao}\nBatch Size: {batch_size}, Units: {units}, Dropout: {dropout}, Window Size: {window_size}\n')
        f.write(f'MSE: {mse}\n')
        f.write(f'MAE: {mae}\n')
        f.write(f'MAPE: {mape}\n')
        f.write(f'R2: {r2}\n')
        f.write(f'RMSE: {rmse}\n')
        f.write('-' * 40 + '\n')

# Função para salvar o gráfico em um arquivo PNG
# Função para salvar o gráfico em um arquivo PNG em um diretório específico
def salvar_grafico(y_real, y_pred, batch_size, units, dropout, window_size, nome_grafico='grafico_comparacao', descricao='', pasta='resultados'):
    # Cria a pasta se não existir
    if not os.path.exists(pasta):
        os.makedirs(pasta)

    # Caminho completo do arquivo
    caminho_arquivo = os.path.join(pasta, f'{nome_grafico}_{descricao}_batch_{batch_size}_units_{units}_dropout_{dropout}_window_{window_size}.png')
    
    plt.figure(figsize=(10, 6))
    plt.plot(y_real, color='blue', label='Preço Real')
    plt.plot(y_pred, color='red', label='Preço Previsto')
    plt.title(f'{descricao}\nComparação Preço Real vs Previsto\n(Batch Size: {batch_size}, Units: {units}, Dropout: {dropout}, Window Size: {window_size})')
    plt.xlabel('Período')
    plt.ylabel('Preço de Fechamento')
    plt.legend()
    
    # Salva o gráfico na pasta especificada
    plt.savefig(caminho_arquivo)
    plt.close()

# Função para salvar o modelo treinado em um diretório específico
def salvar_modelo(model, batch_size, units, dropout, window_size, pasta='modelos', descricao=''):
    # Cria a pasta se não existir
    if not os.path.exists(pasta):
        os.makedirs(pasta)
    
    # Caminho completo para salvar o modelo
    caminho_modelo = os.path.join(pasta, f'modelo_{descricao}_batch_{batch_size}_units_{units}_dropout_{dropout}_window_{window_size}')
    
    # Salvar o modelo no formato .h5
    model.save(f'{caminho_modelo}.h5')
    print(f'Modelo salvo em: {caminho_modelo}.h5')


# Função para rodar o treinamento com diferentes parâmetros e fazer o segundo teste com dados de 2024
def rodar_treinamento_com_parametros(parametros, precos_fechamento_scaled, scaler, novos_dados_2024, epochs=50):
    for batch_size, units, dropout, window_size in parametros:
        print(f"Treinando com Batch Size: {batch_size}, Units: {units}, Dropout: {dropout}, Window Size: {window_size}")

        # Criar as janelas de `window_size` períodos para previsão
        X, y = [], []
        for i in range(window_size, len(precos_fechamento_scaled)):
            X.append(precos_fechamento_scaled[i-window_size:i, 0])  # Últimos períodos
            y.append(precos_fechamento_scaled[i, 0])  # Próximo valor de fechamento

        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # (samples, timesteps, features)

        # Dividir em treino (80%) e teste (20%)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Construir o modelo LSTM
        model = Sequential()
        model.add(LSTM(units=units, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(dropout))
        model.add(LSTM(units=units, return_sequences=False))
        model.add(Dropout(dropout))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Adicionar Early Stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Treinar o modelo com os dados de treino e validar nos dados de teste
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[early_stopping])

        # Salvar o modelo treinado
        salvar_modelo(model, batch_size, units, dropout, window_size, pasta='modelos', descricao='modelo_treinado')

        # Fazer previsões nos dados de teste (primeiro teste)
        y_pred_test = model.predict(X_test)

        # Inverter a normalização para obter os preços reais
        y_pred_test_inverted = scaler.inverse_transform(y_pred_test)
        y_real_test_inverted = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Calcular métricas de avaliação para o conjunto de teste
        mse_test = mean_squared_error(y_real_test_inverted, y_pred_test_inverted)
        mae_test = mean_absolute_error(y_real_test_inverted, y_pred_test_inverted)
        mape_test = mean_absolute_percentage_error(y_real_test_inverted, y_pred_test_inverted)
        r2_test = r2_score(y_real_test_inverted, y_pred_test_inverted)
        rmse_test = np.sqrt(mse_test)

        # Salvar métricas e gráfico para o primeiro teste
        salvar_metrica_em_txt(batch_size, units, dropout, window_size, mse_test, mae_test, mape_test, r2_test, rmse_test, descricao='Teste (Historico)')
        salvar_grafico(y_real_test_inverted, y_pred_test_inverted, batch_size, units, dropout, window_size, descricao='teste_treino',pasta='result/teste_treino')

        # ------------------------ Segundo Teste com Dados de 2024 ------------------------

        # Normalizar os novos dados de 2024 usando o mesmo scaler
        novos_precos_fechamento_scaled = scaler.transform(novos_dados_2024.reshape(-1, 1))

        # Criar as janelas de `window_size` períodos para previsão nos novos dados
        X_novo, y_real_novo = [], []
        for i in range(window_size, len(novos_precos_fechamento_scaled)):
            X_novo.append(novos_precos_fechamento_scaled[i-window_size:i, 0])
            y_real_novo.append(novos_precos_fechamento_scaled[i, 0])

        X_novo = np.array(X_novo)
        y_real_novo = np.array(y_real_novo)
        X_novo = np.reshape(X_novo, (X_novo.shape[0], X_novo.shape[1], 1))

        # Fazer previsões nos novos dados (2024)
        y_pred_novo_scaled = model.predict(X_novo)

        # Inverter a normalização para obter os preços reais
        y_pred_novo = scaler.inverse_transform(y_pred_novo_scaled)
        y_real_novo_invertido = scaler.inverse_transform(np.array(y_real_novo).reshape(-1, 1))

        # Calcular métricas de avaliação para os novos dados
        mse_novo = mean_squared_error(y_real_novo_invertido, y_pred_novo)
        mae_novo = mean_absolute_error(y_real_novo_invertido, y_pred_novo)
        mape_novo = mean_absolute_percentage_error(y_real_novo_invertido, y_pred_novo)
        r2_novo = r2_score(y_real_novo_invertido, y_pred_novo)
        rmse_novo = np.sqrt(mse_novo)

        # Salvar métricas e gráfico para o segundo teste
        salvar_metrica_em_txt(batch_size, units, dropout, window_size, mse_novo, mae_novo, mape_novo, r2_novo, rmse_novo, descricao='Teste Bitcoin (2024)')
        salvar_grafico(y_real_novo_invertido, y_pred_novo, batch_size, units, dropout, window_size, descricao='teste_bitcoin_2024',pasta='result/teste_bitcoin')

# Carregar os dados históricos e calcular indicadores
data = [
    'output/binance/bitcoin/hourly/BTCUSDT_hourly_2021_2021.csv',
    'output/binance/bitcoin/hourly/BTCUSDT_hourly_2022_2022.csv',
    'output/binance/bitcoin/hourly/BTCUSDT_hourly_2023_2023.csv'
]
df = CSVHandler.read_multiple_csvs(data)

# Usar a coluna 'close' para previsões
precos_fechamento = df['close'].values

# Normalizar os preços (entre 0 e 1)
scaler = MinMaxScaler(feature_range=(0, 1))
precos_fechamento_scaled = scaler.fit_transform(precos_fechamento.reshape(-1, 1))

# Carregar os novos dados de 2024
novos_dados_2024 = pd.read_csv('output/binance/bitcoin/hourly/BTCUSDT_hourly_2024_2024.csv')['close'].values

# Definir os parâmetros para testar (Batch Size, Units, Dropout, Window Size)
parametros = [ 
    (16, 80, 0.5, 100), 
    (16, 80, 0.5, 200), 
    (16, 80, 0.5, 300), 
    (16, 80, 0.5, 400), 
    (16, 80, 0.5, 500), 
    (16, 90, 0.2, 100), 
    (16, 90, 0.2, 200), 
    (16, 90, 0.2, 300), 
    (16, 90, 0.2, 400), 
    (16, 90, 0.2, 500), 
    (16, 90, 0.3, 100), 
    (16, 90, 0.3, 200), 
    (16, 90, 0.3, 300), 
    (16, 90, 0.3, 400), 
    (16, 90, 0.3, 500), 
    (16, 90, 0.4, 100), 
    (16, 90, 0.4, 200), 
    (16, 90, 0.4, 300), 
    (16, 90, 0.4, 400), 
    (16, 90, 0.4, 500), 
    (16, 90, 0.5, 100), 
    (16, 90, 0.5, 200), 
    (16, 90, 0.5, 300), 
    (16, 90, 0.5, 400), 
    (16, 90, 0.5, 500), 
    (16, 100, 0.2, 100),
    (16, 100, 0.2, 200), 
    (16, 100, 0.2, 300), 
    (16, 100, 0.2, 400), 
    (16, 100, 0.2, 500), 
    (16, 100, 0.3, 100), 
    (16, 100, 0.3, 200), 
    (16, 100, 0.3, 300), 
    (16, 100, 0.3, 400), 
    (16, 100, 0.3, 500), 
    (16, 100, 0.4, 100), 
    (16, 100, 0.4, 200), 
    (16, 100, 0.4, 300), 
    (16, 100, 0.4, 400), 
    (16, 100, 0.4, 500), 
    (16, 100, 0.5, 100), 
    (16, 100, 0.5, 200), 
    (16, 100, 0.5, 300), 
    (16, 100, 0.5, 400), 
    (16, 100, 0.5, 500), 
    (32, 50, 0.2, 100), 
    (32, 50, 0.2, 200), 
    (32, 50, 0.2, 300), 
    (32, 50, 0.2, 400), 
    (32, 50, 0.2, 500), 
    (32, 50, 0.3, 100), 
    (32, 50, 0.3, 200), 
    (32, 50, 0.3, 300), 
    (32, 50, 0.3, 400), 
    (32, 50, 0.3, 500), 
    (32, 50, 0.4, 100), 
    (32, 50, 0.4, 200), 
    (32, 50, 0.4, 300), 
    (32, 50, 0.4, 400), 
    (32, 50, 0.4, 500), 
    (32, 50, 0.5, 100), 
    (32, 50, 0.5, 200), 
    (32, 50, 0.5, 300), 
    (32, 50, 0.5, 400), 
    (32, 50, 0.5, 500), 
    (32, 60, 0.2, 100), 
    (32, 60, 0.2, 200), 
    (32, 60, 0.2, 300), 
    (32, 60, 0.2, 400), 
    (32, 60, 0.2, 500), 
    (32, 60, 0.3, 100), 
    (32, 60, 0.3, 200), 
    (32, 60, 0.3, 300), 
    (32, 60, 0.3, 400), 
    (32, 60, 0.3, 500), 
    (32, 60, 0.4, 100), 
    (32, 60, 0.4, 200), 
    (32, 60, 0.4, 300), 
    (32, 60, 0.4, 400), 
    (32, 60, 0.4, 500), 
    (32, 60, 0.5, 100), 
    (32, 60, 0.5, 200), 
    (32, 60, 0.5, 300), 
    (32, 60, 0.5, 400), 
    (32, 60, 0.5, 500), 
    (32, 70, 0.2, 100), 
    (32, 70, 0.2, 200), 
    (32, 70, 0.2, 300), 
    (32, 70, 0.2, 400), 
    (32, 70, 0.2, 500), 
    (32, 70, 0.3, 100), 
    (32, 70, 0.3, 200), 
    (32, 70, 0.3, 400), 
    (32, 70, 0.3, 500), 
    (32, 70, 0.4, 100), 
    (32, 70, 0.4, 200), 
    (32, 70, 0.4, 300), 
    (32, 70, 0.4, 400), 
    (32, 70, 0.4, 500), 
    (32, 70, 0.5, 100), 
    (32, 70, 0.5, 200), 
    (32, 70, 0.5, 400), 
    (32, 70, 0.5, 500), 
    (32, 80, 0.2, 100), 
    (32, 80, 0.2, 200), 
    (32, 80, 0.2, 300), 
    (32, 80, 0.2, 400), 
    (32, 80, 0.2, 500), 
    (32, 80, 0.3, 100), 
    (32, 80, 0.3, 200), 
    (32, 80, 0.3, 300), 
    (32, 80, 0.3, 400), 
    (32, 80, 0.3, 500), 
    (32, 80, 0.4, 100), 
    (32, 80, 0.4, 200), 
    (32, 80, 0.4, 300), 
    (32, 80, 0.4, 400), 
    (32, 80, 0.4, 500), 
    (32, 80, 0.5, 100), 
    (32, 80, 0.5, 200), 
    (32, 80, 0.5, 300), 
    (32, 80, 0.5, 400), 
    (32, 80, 0.5, 500), 
    (32, 90, 0.2, 100), 
    (32, 90, 0.2, 200), 
    (32, 90, 0.2, 300), 
    (32, 90, 0.2, 400), 
    (32, 90, 0.2, 500), 
    (32, 90, 0.3, 100), 
    (32, 90, 0.3, 200), 
    (32, 90, 0.3, 300), 
    (32, 90, 0.3, 400), 
    (32, 90, 0.3, 500), 
    (32, 90, 0.4, 100), 
    (32, 90, 0.4, 200), 
    (32, 90, 0.4, 300), 
    (32, 90, 0.4, 400), 
    (32, 90, 0.4, 500), 
    (32, 90, 0.5, 100), 
    (32, 90, 0.5, 200), 
    (32, 90, 0.5, 300), 
    (32, 90, 0.5, 400), 
    (32, 90, 0.5, 500), 
    (32, 100, 0.2, 100),
    (32, 100, 0.2, 200),
    (32, 100, 0.2, 300), 
    (32, 100, 0.2, 400), 
    (32, 100, 0.2, 500), 
    (32, 100, 0.3, 100), 
    (32, 100, 0.3, 200), 
    (32, 100, 0.3, 300), 
    (32, 100, 0.3, 400), 
    (32, 100, 0.3, 500), 
    (32, 100, 0.4, 100), 
    (32, 100, 0.4, 200), 
    (32, 100, 0.4, 300), 
    (32, 100, 0.4, 400), 
    (32, 100, 0.4, 500), 
    (32, 100, 0.5, 100), 
    (32, 100, 0.5, 200), 
    (32, 100, 0.5, 300), 
    (32, 100, 0.5, 400), 
    (32, 100, 0.5, 500), 
    (64, 50, 0.2, 100), 
    (64, 50, 0.2, 200), 
    (64, 50, 0.2, 300), 
    (64, 50, 0.2, 400), 
    (64, 50, 0.2, 500), 
    (64, 50, 0.3, 100), 
    (64, 50, 0.3, 200), 
    (64, 50, 0.3, 300), 
    (64, 50, 0.3, 400), 
    (64, 50, 0.3, 500), 
    (64, 50, 0.4, 100), 
    (64, 50, 0.4, 200), 
    (64, 50, 0.4, 300), 
    (64, 50, 0.4, 400), 
    (64, 50, 0.4, 500), 
    (64, 50, 0.5, 100), 
    (64, 50, 0.5, 200), 
    (64, 50, 0.5, 300), 
    (64, 50, 0.5, 400), 
    (64, 50, 0.5, 500), 
    (64, 60, 0.2, 100), 
    (64, 60, 0.2, 200), 
    (64, 60, 0.2, 300), 
    (64, 60, 0.2, 400), 
    (64, 60, 0.2, 500), 
    (64, 60, 0.3, 100), 
    (64, 60, 0.3, 200), 
    (64, 60, 0.3, 300), 
    (64, 60, 0.3, 400), 
    (64, 60, 0.3, 500), 
    (64, 60, 0.4, 100), 
    (64, 60, 0.4, 200), 
    (64, 60, 0.4, 300), 
    (64, 60, 0.4, 400), 
    (64, 60, 0.4, 500), 
    (64, 60, 0.5, 100), 
    (64, 60, 0.5, 200), 
    (64, 60, 0.5, 300), 
    (64, 60, 0.5, 400), 
    (64, 60, 0.5, 500), 
    (64, 70, 0.2, 100), 
    (64, 70, 0.2, 200), 
    (64, 70, 0.2, 300), 
    (64, 70, 0.2, 400), 
    (64, 70, 0.2, 500), 
    (64, 70, 0.3, 100), 
    (64, 70, 0.3, 200), 
    (64, 70, 0.3, 300), 
    (64, 70, 0.3, 400), 
    (64, 70, 0.3, 500), 
    (64, 70, 0.4, 100), 
    (64, 70, 0.4, 200), 
    (64, 70, 0.4, 400), 
    (64, 70, 0.4, 500), 
    (64, 70, 0.5, 100), 
    (64, 70, 0.5, 200), 
    (64, 70, 0.5, 300), 
    (64, 70, 0.5, 400), 
    (64, 70, 0.5, 500), 
    (64, 80, 0.2, 100), 
    (64, 80, 0.2, 200), 
    (64, 80, 0.2, 300), 
    (64, 80, 0.2, 400), 
    (64, 80, 0.2, 500), 
    (64, 80, 0.3, 100), 
    (64, 80, 0.3, 200), 
    (64, 80, 0.3, 300), 
    (64, 80, 0.3, 400), 
    (64, 80, 0.3, 500), 
    (64, 80, 0.4, 100), 
    (64, 80, 0.4, 200), 
    (64, 80, 0.4, 300), 
    (64, 80, 0.4, 400), 
    (64, 80, 0.4, 500), 
    (64, 80, 0.5, 100), 
    (64, 80, 0.5, 200), 
    (64, 80, 0.5, 300), 
    (64, 80, 0.5, 400), 
    (64, 80, 0.5, 500), 
    (64, 90, 0.2, 100), 
    (64, 90, 0.2, 200), 
    (64, 90, 0.2, 300), 
    (64, 90, 0.2, 400), 
    (64, 90, 0.2, 500), 
    (64, 90, 0.3, 100), 
    (64, 90, 0.3, 200), 
    (64, 90, 0.3, 300), 
    (64, 90, 0.3, 400), 
    (64, 90, 0.3, 500), 
    (64, 90, 0.4, 100), 
    (64, 90, 0.4, 200), 
    (64, 90, 0.4, 300), 
    (64, 90, 0.4, 400), 
    (64, 90, 0.4, 500), 
    (64, 90, 0.5, 100), 
    (64, 90, 0.5, 200), 
    (64, 90, 0.5, 300), 
    (64, 90, 0.5, 400), 
    (64, 90, 0.5, 500),
    (64, 100, 0.2, 100),
    (64, 100, 0.2, 200), 
    (64, 100, 0.2, 300), 
    (64, 100, 0.2, 400), 
    (64, 100, 0.2, 500), 
    (64, 100, 0.3, 100), 
    (64, 100, 0.3, 200), 
    (64, 100, 0.3, 300), 
    (64, 100, 0.3, 400), 
    (64, 100, 0.3, 500), 
    (64, 100, 0.4, 100), 
    (64, 100, 0.4, 200), 
    (64, 100, 0.4, 300), 
    (64, 100, 0.4, 400), 
    (64, 100, 0.4, 500), 
    (64, 100, 0.5, 100), 
    (64, 100, 0.5, 200), 
    (64, 100, 0.5, 300), 
    (64, 100, 0.5, 400), 
    (64, 100, 0.5, 500)
              ]

# Rodar o treinamento e avaliação com diferentes parâmetros
rodar_treinamento_com_parametros(parametros, precos_fechamento_scaled, scaler, novos_dados_2024)