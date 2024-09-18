import tensorflow as tf
import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from keras.callbacks import EarlyStopping
import os
from src.utils.csv_handler import CSVHandler
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot

# Fixar a semente para garantir reprodutibilidade
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

def plot_comparacao_regressao_linear(y_real, y_pred_linear, save_path='comparacao_regressao_linear.png'):
    """
    Gera um gráfico comparativo entre os valores reais e os preditos pela regressão linear,
    e salva o gráfico em um arquivo PNG.
    
    :param y_real: Valores reais do preço de fechamento.
    :param y_pred_linear: Valores preditos pela regressão linear.
    :param save_path: Caminho para salvar a imagem do gráfico.
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(y_real, color='blue', label='Preço Real')
    plt.plot(y_pred_linear, color='red', label='Previsão Regressão Linear')
    
    plt.title('Comparação entre Preço Real e Previsão - Regressão Linear')
    plt.xlabel('Período')
    plt.ylabel('Preço de Fechamento')
    plt.legend()
    
    # Salvar o gráfico como imagem
    plt.savefig(save_path)
    print(f'Gráfico salvo em: {save_path}')
    plt.close()

# Função para plotar o gráfico de autocorrelação
def plotar_autocorrelacao(autocorrelacoes):
    window_sizes = list(autocorrelacoes.keys())
    valores_autocorrelacao = list(autocorrelacoes.values())

    plt.figure(figsize=(10, 6))
    plt.plot(window_sizes, valores_autocorrelacao, marker='o', linestyle='-', color='b')
    plt.title('Autocorrelação para Diferentes Tamanhos de Janela')
    plt.xlabel('Tamanho da Janela')
    plt.ylabel('Autocorrelação')
    plt.grid(True)
    
    # Exibir o gráfico
    plt.show()

def calcular_autocorrelacao(precos_fechamento, lista_window_sizes):
    autocorrelacoes = {}

    for window_size in lista_window_sizes:
        print(f"\nCalculando autocorrelação para window_size = {window_size}...")
        
        # Seleciona os últimos 'window_size' valores
        dados_janela = precos_fechamento[-window_size:]

        # Calcular a autocorrelação
        autocorrelacao = pd.Series(dados_janela).autocorr(lag=1)
        autocorrelacoes[window_size] = autocorrelacao

        print(f"Autocorrelação para window_size {window_size}: {autocorrelacao}")

    return autocorrelacoes


# Função para salvar o modelo treinado em um diretório específico
def salvar_modelo(model, batch_size, units, dropout, window_size, pasta='modelos', descricao=''):
    if not os.path.exists(pasta):
        os.makedirs(pasta)
    
    caminho_modelo = os.path.join(pasta, f'modelo_{descricao}_batch_{batch_size}_units_{units}_dropout_{dropout}_window_{window_size}.h5')
    model.save(caminho_modelo)
    print(f'Modelo salvo em: {caminho_modelo}')

# Função para criar janelas de dados (idêntica ao primeiro código)
def criar_janelas(precos_fechamento_scaled, window_size):
    X, y = [], []
    for i in range(window_size, len(precos_fechamento_scaled)):
        X.append(precos_fechamento_scaled[i-window_size:i, 0])  # Últimos períodos
        y.append(precos_fechamento_scaled[i, 0])  # Próximo valor de fechamento
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # (samples, timesteps, features)
    return X, y

# Função para treinar e avaliar o LSTM (idêntico ao primeiro código)
def treinar_lstm(X_train, y_train, X_test, y_test, scaler, batch_size, units, dropout, epochs):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(dropout))
    model.add(LSTM(units=units, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[early_stopping])

    y_pred_lstm = model.predict(X_test)

    y_pred_lstm_inverted = scaler.inverse_transform(y_pred_lstm)
    y_real_test_inverted = scaler.inverse_transform(y_test.reshape(-1, 1))

    mse_lstm = mean_squared_error(y_real_test_inverted, y_pred_lstm_inverted)
    mae_lstm = mean_absolute_error(y_real_test_inverted, y_pred_lstm_inverted)
    mape_lstm = mean_absolute_percentage_error(y_real_test_inverted, y_pred_lstm_inverted)
    r2_lstm = r2_score(y_real_test_inverted, y_pred_lstm_inverted)
    rmse_lstm = np.sqrt(mse_lstm)

    print(f'LSTM - MSE: {mse_lstm}, MAE: {mae_lstm}, MAPE: {mape_lstm}, R2: {r2_lstm}, RMSE: {rmse_lstm}')

    return y_pred_lstm_inverted, mse_lstm, mae_lstm, mape_lstm, r2_lstm, rmse_lstm

# Função para treinar e avaliar a Regressão Linear
def treinar_regressao_linear(X_train, y_train, X_test, y_test, scaler):
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    reg = LinearRegression()
    reg.fit(X_train_flat, y_train)

    y_pred_linear = reg.predict(X_test_flat)

    y_pred_linear_inverted = scaler.inverse_transform(y_pred_linear.reshape(-1, 1))
    y_real_test_inverted = scaler.inverse_transform(y_test.reshape(-1, 1))

    mse_linear = mean_squared_error(y_real_test_inverted, y_pred_linear_inverted)
    mae_linear = mean_absolute_error(y_real_test_inverted, y_pred_linear_inverted)
    mape_linear = mean_absolute_percentage_error(y_real_test_inverted, y_pred_linear_inverted)
    r2_linear = r2_score(y_real_test_inverted, y_pred_linear_inverted)
    rmse_linear = np.sqrt(mse_linear)

    print(f'Regressão Linear - MSE: {mse_linear}, MAE: {mae_linear}, MAPE: {mape_linear}, R2: {r2_linear}, RMSE: {rmse_linear}')

    return y_pred_linear_inverted, mse_linear, mae_linear, mape_linear, r2_linear, rmse_linear

# Função para hibridizar os modelos
def hibridizar_modelos(y_pred_lstm, y_pred_linear):
    return 0.5 * y_pred_lstm + 0.5 * y_pred_linear

def plot_comparacao(y_real, y_pred_lstm, y_pred_hibrido, save_path='comparacao_lstm_hibrido.png'):
    plt.figure(figsize=(10, 6))
    
    plt.plot(y_real, color='blue', label='Preço Real')
    plt.plot(y_pred_lstm, color='green', label='Previsão LSTM')
    plt.plot(y_pred_hibrido, color='red', label='Previsão Híbrida')

    plt.title('Comparação entre Preço Real, Previsão LSTM e Previsão Híbrida')
    plt.xlabel('Período')
    plt.ylabel('Preço de Fechamento')
    plt.legend()
    
    # Salvar o gráfico como imagem
    plt.savefig(save_path)
    plt.close()

# Função principal para treinar o modelo LSTM, Regressão Linear e Híbrido
def rodar_modelo(X_train, X_test, y_train, y_test, scaler, batch_size, units, dropout, epochs):
    # Treinar LSTM
    y_pred_lstm, mse_lstm, mae_lstm, mape_lstm, r2_lstm, rmse_lstm = treinar_lstm(X_train, y_train, X_test, y_test, scaler, batch_size, units, dropout, epochs)

    # Treinar Regressão Linear
    y_pred_linear, mse_linear, mae_linear, mape_linear, r2_linear, rmse_linear = treinar_regressao_linear(X_train, y_train, X_test, y_test, scaler)

    # Hibridizar os modelos
    y_pred_hybrid = hibridizar_modelos(y_pred_lstm, y_pred_linear)

    # Calcular métricas para o modelo híbrido
    y_real_test_inverted = scaler.inverse_transform(y_test.reshape(-1, 1))
    mse_hybrid = mean_squared_error(y_real_test_inverted, y_pred_hybrid)
    mae_hybrid = mean_absolute_error(y_real_test_inverted, y_pred_hybrid)
    mape_hybrid = mean_absolute_percentage_error(y_real_test_inverted, y_pred_hybrid)
    r2_hybrid = r2_score(y_real_test_inverted, y_pred_hybrid)
    rmse_hybrid = np.sqrt(mse_hybrid)

    print(f'Híbrido - MSE: {mse_hybrid}, MAE: {mae_hybrid}, MAPE: {mape_hybrid}, R2: {r2_hybrid}, RMSE: {rmse_hybrid}')

    # Gerar gráfico comparativo para a Regressão Linear e salvar a imagem
    plot_comparacao_regressao_linear(y_real_test_inverted, y_pred_linear, save_path='comparacao_regressao_linear.png')

    # Gerar gráfico comparativo e salvar a imagem
    plot_comparacao(y_real_test_inverted, y_pred_lstm, y_pred_hybrid, save_path='comparacao_lstm_hibrido.png')

    return {
        'lstm': (mse_lstm, mae_lstm, mape_lstm, r2_lstm, rmse_lstm),
        'regressao_linear': (mse_linear, mae_linear, mape_linear, r2_linear, rmse_linear),
        'hibrido': (mse_hybrid, mae_hybrid, mape_hybrid, r2_hybrid, rmse_hybrid)
    }

# Parâmetros do modelo
batch_size = 16
units = 70
dropout = 0.3
epochs = 50
window_size = 300

# Carregar os dados
data = [
    'output/binance/bitcoin/hourly/BTCUSDT_hourly_2021_2021.csv',
    'output/binance/bitcoin/hourly/BTCUSDT_hourly_2022_2022.csv',
    'output/binance/bitcoin/hourly/BTCUSDT_hourly_2023_2023.csv'
]
df = CSVHandler.read_multiple_csvs(data)

precos_fechamento = df['close'].values

# Normalizar os preços (entre 0 e 1)
scaler = MinMaxScaler(feature_range=(0, 1))
precos_fechamento_scaled = scaler.fit_transform(precos_fechamento.reshape(-1, 1))

#passar antes da normalização
#verificar a porcentagem do mim max
#ajustar mim max apenas com conjunto de treino e utilizar o mesmo para teste
#quebrar os 3 conjuntos teste treino e validação
# Criar os dados para o LSTM (janelas)
X, y = criar_janelas(precos_fechamento_scaled, window_size)

# Dividir em treino e teste
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Rodar o modelo LSTM, Regressão Linear e Híbrido
resultados = rodar_modelo(X_train, X_test, y_train, y_test, scaler, batch_size, units, dropout, epochs)

# Imprimir os resultados
print(resultados)


"""
# Lista de diferentes tamanhos de janela a serem testados
lista_window_sizes = [50, 100, 150, 200, 250, 300]

# Carregar os dados (preços de fechamento)
precos_fechamento = df['close'].values

# Calcular a autocorrelação para cada tamanho de janela
autocorrelacoes = calcular_autocorrelacao(precos_fechamento, lista_window_sizes)

# Plotar o gráfico de autocorrelação
plotar_autocorrelacao(autocorrelacoes)

# Imprimir os resultados
for window_size, autocorrelacao in autocorrelacoes.items():
    print(f"Window Size: {window_size}, Autocorrelação: {autocorrelacao}")
"""