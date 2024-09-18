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
import joblib  # Para salvar o scaler
from src.utils.technical_indicators import TechnicalIndicators

# Fixar a semente para garantir reprodutibilidade
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

def adicionar_indicadores(df):
    """
    Adiciona indicadores técnicos ao DataFrame.
    
    :param df: DataFrame contendo os dados de preços.
    :return: DataFrame com indicadores técnicos adicionados.
    """
    # Adicionar indicadores ao DataFrame
    df['SMA_20'] = TechnicalIndicators.sma(df['close'], period=20)  # Média Móvel Simples
    df['EMA_20'] = TechnicalIndicators.ema(df['close'], period=20)  # Média Móvel Exponencial
    df['RSI_14'] = TechnicalIndicators.rsi(df['close'], period=14)  # Índice de Força Relativa
    
    macd_df = TechnicalIndicators.macd(df['close'])  # MACD
    df = pd.concat([df, macd_df], axis=1)  # Adicionando MACD, Signal e Histograma ao DataFrame

    envelopes_df = TechnicalIndicators.envelopes(df['close'], period=20, percent=3.0)  # Envelopes de Preço
    df = pd.concat([df, envelopes_df], axis=1)  # Adicionando os Envelopes ao DataFrame

    return df

def calcular_e_exibir_correlacao(df):
    """
    Calcula e exibe a correlação entre indicadores técnicos e o preço de fechamento.
    
    :param df: DataFrame com indicadores técnicos e preços.
    """

    # Selecionar apenas colunas numéricas
    df_numerico = df.select_dtypes(include=[np.number])
    # Calcular a correlação com a coluna 'close'
    if 'close' in df_numerico.columns:
        correlacoes = df_numerico.corr()['close'].drop('close')
        print(correlacoes)
    else:
        print("A coluna 'close' não está presente no DataFrame ou não é numérica.")

    # Exibir correlações por escrito
    print("\nCorrelação entre indicadores técnicos e o preço de fechamento:")
    print(correlacoes)

    # Plotar correlação
    plt.figure(figsize=(10, 6))
    correlacoes.plot(kind='bar', color='skyblue')
    plt.title('Correlação entre Indicadores Técnicos e Preço de Fechamento')
    plt.xlabel('Indicadores Técnicos')
    plt.ylabel('Correlação')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Salvar o gráfico como imagem
    plt.savefig('correlacao_indicadores.png')
    print("Gráfico de correlação salvo em: correlacao_indicadores.png")
    plt.show()

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

def listar_modelos(pasta_modelos='modelos'):
    """
    Lista todos os arquivos de modelos salvos na pasta especificada.
    
    :param pasta_modelos: Caminho da pasta onde os modelos estão salvos.
    :return: Lista de caminhos completos dos modelos disponíveis.
    """
    # Verifica se a pasta existe
    if not os.path.exists(pasta_modelos):
        print(f"A pasta '{pasta_modelos}' não foi encontrada.")
        return []
    
    # Lista todos os arquivos da pasta de modelos
    arquivos_modelos = [f for f in os.listdir(pasta_modelos) if os.path.isfile(os.path.join(pasta_modelos, f))]
    
    # Verifica se há arquivos de modelos disponíveis
    if len(arquivos_modelos) == 0:
        print("Nenhum modelo foi encontrado na pasta.")
        return []
    
    # Retorna a lista de arquivos de modelos
    return arquivos_modelos

# Função para salvar o modelo treinado em um diretório específico
def salvar_modelo(model, batch_size, units, dropout, window_size, pasta='modelos', descricao=''):
    if not os.path.exists(pasta):
        os.makedirs(pasta)
    
    # Se for um modelo Keras (como LSTM)
    if hasattr(model, 'save'):
        caminho_modelo = os.path.join(pasta, f'modelo_{descricao}_batch_{batch_size}_units_{units}_dropout_{dropout}_window_{window_size}.h5')
        model.save(caminho_modelo)
        print(f'Modelo Keras salvo em: {caminho_modelo}')
    else:
        # Se for um modelo scikit-learn (como LinearRegression)
        caminho_modelo = os.path.join(pasta, f'modelo_{descricao}_batch_{batch_size}_units_{units}_dropout_{dropout}_window_{window_size}.pkl')
        joblib.dump(model, caminho_modelo)
        print(f'Modelo scikit-learn salvo em: {caminho_modelo}')

# Função para criar janelas de dados (idêntica ao primeiro código)
def criar_janelas(dados, window_size, coluna_alvo=None):
    """
    Cria janelas de dados a partir de um DataFrame ou uma única coluna.
    
    :param dados: Pode ser uma coluna única (1D array) ou um DataFrame (2D array).
    :param window_size: Tamanho da janela para o LSTM.
    :param coluna_alvo: Coluna a ser utilizada como valor de saída (y). Se for None, assume que é uma única coluna ou a última coluna.
    """
    X, y = [], []

    # Se 'dados' for um DataFrame (múltiplas colunas)
    if len(dados.shape) == 2:
        # Se a coluna alvo for fornecida, usar essa coluna. Caso contrário, usar a última coluna como padrão
        if coluna_alvo is None:
            coluna_alvo = dados.shape[1] - 1  # Última coluna se não especificar
        for i in range(window_size, len(dados)):
            # Seleciona todas as colunas nas janelas
            X.append(dados[i-window_size:i])  # Últimos 'window_size' períodos para todas as features
            y.append(dados[i, coluna_alvo])  # Valor da coluna alvo no próximo período

    # Se 'dados' for uma única coluna (1D array)
    elif len(dados.shape) == 1:
        for i in range(window_size, len(dados)):
            # Seleciona os últimos 'window_size' períodos
            X.append(dados[i-window_size:i])
            y.append(dados[i])  # Próximo valor da mesma coluna

    # Convertendo para numpy arrays
    X, y = np.array(X), np.array(y)

    # Se o dado de entrada for um DataFrame ou array 2D, precisamos manter o formato 3D
    if len(X.shape) == 2:  # Caso seja 2D, adicionar uma dimensão extra para ser compatível com LSTM
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # (samples, timesteps, features)
    
    return X, y

# Função para treinar e avaliar o LSTM (idêntico ao primeiro código)
def treinar_lstm(X_train, y_train, X_test, y_test, X_validation, y_validation, scaler, batch_size, units, dropout, epochs):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(dropout))
    model.add(LSTM(units=units, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_validation, y_validation), callbacks=[early_stopping])

    # Salvar o modelo LSTM usando a função salvar_modelo
    salvar_modelo(model, batch_size, units, dropout, window_size, pasta='teste/modelos', descricao='lstm')

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

    # Salvar o modelo de Regressão Linear usando a função salvar_modelo
    salvar_modelo(reg, batch_size=16, units=0, dropout=0, window_size=0, pasta='teste/modelos', descricao='regressao_linear')


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


def carregar_e_executar_modelo_da_pasta(X_test, y_test, scaler, pasta_modelos='modelos'):
    """
    Lista e permite ao usuário selecionar um modelo salvo na pasta especificada.
    O modelo selecionado será carregado e executará as previsões nos dados de teste.
    
    :param X_test: Dados de entrada para teste.
    :param y_test: Labels verdadeiros para teste.
    :param scaler: Scaler usado para normalizar/desnormalizar os dados.
    :param pasta_modelos: Caminho da pasta onde os modelos estão salvos.
    """
    # Listar os modelos disponíveis na pasta
    modelos_disponiveis = listar_modelos(pasta_modelos)

    if not modelos_disponiveis:
        print("Não há modelos disponíveis para carregar.")
        return
    
    print("\nModelos disponíveis:")
    for idx, modelo in enumerate(modelos_disponiveis):
        print(f"{idx + 1} - {modelo}")
    
    # Permitir ao usuário escolher um modelo
    escolha = int(input("Digite o número do modelo que deseja carregar: ")) - 1

    if escolha < 0 or escolha >= len(modelos_disponiveis):
        print("Escolha inválida. Tente novamente.")
        return
    
    modelo_escolhido = modelos_disponiveis[escolha]
    caminho_modelo = os.path.join(pasta_modelos, modelo_escolhido)
    
    # Carregar o modelo com base na extensão do arquivo
    if caminho_modelo.endswith('.h5'):
        print(f"Carregando o modelo LSTM: {modelo_escolhido}")
        model = load_model(caminho_modelo)
        y_pred = model.predict(X_test)
    elif caminho_modelo.endswith('.pkl'):
        print(f"Carregando o modelo de Regressão Linear: {modelo_escolhido}")
        model = joblib.load(caminho_modelo)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)  # Achatar para regressão linear
        y_pred = model.predict(X_test_flat)
    else:
        print("Tipo de modelo não suportado.")
        return
    
    # Inverter a normalização para os valores preditos e reais
    y_pred_inverted = scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_real_test_inverted = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    mse = mean_squared_error(y_real_test_inverted, y_pred_inverted)
    mae = mean_absolute_error(y_real_test_inverted, y_pred_inverted)
    mape = mean_absolute_percentage_error(y_real_test_inverted, y_pred_inverted)
    r2 = r2_score(y_real_test_inverted, y_pred_inverted)
    rmse = np.sqrt(mse)

    print(f'Modelo {modelo_escolhido.upper()} - MSE: {mse}, MAE: {mae}, MAPE: {mape}, R2: {r2}, RMSE: {rmse}')
    
    # Gerar gráfico comparativo
    plot_comparacao(y_real_test_inverted, y_pred_inverted, y_pred_inverted, save_path=f'comparacao_{modelo_escolhido}.png')
    print(f'Gráfico de comparação para o modelo {modelo_escolhido.upper()} gerado.')

# Função principal para treinar o modelo LSTM, Regressão Linear e Híbrido
def rodar_modelo(X_train, X_test, y_train, y_test,X_validation, y_validation, scaler, batch_size, units, dropout, epochs):
    # Treinar LSTM
    y_pred_lstm, mse_lstm, mae_lstm, mape_lstm, r2_lstm, rmse_lstm = treinar_lstm(X_train, y_train, X_test, y_test,X_validation, y_validation, scaler, batch_size, units, dropout, epochs)

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

#verificar a porcentagem do mim max
#ajustar mim max apenas com conjunto de treino e utilizar o mesmo para teste
#quebrar os 3 conjuntos teste treino e validação
# Criar os dados para o LSTM (janelas)
X, y = criar_janelas(precos_fechamento, window_size)

# Definir os tamanhos das divisões
train_size = int(len(X) * 0.7)
validation_size = int(len(X) * 0.15)
test_size = len(X) - train_size - validation_size

# Dividir os dados
X_train, X_validation, X_test = X[:train_size], X[train_size:train_size + validation_size], X[train_size + validation_size:]
y_train, y_validation, y_test = y[:train_size], y[train_size:train_size + validation_size], y[train_size + validation_size:]

# Definir o scaler para normalizar no intervalo de 0 a 1
scaler = MinMaxScaler(feature_range=(0, 1))

# Normalizar apenas os dados de treino
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[2]))
y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1))

# Aplicar a mesma normalização nos dados de validação e teste
#X_validation_scaled = scaler.transform(X_validation.reshape(-1, 1))
#X_test_scaled = scaler.transform(X_test.reshape(-1, 1))
# Salvar o scaler para uso posterior
joblib.dump(scaler, 'scaler.pkl')

# Exemplo de uso posterior do scaler ---------------------------------------------
# Carregar o scaler salvo
#scaler = joblib.load('scaler.pkl')
# Suponha que `novos_dados` seja o conjunto de dados novos que você quer normalizar
#novos_dados_scaled = scaler.transform(novos_dados.reshape(-1, 1))
#-------------------------------------------------------------------------
scaler = joblib.load('scaler.pkl')
# Aplicar a mesma normalização nos dados de validação e teste
X_validation_scaled = scaler.transform(X_validation.reshape(-1, X_validation.shape[2]))
X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[2]))

# Redimensionando de volta para o formato 3D após normalização
X_train_scaled = X_train_scaled.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
X_validation_scaled = X_validation_scaled.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2])
X_test_scaled = X_test_scaled.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

# Normalizar os targets (y) de validação e teste
y_validation_scaled = scaler.transform(y_validation.reshape(-1, 1))
y_test_scaled = scaler.transform(y_test.reshape(-1, 1))

def menu():
    while True:
        print("\nEscolha uma opção:")
        print("1 - Treinar Regressão Linear")
        print("2 - Treinar LSTM")
        print("3 - Treinar ambos (LSTM + Regressão Linear) e Hibridizar")
        print("4 - Gerar Matriz de Autocorrelação")
        print("5 - Carregar e Executar um Modelo Salvo")
        print("6 - Verificar Correlação entre Indicadores e Preço de Fechamento")
        print("7 - Sair")

        escolha = input("Digite o número da opção desejada: ")

        if escolha == '1':
            print("Treinando Regressão Linear...")
            y_pred_linear, mse_linear, mae_linear, mape_linear, r2_linear, rmse_linear = treinar_regressao_linear(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, scaler)
        
        elif escolha == '2':
            print("Treinando LSTM...")
            y_pred_lstm, mse_lstm, mae_lstm, mape_lstm, r2_lstm, rmse_lstm = treinar_lstm(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, X_validation_scaled, y_validation_scaled, scaler, batch_size, units, dropout, epochs)

        elif escolha == '3':
            print("Treinando ambos (LSTM + Regressão Linear) e Hibridizando...")
            resultados = rodar_modelo(X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, X_validation_scaled, y_validation_scaled, scaler, batch_size, units, dropout, epochs)
            print(resultados)

        elif escolha == '4':
            print("Gerando Matriz de Autocorrelação...")
            lista_window_sizes = [50, 100, 150, 200, 250, 300]
            autocorrelacoes = calcular_autocorrelacao(precos_fechamento, lista_window_sizes)
            plotar_autocorrelacao(autocorrelacoes)

        elif escolha == '5':
            print("Carregar e Executar um Modelo Salvo da Pasta")
            carregar_e_executar_modelo_da_pasta(X_test_scaled, y_test_scaled, scaler, 'teste/modelos')

        elif escolha == '6':
            print("Verificando Correlação entre Indicadores e Preço de Fechamento...")
            df_com_indicadores = adicionar_indicadores(df)
            calcular_e_exibir_correlacao(df_com_indicadores)

        elif escolha == '7':
            print("Saindo...")
            break
        else:
            print("Opção inválida, por favor tente novamente.")

menu()
