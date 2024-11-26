from src.utils.csv_handler import CSVHandler
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from collections import Counter
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout

# Funções para calcular indicadores técnicos
def SMA(data, window):
    return data['Close'].rolling(window=window).mean()

def WMA(data, window):
    weights = np.arange(1, window + 1)
    return data['Close'].rolling(window=window).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)

def EMA(data, window):
    return data['Close'].ewm(span=window, adjust=False).mean()

def PPO(data, fast_period=12, slow_period=26):
    fast_ema = EMA(data, fast_period)
    slow_ema = EMA(data, slow_period)
    return ((fast_ema - slow_ema) / slow_ema) * 100

def PAIN(data):
    return (data['Close'] - data['Open']) / (data['High'] - data['Low'])

def MACD(data, fast_period=12, slow_period=26, signal_period=9):
    fast_ema = EMA(data, fast_period)
    slow_ema = EMA(data, slow_period)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    return macd_line, signal_line

def RSI(data, window=14):
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def Momentum(data, window=1):
    return data['Close'] - data['Close'].shift(window)

def Stochastic_K(data, window=14):
    lowest_low = data['Low'].rolling(window=window).min()
    highest_high = data['High'].rolling(window=window).max()
    return 100 * (data['Close'] - lowest_low) / (highest_high - lowest_low)

def Stochastic_D(data, window=3):
    return Stochastic_K(data).rolling(window=window).mean()

# Transformar valores numéricos em tendências textuais
def transformar_dados_texto(df, coluna, limite=0.001):
    tendencias = []
    for i in range(1, len(df)):
        if (df[coluna].iloc[i] - df[coluna].iloc[i-1]) / df[coluna].iloc[i-1] > limite:
            tendencias.append('uptrend')
        elif (df[coluna].iloc[i] - df[coluna].iloc[i-1]) / df[coluna].iloc[i-1] < -limite:
            tendencias.append('downtrend')
        else:
            tendencias.append('no trend')
    tendencias.insert(0, 'no trend')  # Para o primeiro valor
    return tendencias

# Definir ação com base em tendências
def definir_acao(df):
    acoes = []
    for i in range(len(df)):
        uptrends = sum([1 for col in df.columns if df[col].iloc[i] == 'uptrend'])
        
        if uptrends >= 7:  # Se houver 7 ou mais tendências de alta
            acoes.append('Buy')
        elif uptrends <= 4:  # Se houver 4 ou menos tendências de alta
            acoes.append('Sell')
        else:
            acoes.append('Hold')
    return acoes

# Avaliar o modelo
def avaliar_modelo(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='weighted')
    kappa = cohen_kappa_score(y_test, y_pred)
    
    # Exibir resultados
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Kappa: {kappa}")

# Treinar e avaliar a árvore de decisão, gerando a imagem da árvore
def treinar_avaliar_arvore(X, y, feature_names, class_weights):
    # Dividir os dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Treinar a árvore de decisão com profundidade limitada
    for depth in range(1, 10):
        clf = DecisionTreeClassifier(criterion="entropy",class_weight=class_weights, max_depth=depth, min_samples_split=10, min_samples_leaf=5)
        scores = cross_val_score(clf, X_train, y_train, cv=5)  # 5-fold cross-validation
        print(f"Profundidade: {depth}, Acurácia média (cross-validação): {scores.mean()}")

    #clf = DecisionTreeClassifier(criterion="entropy",class_weight=class_weights, max_depth=5, min_samples_split=10, min_samples_leaf=5,max_features="sqrt")

    clf.fit(X_train, y_train)
    
    # Fazer previsões
    y_pred = clf.predict(X_test)
    
    # Avaliar o modelo
    avaliar_modelo(y_test, y_pred)
    
    # Gerar e salvar a imagem da árvore
    plt.figure(figsize=(20,10))
    plot_tree(clf, feature_names=feature_names, class_names=['Buy', 'Sell', 'Hold'], filled=True, fontsize=10)
    plt.savefig('arvore_decisao.png')
    plt.show()

def treinar_avaliar_arvore_time_series(X, y, feature_names, class_weights):
    # Definir o TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)  # 5 divisões temporais
    
    # Loop sobre cada split
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Treinar a árvore de decisão
        clf = DecisionTreeClassifier(criterion="entropy", class_weight=class_weights, max_depth=5, min_samples_split=10, min_samples_leaf=5)
        clf.fit(X_train, y_train)
        
        # Fazer previsões
        y_pred = clf.predict(X_test)
        
        # Avaliar o modelo
        avaliar_modelo(y_test, y_pred)
        
        # Gerar e salvar a imagem da árvore
        plt.figure(figsize=(20,10))
        plot_tree(clf, feature_names=feature_names, class_names=['Buy', 'Sell', 'Hold'], filled=True, fontsize=10)
        plt.savefig('arvore_decisao_split.png')
        plt.show()

# Função para treinar e avaliar a árvore de decisão, gerando probabilidades
def treinar_avaliar_arvore_e_probabilidades(X, y, feature_names, class_weights):
    tscv = TimeSeriesSplit(n_splits=5)
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index].copy()  # Usar .copy() para evitar "SettingWithCopyWarning"
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        clf = DecisionTreeClassifier(criterion="entropy", class_weight=class_weights, max_depth=5, min_samples_split=10, min_samples_leaf=5)
        clf.fit(X_train, y_train)
        
        probas = clf.predict_proba(X_test)
        
        # Adicionar probabilidades ao DataFrame de teste
        X_test.loc[:, 'Prob_Buy'] = probas[:, 0]   # Usar .loc com .copy para evitar warnings
        X_test.loc[:, 'Prob_Sell'] = probas[:, 1]
        X_test.loc[:, 'Prob_Hold'] = probas[:, 2]
        
        return X_test[['Prob_Buy', 'Prob_Sell', 'Prob_Hold']]

# Criar o modelo LSTM ajustado para múltiplas features
def criar_modelo_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(units=70, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.4))
    model.add(LSTM(units=70, return_sequences=False))
    model.add(Dropout(0.4))
    model.add(Dense(units=1))  # Saída para uma única previsão (preço futuro)
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Função para treinar e avaliar o modelo LSTM
def avaliar_modelo_lstm(X, y, modelo):
    previsoes = modelo.predict(X)
    mse = np.mean(np.square(previsoes - y))
    print(f'MSE do LSTM: {mse}')
    return mse

# Função para preparar janelas de entrada e saída para o LSTM
def preparar_janelas_lstm(data, window_size=300):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, :])  # Últimos 'window_size' períodos
        y.append(data[i, 0])  # Próximo valor de fechamento (ou outro target)
    return np.array(X), np.array(y)

# Função para normalizar os dados
def normalizar_dados(df, colunas=None):
    scaler = MinMaxScaler()
    if colunas:
        df[colunas] = scaler.fit_transform(df[colunas])
    else:
        df = scaler.fit_transform(df)
    return df, scaler


# Carregar dados (ajustar para o caminho dos arquivos do artigo)
df_sensex = pd.read_csv('data-nse/sensex.csv')

# Imprimir o dataframe carregado para verificar os dados
print("\nDataFrame Original:\n", df_sensex.head())

# Calcular indicadores técnicos
df_sensex['SMA_21'] = SMA(df_sensex, 21)
df_sensex['WMA_65'] = WMA(df_sensex, 65)
df_sensex['EMA_100'] = EMA(df_sensex, 100)
df_sensex['PPO'] = PPO(df_sensex)
df_sensex['PAIN'] = PAIN(df_sensex)
df_sensex['MACD'], df_sensex['Signal_Line'] = MACD(df_sensex)
df_sensex['RSI'] = RSI(df_sensex)
df_sensex['Momentum'] = Momentum(df_sensex, 1)
df_sensex['%K'] = Stochastic_K(df_sensex)
df_sensex['%D'] = Stochastic_D(df_sensex)

# Tratar dados ausentes
df_sensex.fillna(method='bfill', inplace=True)
df_lstm = df_sensex.copy()
# Imprimir o DataFrame após o cálculo dos indicadores técnicos
print("\nDataFrame com Indicadores Técnicos:\n", df_sensex[['SMA_21', 'WMA_65', 'EMA_100', 'PPO', 'PAIN', 'MACD', 'RSI', 'Momentum', '%K', '%D']].head())

# Transformar os indicadores técnicos em valores textuais
df_sensex['Open_trend'] = transformar_dados_texto(df_sensex, 'Open')
df_sensex['High_trend'] = transformar_dados_texto(df_sensex, 'High')
df_sensex['Low_trend'] = transformar_dados_texto(df_sensex, 'Low')
df_sensex['Close_trend'] = transformar_dados_texto(df_sensex, 'Close')
df_sensex['Volume_trend'] = transformar_dados_texto(df_sensex, 'Volume')
df_sensex['SMA_21_trend'] = transformar_dados_texto(df_sensex, 'SMA_21')
df_sensex['WMA_65_trend'] = transformar_dados_texto(df_sensex, 'WMA_65')
df_sensex['EMA_100_trend'] = transformar_dados_texto(df_sensex, 'EMA_100')
df_sensex['PPO_trend'] = transformar_dados_texto(df_sensex, 'PPO')
df_sensex['PAIN_trend'] = transformar_dados_texto(df_sensex, 'PAIN')
df_sensex['MACD_trend'] = transformar_dados_texto(df_sensex, 'MACD')
df_sensex['RSI_trend'] = transformar_dados_texto(df_sensex, 'RSI')
df_sensex['Momentum_trend'] = transformar_dados_texto(df_sensex, 'Momentum')
df_sensex['%K_trend'] = transformar_dados_texto(df_sensex, '%K')
df_sensex['%D_trend'] = transformar_dados_texto(df_sensex, '%D')

# Imprimir o DataFrame após a transformação em tendências
print("\nDataFrame com Tendências Textuais:\n", df_sensex[['SMA_21_trend', 'WMA_65_trend', 'EMA_100_trend', 'PPO_trend', 'PAIN_trend', 'MACD_trend', 'RSI_trend', 'Momentum_trend', '%K_trend', '%D_trend']].head())

# Criar a coluna "Action"
df_sensex['Action'] = definir_acao(df_sensex[['SMA_21_trend', 'WMA_65_trend', 'EMA_100_trend', 'PPO_trend', 'PAIN_trend', 'MACD_trend', 'RSI_trend', 'Momentum_trend', '%K_trend', '%D_trend']])

colunas_para_dropar = ['Signal_Line','Adj Close','Date','SMA_21', 'WMA_65', 'EMA_100', 'PPO', 'PAIN', 'MACD', 'RSI', 'Momentum', '%K', '%D','Open','Close', 'High', 'Low', 'Volume']  # Exemplo: vamos dropar estas colunas para um teste
df_sense = df_sensex.drop(columns=colunas_para_dropar)

print("Distribuição das classes:")
print(Counter(df_sense['Action']))
class_counts = Counter(df_sense['Action'])
total_samples = len(df_sense['Action'])

# Pesos para cada classe
class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
print("Pesos calculados para as classes:", class_weights)

print("\nDataFrame:\n", df_sense.head())

# Modelo com valores textuais
X_textual = df_sense[['SMA_21_trend', 'WMA_65_trend', 'EMA_100_trend', 'PPO_trend', 'PAIN_trend', 'MACD_trend', 'RSI_trend', 'Momentum_trend', '%K_trend', '%D_trend']]
y_textual = df_sense['Action']

# Convertendo valores textuais para numéricos
label_encoder = LabelEncoder()
for col in X_textual.columns:
    X_textual[col] = label_encoder.fit_transform(X_textual[col])
    
# Imprimir o DataFrame com valores numéricos após a codificação
print("\nDataFrame com valores codificados para treinamento:\n", X_textual.head())

# Gerar a árvore de decisão e salvar a imagem
print("\nModelo com valores textuais:")
#treinar_avaliar_arvore(X_textual, y_textual, X_textual.columns, class_weights)

# Aplicar o TimeSeriesSplit para treinar e avaliar o modelo em cada divisão
#treinar_avaliar_arvore_time_series(X_textual, y_textual, X_textual.columns, class_weights)

# Treinar e avaliar a árvore de decisão, obtendo as probabilidades
#treinar_avaliar_arvore_e_probabilidades(X_textual, y_textual, X_textual.columns, class_weights)


# Adicionar as probabilidades da árvore de decisão ao DataFrame
probabilidades = treinar_avaliar_arvore_e_probabilidades(X_textual, y_textual, X_textual.columns, class_weights=None)

# Concatenar as probabilidades no DataFrame original
df_lstm = pd.concat([df_lstm, probabilidades], axis=1)
#coluns =['Date','Action','SMA_21_trend', 'WMA_65_trend', 'EMA_100_trend', 'PPO_trend', 'PAIN_trend', 'MACD_trend', 'RSI_trend', 'Momentum_trend', '%K_trend', '%D_trend']
df_lstm.drop(columns='Date', inplace=True)
# Preencher valores ausentes
df_lstm.fillna(method='bfill', inplace=True)
print(df_lstm.head())
# **Normalizar todas as colunas do DataFrame para o LSTM**
# Vamos incluir todas as colunas numéricas, incluindo as colunas de probabilidades
df_lstm_normalizado, scaler = normalizar_dados(df_lstm)

# Preparar as janelas de dados para o LSTM sem probabilidades
X_lstm_sem_probas = []
y_lstm_sem_probas = []

for i in range(300, len(df_lstm_normalizado)):
    X_lstm_sem_probas.append(df_lstm_normalizado[i-300:i, 0])  # Primeira coluna, assumida como 'Close' após normalização
    y_lstm_sem_probas.append(df_lstm_normalizado[i, 0])

X_lstm_sem_probas = np.array(X_lstm_sem_probas)
y_lstm_sem_probas = np.array(y_lstm_sem_probas)

# Preparar as janelas de dados para o LSTM com probabilidades
X_lstm_com_probas = []
for i in range(300, len(df_lstm_normalizado)):
    X_lstm_com_probas.append(df_lstm_normalizado[i-300:i, -3:])  # Últimas 3 colunas, que são as probabilidades

X_lstm_com_probas = np.array(X_lstm_com_probas)

# Ajustar o shape para o LSTM
X_lstm_sem_probas = np.reshape(X_lstm_sem_probas, (X_lstm_sem_probas.shape[0], X_lstm_sem_probas.shape[1], 1))
X_lstm_com_probas = np.reshape(X_lstm_com_probas, (X_lstm_com_probas.shape[0], X_lstm_com_probas.shape[1], 3))

# Criar os modelos LSTM
input_shape_sem_probas = (300, 1)
input_shape_com_probas = (300, 3)

modelo_lstm_sem_probas = criar_modelo_lstm(input_shape_sem_probas)
modelo_lstm_com_probas = criar_modelo_lstm(input_shape_com_probas)

# Avaliar ambos os modelos
mse_sem_probas = avaliar_modelo_lstm(X_lstm_sem_probas, y_lstm_sem_probas, modelo_lstm_sem_probas)
mse_com_probas = avaliar_modelo_lstm(X_lstm_com_probas, y_lstm_sem_probas, modelo_lstm_com_probas)

print(f"MSE sem probabilidades: {mse_sem_probas}")
print(f"MSE com probabilidades: {mse_com_probas}")