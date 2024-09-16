"""
To replicate the stock market forecasting model described in the article, the dataset is collected from two leading Indian stock markets: BSE Sensex and NSE Nifty, covering the period from January 2008 to July 2015. The data consists of the daily values of Open, High, Low, Close, and Volume, along with 12 technical indicators, such as:

Highest high (past 21 days)
Lowest low (past 21 days)
SMA (Simple Moving Average of the last 21 days)
WMA (Weighted Moving Average of the past 65 days)
EMA (Exponential Moving Average for the past 100 days)
PPO (Price Oscillator in percentage)
PAIN (Price Action Indicator based on Open, Close, High, and Low)
MACD (Moving Average Convergence Divergence)
RSI (Relative Strength Index, 14-day)
Momentum (Difference between the previous and current Close)
%k (Stochastic percentage alert line)
%D (Stochastic percentage definitive line)​

"""


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_absolute_error
from sklearn.metrics import mean_squared_error, cohen_kappa_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from src.utils.csv_handler import CSVHandler



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









data_sensex=[
    'data/bse/Download Data - INDEX_IN_XBOM_1.csv',
    'data-nse/Download Data - INDEX_IN_XBOM_1 (1).csv',
    'data-nse/Download Data - INDEX_IN_XBOM_1 (2).csv',
    'data-nse/Download Data - INDEX_IN_XBOM_1 (3).csv',
    'data-nse/Download Data - INDEX_IN_XBOM_1 (4).csv',
    'data-nse/Download Data - INDEX_IN_XBOM_1 (5).csv',
    'data-nse/Download Data - INDEX_IN_XBOM_1 (6).csv',
    'data-nse/Download Data - INDEX_IN_XBOM_1 (7).csv'
]
data_nifty=[
    'data-nse/NIFTY 50-01-01-2008-to-31-12-2008.csv',
    'data-nse/NIFTY 50-01-01-2009-to-31-12-2009.csv',
    'data-nse/NIFTY 50-01-01-2010-to-31-12-2010.csv',
    'data-nse/NIFTY 50-01-01-2011-to-31-12-2011.csv',
    'data-nse/NIFTY 50-01-01-2012-to-31-12-2012.csv',
    'data-nse/NIFTY 50-01-01-2013-to-31-12-2013.csv',
    'data-nse/NIFTY 50-01-01-2014-to-31-12-2014.csv',
    'data-nse/NIFTY 50-01-01-2015-to-31-12-2015.csv'
]
# Load dataset (Assuming the data is in a CSV file format)
#df_sensex = CSVHandler.read_multiple_csvs(data_sensex)data_sensex
df_sensex = CSVHandler.read_from_csv('data-nse/sensex.csv')
print(df_sensex.head())
df_nifty = CSVHandler.read_multiple_csvs(data_nifty)


# Remover vírgulas e converter colunas para numérico
#df_sensex['Close'] = df_sensex['Close'].str.replace(',', '').astype(float)
#df_sensex['Open'] = df_sensex['Open'].str.replace(',', '').astype(float)
#df_sensex['High'] = df_sensex['High'].str.replace(',', '').astype(float)
#df_sensex['Low'] = df_sensex['Low'].str.replace(',', '').astype(float)
#df_sensex['Volume'] = df_sensex['Volume'].str.replace(',', '').astype(float) 



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

# Seleção de Features e Target
features = ['High', 'Low', 'SMA_21', 'WMA_65', 'EMA_100', 'PPO', 'PAIN', 'MACD', 'RSI', 'Momentum', '%K', '%D']
X = df_sensex[features]
y = df_sensex['Action']  # Assumindo que já exista uma coluna 'Action'

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinamento da árvore de decisão
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Previsões
y_pred = clf.predict(X_test)

# Métricas de Avaliação
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
kappa = cohen_kappa_score(y_test, y_pred)

# Matriz de Confusão para calcular True Positive Rate e False Positive Rate
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()  # Ajuste se você tiver mais de duas classes
true_positive_rate = tp / (tp + fn)
false_positive_rate = fp / (fp + tn)

# Mean Absolute Error e Root Mean Squared Error
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Relative Absolute Error e Root Relative Squared Error
rae = mae / np.mean(np.abs(y_test - np.mean(y_test)))
rrse = rmse / np.sqrt(np.mean((y_test - np.mean(y_test)) ** 2))

# Exibindo Resultados
print(f"Accuracy: {accuracy}")
print(f"True Positive Rate: {true_positive_rate}")
print(f"False Positive Rate: {false_positive_rate}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Kappa Statistic: {kappa}")
print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Relative Absolute Error: {rae}")
print(f"Root Relative Squared Error: {rrse}")