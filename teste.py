import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler

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


# Supondo que você tenha uma coluna 'Close' como variável alvo
X = df_sensex[['SMA_21', 'WMA_65', 'EMA_100', 'PPO', 'PAIN', 'MACD', 'Signal_Line', 'RSI', 'Momentum', '%K', '%D']]
y = df_sensex['Close']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


# Normalizar os dados
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Criar janelas para o LSTM
window_size = 30
def create_lstm_data(X, y, window_size):
    X_lstm, y_lstm = [], []
    for i in range(window_size, len(X)):
        X_lstm.append(X[i-window_size:i])
        y_lstm.append(y[i])
    return np.array(X_lstm), np.array(y_lstm)

X_train_lstm, y_train_lstm = create_lstm_data(X_train_scaled, y_train.values, window_size)
X_test_lstm, y_test_lstm = create_lstm_data(X_test_scaled, y_test.values, window_size)

# Reshape para [samples, time steps, features]
X_train_lstm = np.reshape(X_train_lstm, (X_train_lstm.shape[0], X_train_lstm.shape[1], X_train_lstm.shape[2]))
X_test_lstm = np.reshape(X_test_lstm, (X_test_lstm.shape[0], X_test_lstm.shape[1], X_test_lstm.shape[2]))

# Construir o modelo LSTM
model_lstm = Sequential()
model_lstm.add(LSTM(70, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
model_lstm.add(Dropout(0.4))
model_lstm.add(LSTM(70, return_sequences=False))
model_lstm.add(Dropout(0.4))
model_lstm.add(Dense(1))

model_lstm.compile(optimizer='adam', loss='mean_squared_error')

# Treinar o modelo LSTM
model_lstm.fit(X_train_lstm, y_train_lstm, epochs=200, batch_size=32, validation_data=(X_test_lstm, y_test_lstm))

# Fazer previsões com o LSTM no conjunto de teste
lstm_predictions = model_lstm.predict(X_test_lstm)

# Fazer previsões com o modelo de Regressão Linear
reg = LinearRegression()
reg.fit(X_train, y_train)
linear_predictions = reg.predict(X_test[window_size:])

# Combinar previsões LSTM e Regressão Linear (Modelo Híbrido)
hybrid_predictions = 0.5 * lstm_predictions.flatten() + 0.5 * linear_predictions

# Avaliar as métricas
rmse_lstm = np.sqrt(mean_squared_error(y_test_lstm, lstm_predictions))
mae_lstm = mean_absolute_error(y_test_lstm, lstm_predictions)

rmse_linear = np.sqrt(mean_squared_error(y_test[window_size:], linear_predictions))
mae_linear = mean_absolute_error(y_test[window_size:], linear_predictions)

rmse_hybrid = np.sqrt(mean_squared_error(y_test_lstm, hybrid_predictions))
mae_hybrid = mean_absolute_error(y_test_lstm, hybrid_predictions)

# Imprimir as métricas
print(f"RMSE - Regressão Linear: {rmse_linear}")
print(f"MAE - Regressão Linear: {mae_linear}")

print(f"RMSE - LSTM: {rmse_lstm}")
print(f"MAE - LSTM: {mae_lstm}")

print(f"RMSE - Modelo Híbrido: {rmse_hybrid}")
print(f"MAE - Modelo Híbrido: {mae_hybrid}")