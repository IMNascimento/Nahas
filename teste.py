import pandas as pd

df = pd.read_csv('output/yahoo/ITUB4/month/ITUB4_month_2010_2020.csv')
# DataFrame com os preços do ativo
df['SMA_50'] = df['close'].rolling(window=50).mean()  # Média Móvel de 50 períodos
df['SMA_200'] = df['close'].rolling(window=200).mean()  # Média Móvel de 200 períodos

import ta

# DataFrame com os preços
df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

df['SMA_20'] = df['close'].rolling(window=20).mean()
df['stddev'] = df['close'].rolling(window=20).std()
df['Upper_Band'] = df['SMA_20'] + (df['stddev'] * 2)
df['Lower_Band'] = df['SMA_20'] - (df['stddev'] * 2)

df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Defina suas features (RSI, MACD, Bollinger Bands, etc.) e a label (compra ou não)
X = df[['RSI', 'MACD', 'Upper_Band', 'Lower_Band']]  # Seus indicadores
y = df['buy_signal']  # Define um sinal de compra como target (0 ou 1)

# Divida os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treine o modelo de árvore de decisão
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Avalie a árvore de decisão
accuracy = model.score(X_test, y_test)
print(f'Accuracy: {accuracy}')



import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import ta  # Biblioteca para indicadores técnicos

# Supondo que você tenha um DataFrame `df` com os preços de fechamento
df['SMA_50'] = df['close'].rolling(window=50).mean()
df['SMA_200'] = df['close'].rolling(window=200).mean()
df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

# Bollinger Bands
df['SMA_20'] = df['close'].rolling(window=20).mean()
df['stddev'] = df['close'].rolling(window=20).std()
df['Upper_Band'] = df['SMA_20'] + (df['stddev'] * 2)
df['Lower_Band'] = df['SMA_20'] - (df['stddev'] * 2)

# Sinal de compra (simplificação: pode ser qualquer regra)
df['buy_signal'] = (df['RSI'] < 30).astype(int)  # Exemplo: compra quando RSI < 30

# Defina as features (indicadores) e o label (buy_signal)
X = df[['RSI', 'MACD', 'Upper_Band', 'Lower_Band']]  # Indicadores
y = df['buy_signal']  # Sinal de compra (0 ou 1)

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinamento da árvore de decisão
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Avaliação do modelo
accuracy = model.score(X_test, y_test)
print(f'Accuracy with indicators only: {accuracy}')



# Suponha que você já tenha as previsões do LSTM no DataFrame
# df['lstm_prediction'] contém a previsão do LSTM para o preço futuro

# Atualize as features para incluir a previsão do LSTM
X_with_lstm = df[['RSI', 'MACD', 'Upper_Band', 'Lower_Band', 'lstm_prediction']]

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_with_lstm, y, test_size=0.2, random_state=42)

# Treinamento da árvore de decisão com LSTM
model_with_lstm = DecisionTreeClassifier()
model_with_lstm.fit(X_train, y_train)

# Avaliação do modelo com LSTM
accuracy_with_lstm = model_with_lstm.score(X_test, y_test)
print(f'Accuracy with indicators and LSTM prediction: {accuracy_with_lstm}')





















import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import ta

# Supondo que você tenha um DataFrame `df` com os preços de fechamento
# Cálculo dos indicadores técnicos
df['SMA_50'] = df['close'].rolling(window=50).mean()
df['SMA_200'] = df['close'].rolling(window=200).mean()
df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

# Bollinger Bands
df['SMA_20'] = df['close'].rolling(window=20).mean()
df['stddev'] = df['close'].rolling(window=20).std()
df['Upper_Band'] = df['SMA_20'] + (df['stddev'] * 2)
df['Lower_Band'] = df['SMA_20'] - (df['stddev'] * 2)

# Sinal de compra (simplificação: pode ser qualquer regra)
df['buy_signal'] = (df['RSI'] < 30).astype(int)  # Exemplo: compra quando RSI < 30

# Defina as features (indicadores) e o label (buy_signal)
X = df[['RSI', 'MACD', 'Upper_Band', 'Lower_Band']]  # Indicadores técnicos
y = df['buy_signal']  # Sinal de compra (0 ou 1)

# Divida os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinamento da árvore de decisão
model = DecisionTreeClassifier()

# Validação cruzada com 5 dobras (StratifiedKFold para manter a distribuição de classes)
skf = StratifiedKFold(n_splits=5)

# Métricas com validação cruzada
scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
print(f'Cross-validated Accuracy: {scores.mean()}')

# Treinar o modelo nos dados de treino
model.fit(X_train, y_train)

# Previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliação detalhada do modelo
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# Métrica F1
f1 = f1_score(y_test, y_pred)
print(f'F1-Score: {f1}')
