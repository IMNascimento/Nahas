# lstm_model.py
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import os

class CustomLSTMTrainerGPU:
    def __init__(
        self, input_shape, layers_config=None, dropout=None, batch_size=None,
        epochs=None, patience=None
    ):
        self.input_shape = input_shape
        self.layers_config = layers_config or [50]
        self.dropout = dropout or 0.2
        self.batch_size = batch_size or 32
        self.epochs = epochs or 50
        self.patience = patience or 5

    def build_model(self):
        model = Sequential()

        for i, units in enumerate(self.layers_config):
            if i == 0:
                model.add(LSTM(units=units, return_sequences=True, input_shape=self.input_shape))
            elif i == len(self.layers_config) - 1:
                model.add(LSTM(units=units, return_sequences=False))
            else:
                model.add(LSTM(units=units, return_sequences=True))
            model.add(Dropout(self.dropout))

        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train(self, X_train, y_train, X_val, y_val):
        print(f"Dimensões de X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"Dimensões de X_val: {X_val.shape}, y_val: {y_val.shape}")
        model = self.build_model()

        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stopping]
        )
        return model

    def save_model(self, model, model_path='result/models/lstm_model.h5'):
        directory = os.path.dirname(model_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        model.save(model_path)
        print(f"Modelo salvo em: {model_path}")

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"O modelo no caminho {model_path} não foi encontrado.")
        return load_model(model_path)