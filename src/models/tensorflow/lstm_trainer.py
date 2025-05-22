# src/models/tensorflow/lstm_trainer.py
from models.base.base_trainer import BaseTrainer
import tensorflow as tf
import os

class TensorFlowLSTMTrainer(BaseTrainer):
    def __init__(
        self,
        input_shape,
        layers_config,
        dropout=0.2,
        recurrent_dropout=0.0,
        batch_size=32,
        epochs=50,
        patience=5,
        loss_fn="mse",
        metrics=None,
        optimizer="adam",
        l1_reg=0.0,
        l2_reg=0.0,
        bidirectional=False,
        output_units=1,
        activation_functions=None,
        kernel_initializer="glorot_uniform",
        callbacks=None,
        learning_rate=0.001,
    ):
        self.input_shape = input_shape
        self.layers_config = layers_config
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.loss_fn = loss_fn
        self.metrics = metrics or []
        self.optimizer = optimizer
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.bidirectional = bidirectional
        self.output_units = output_units
        self.activation_functions = activation_functions or ["tanh"] * len(layers_config)
        self.kernel_initializer = kernel_initializer
        self.callbacks = callbacks
        self.learning_rate = learning_rate
        self.model = None

    def build_model(self):
        inputs = tf.keras.Input(shape=self.input_shape)
        x = inputs
        for i, units in enumerate(self.layers_config):
            regularizer = tf.keras.regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg)
            activation = self.activation_functions[i]
            return_sequences = i != len(self.layers_config) - 1
            lstm_layer = tf.keras.layers.LSTM(
                units=units,
                activation=activation,
                return_sequences=return_sequences,
                kernel_regularizer=regularizer,
                dropout=self.dropout,
                recurrent_dropout=self.recurrent_dropout,
                kernel_initializer=self.kernel_initializer,
            )
            if self.bidirectional:
                x = tf.keras.layers.Bidirectional(lstm_layer)(x)
            else:
                x = lstm_layer(x)
        outputs = tf.keras.layers.Dense(self.output_units)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        if self.optimizer.lower() == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer.lower() == "sgd":
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        else:
            raise ValueError(f"Otimizador '{self.optimizer}' não suportado.")

        model.compile(optimizer=optimizer, loss=self.loss_fn, metrics=self.metrics)
        self.model = model
        return model

    def train(self, X_train, y_train, X_val, y_val):
        if self.model is None:
            self.build_model()
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.patience, restore_best_weights=True)
        callbacks = [early_stopping] + (self.callbacks or [])
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks
        )
        return self.model

    def save_model(self, model_path: str):
        directory = os.path.dirname(model_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.model.save(model_path)

    def load_model(self, model_path: str):
        self.model = tf.keras.models.load_model(model_path)
        return self.model

    def predict(self, X):
        return self.model.predict(X)
