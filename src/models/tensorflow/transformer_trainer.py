from models.base.base_trainer import BaseTrainer
from utils.validation import DataValidator
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping
import os

def transformer_encoder_block(
    inputs,
    embed_dim,
    num_heads,
    ff_dim,
    dropout_rate=0.1,
    activation="relu",
    l1_reg=0.0,
    l2_reg=0.0,
):
    # Multi-head Attention
    attention_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embed_dim,
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg)
    )(inputs, inputs)
    attention_output = Dropout(dropout_rate)(attention_output)
    # Add & Norm
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
    # Feed Forward Network
    ffn = Dense(ff_dim, activation=activation, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg))(out1)
    ffn = Dropout(dropout_rate)(ffn)
    ffn = Dense(embed_dim, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg))(ffn)
    # Add & Norm
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn)
    return out2

class TensorFlowTransformerTrainer(BaseTrainer):
    def __init__(
        self,
        input_shape,
        num_layers=2,
        embed_dim=32,
        num_heads=2,
        ff_dim=64,
        dropout=0.1,
        batch_size=32,
        epochs=50,
        patience=5,
        loss_fn="mean_squared_error",
        metrics=None,
        optimizer="adam",
        l1_reg=0.0,
        l2_reg=0.0,
        output_units=1,
        activation="relu",
        callbacks=None,
        learning_rate=0.001,
    ):
        self.input_shape = input_shape
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.loss_fn = loss_fn
        self.metrics = metrics or []
        self.optimizer = optimizer
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.output_units = output_units
        self.activation = activation
        self.callbacks = callbacks
        self.learning_rate = learning_rate
        self.model = None

        self.validate_args()

    def validate_args(self):
        DataValidator.validate_tuple(self.input_shape, min_length=2)
        DataValidator.validate_integer(self.num_layers, min_value=1)
        DataValidator.validate_integer(self.embed_dim, min_value=1)
        DataValidator.validate_integer(self.num_heads, min_value=1)
        DataValidator.validate_integer(self.ff_dim, min_value=1)
        DataValidator.validate_float(self.dropout, min_value=0.0, max_value=1.0)
        DataValidator.validate_integer(self.batch_size, min_value=1)
        DataValidator.validate_integer(self.epochs, min_value=1)
        DataValidator.validate_integer(self.patience, min_value=1)
        DataValidator.validate_string(self.loss_fn)
        DataValidator.validate_string(self.optimizer)
        DataValidator.validate_float(self.learning_rate, min_value=0.0)
        DataValidator.validate_integer(self.output_units, min_value=1)
        DataValidator.validate_string(self.activation)

    def build_model(self):
        inputs = Input(shape=self.input_shape)
        x = Dense(self.embed_dim)(inputs)
        for _ in range(self.num_layers):
            x = transformer_encoder_block(
                x,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout_rate=self.dropout,
                activation=self.activation,
                l1_reg=self.l1_reg,
                l2_reg=self.l2_reg,
            )
        x = Dense(16, activation="relu")(x)
        x = Dropout(self.dropout)(x)
        x = Dense(self.output_units)(x)
        outputs = x

        model = Model(inputs=inputs, outputs=outputs)

        if self.optimizer.lower() == "adam":
            optimizer = Adam(learning_rate=self.learning_rate)
        elif self.optimizer.lower() == "sgd":
            optimizer = SGD(learning_rate=self.learning_rate)
        else:
            raise ValueError(f"Otimizador '{self.optimizer}' não suportado.")

        model.compile(optimizer=optimizer, loss=self.loss_fn, metrics=self.metrics)
        self.model = model
        return model

    def train(self, X_train, y_train, X_val, y_val):
        if self.model is None:
            self.build_model()
        early_stopping = EarlyStopping(monitor="val_loss", patience=self.patience, restore_best_weights=True)
        callbacks = [early_stopping] + (self.callbacks or [])
        self.model.fit(
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
