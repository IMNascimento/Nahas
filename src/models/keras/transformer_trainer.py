from models.base.base_trainer import BaseTrainer
from utils.validation import DataValidator
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Lambda
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping
import tensorflow as tf
import numpy as np
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

class KerasTransformerTrainer(BaseTrainer):
    def __init__(
        self,
        input_shape: tuple = None,
        num_layers: int = 2,
        embed_dim: int = 32,
        num_heads: int = 2,
        ff_dim: int = 64,
        dropout: float = 0.1,
        batch_size: int = 32,
        epochs: int = 50,
        patience: int = 5,
        loss_fn: str = "mean_squared_error",
        metrics: list = None,
        optimizer: str = "adam",
        l1_reg: float = 0.0,
        l2_reg: float = 0.0,
        output_units: int = 1,
        activation: str = "relu",
        callbacks: list = None,
        learning_rate: float = 0.001,
        **kwargs
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
        self.callbacks = callbacks or []
        self.learning_rate = learning_rate
        self.model = None

        # Validação dos parâmetros
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
        DataValidator.validate_list(self.metrics, item_type=str, allow_empty=True)
        DataValidator.validate_string(self.optimizer)
        DataValidator.validate_float(self.l1_reg, min_value=0.0)
        DataValidator.validate_float(self.l2_reg, min_value=0.0)
        DataValidator.validate_integer(self.output_units, min_value=1)
        DataValidator.validate_string(self.activation)
        DataValidator.validate_float(self.learning_rate, min_value=0.0)

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
        x = Lambda(lambda t: t[:, -1, :])(x)  # Always extract last time step
        outputs = Dense(self.output_units)(x)
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
        callbacks = [early_stopping] + self.callbacks
        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        return self.model

    def finetune(self, X_train, y_train, X_val, y_val, config: dict = None):
        """
        Executa fine-tuning do modelo carregado. Se config for passado, atualiza hiperparâmetros.
        """
        if config:
            self.set_hyperparameters(**config)
            self.build_model()  # reconstrói arquitetura se mudou
        early_stopping = EarlyStopping(monitor="val_loss", patience=self.patience, restore_best_weights=True)
        callbacks = [early_stopping] + self.callbacks
        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        return self.model

    def save_model(self, model_path: str):
        directory = os.path.dirname(model_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.model.save(model_path)

    def load_model(self, model_path: str):
        self.model = load_model(model_path)
        return self.model

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y, metrics: list = None) -> dict:
        return self.model.evaluate(X, y, return_dict=True)

    def validate_model(self) -> bool:
        try:
            assert self.model is not None
            dummy = np.zeros((1, ) + self.input_shape)
            self.model.predict(dummy)
            return True
        except Exception as e:
            print(f"Erro de validação do modelo: {e}")
            return False

    def set_hyperparameters(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get_hyperparameters(self) -> dict:
        return {
            'input_shape': self.input_shape,
            'num_layers': self.num_layers,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout': self.dropout,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'patience': self.patience,
            'loss_fn': self.loss_fn,
            'metrics': self.metrics,
            'optimizer': self.optimizer,
            'l1_reg': self.l1_reg,
            'l2_reg': self.l2_reg,
            'output_units': self.output_units,
            'activation': self.activation,
            'callbacks': self.callbacks,
            'learning_rate': self.learning_rate
        }

    def feature_importance(self):
        raise NotImplementedError("Feature importance não implementado para Transformer.")

