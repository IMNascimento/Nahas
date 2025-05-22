import torch
import torch.nn as nn
import torch.optim as optim
from models.base.base_trainer import BaseTrainer
from utils.validation import DataValidator
import os
import numpy as np

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, activation="relu"):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            getattr(nn, activation.capitalize())(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-Head Attention
        attn_output, _ = self.attn(x, x, x)
        x = self.ln1(x + self.dropout(attn_output))
        # Feedforward
        ffn_output = self.ffn(x)
        x = self.ln2(x + self.dropout(ffn_output))
        return x

class PyTorchTransformerModel(nn.Module):
    def __init__(self, input_dim, num_layers, embed_dim, num_heads, ff_dim, dropout, activation, output_units):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, ff_dim, dropout=dropout, activation=activation)
            for _ in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, output_units)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = self.embedding(x)
        for block in self.encoder_blocks:
            x = block(x)
        # Pool last time-step (padrão para série temporal)
        x = x[:, -1, :]  # shape: (batch, embed_dim)
        x = self.head(x)
        return x

class PyTorchTransformerTrainer(BaseTrainer):
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
        loss_fn="mse",
        metrics=None,
        optimizer="adam",
        learning_rate=0.001,
        output_units=1,
        activation="relu",
        device=None,
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
        self.learning_rate = learning_rate
        self.output_units = output_units
        self.activation = activation
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
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
        input_dim = self.input_shape[1]
        self.model = PyTorchTransformerModel(
            input_dim=input_dim,
            num_layers=self.num_layers,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            dropout=self.dropout,
            activation=self.activation,
            output_units=self.output_units,
        ).to(self.device)
        return self.model

    def train(self, X_train, y_train, X_val, y_val):
        if self.model is None:
            self.build_model()

        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        criterion = nn.MSELoss() if self.loss_fn == "mse" else nn.L1Loss()
        if self.optimizer.lower() == "adam":
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer.lower() == "sgd":
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Otimizador '{self.optimizer}' não suportado.")

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val)

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), "best_transformer_temp.pt")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print("Early stopping!")
                    break

            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}")

        self.model.load_state_dict(torch.load("best_transformer_temp.pt"))
        os.remove("best_transformer_temp.pt")
        return self.model

    def save_model(self, model_path: str):
        torch.save(self.model.state_dict(), model_path)

    def load_model(self, model_path: str):
        self.build_model()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        return self.model

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
        return outputs.cpu().numpy()
