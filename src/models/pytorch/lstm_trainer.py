# src/models/pytorch/lstm_trainer.py
from models.base.base_trainer import BaseTrainer
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np

class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size,
        layers_config,
        dropout=0.2,
        bidirectional=False,
        activation_functions=None,
        output_units=1
    ):
        super(LSTMModel, self).__init__()
        self.num_layers = len(layers_config)
        self.hidden_layers = layers_config
        self.bidirectional = bidirectional

        self.lstm_layers = nn.ModuleList()
        self.activation_functions = activation_functions or ["tanh"] * len(layers_config)

        last_size = input_size
        for i, hidden_size in enumerate(layers_config):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=last_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=bidirectional,
                    dropout=dropout if i < len(layers_config) - 1 else 0.0,
                )
            )
            last_size = hidden_size * (2 if bidirectional else 1)

        self.fc = nn.Linear(last_size, output_units)

    def forward(self, x):
        out = x
        for lstm_layer in self.lstm_layers:
            out, _ = lstm_layer(out)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class PyTorchLSTMTrainer(BaseTrainer):
    def __init__(
        self,
        input_shape,
        layers_config,
        dropout=0.2,
        batch_size=32,
        epochs=50,
        patience=5,
        loss_fn="mse",
        metrics=None,
        optimizer="adam",
        learning_rate=0.001,
        bidirectional=False,
        activation_functions=None,
        output_units=1,
        device=None
    ):
        self.input_shape = input_shape
        self.layers_config = layers_config
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.loss_fn = loss_fn
        self.metrics = metrics or []
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.bidirectional = bidirectional
        self.activation_functions = activation_functions
        self.output_units = output_units
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def build_model(self):
        input_size = self.input_shape[1]
        self.model = LSTMModel(
            input_size=input_size,
            layers_config=self.layers_config,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            activation_functions=self.activation_functions,
            output_units=self.output_units
        ).to(self.device)
        return self.model

    def train(self, X_train, y_train, X_val, y_val):
        if self.model is None:
            self.build_model()
        # Converta os arrays para tensores PyTorch
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

            # Validação
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val)

            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                # Salva modelo temporário
                torch.save(self.model.state_dict(), "best_model_temp.pt")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print("Early stopping!")
                    break

            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {loss.item()}, Val Loss: {val_loss.item()}")

        # Carrega melhor modelo
        self.model.load_state_dict(torch.load("best_model_temp.pt"))
        os.remove("best_model_temp.pt")
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
