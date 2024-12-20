import pandas as pd
from models.db.model_binance import HourlyQuote
from data.data_processing import DataProcessor
from models.lstm_model import CustomLSTMTrainer
from optimization.grid_search import GridSearch
import tensorflow as tf
from tensorflow.keras import backend as K

# Configurações principais
STEPS_AHEAD = 1
END_DATE = "2024-09-14 23:59:59"
RELEVANT_COLUMNS = ["open", "high", "low", "close", "volume"]
TARGET_COLUMN = "close"

# Configurar o MirroredStrategy para múltiplas GPUs
strategy = tf.distribute.MirroredStrategy()
print(f"Número de dispositivos detectados: {strategy.num_replicas_in_sync}")

# Função de treino adaptada para integração com GridSearch
def model_trainer(X_train, y_train, X_val, y_val, dropout, batch_size, epochs, patience, layers_config, window_size):
    """
    Função para treinar e avaliar o modelo com os hiperparâmetros fornecidos.
    Retorna a métrica de avaliação (ex: validação loss ou acurácia).
    """
    input_shape = (X_train.shape[1], X_train.shape[2])
    with strategy.scope():
        trainer = CustomLSTMTrainer(
            input_shape=input_shape,
            layers_config=layers_config,
            dropout=dropout,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience
        )
        model = trainer.train(X_train, y_train, X_val, y_val)
    val_loss = model.evaluate(X_val, y_val, verbose=0)  # Retorna a métrica desejada
    return val_loss


# Função para carregar os dados
def load_data_from_db(end_date):
    query = (HourlyQuote
             .select()
             .where(HourlyQuote.timestamp <= end_date)
             .order_by(HourlyQuote.timestamp))
    data = pd.DataFrame(list(query.dicts()))
    if not data.empty:
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        return data
    else:
        print("Nenhum dado encontrado no banco.")
        return pd.DataFrame()

if __name__ == "__main__":
    # Carregar os dados do banco
    data_df = load_data_from_db(END_DATE)
    data_df = data_df[RELEVANT_COLUMNS]
    if data_df.empty:
        raise ValueError("Nenhum dado foi recuperado do banco de dados. Verifique a data ou os dados disponíveis.")

    # Inicializar o DataProcessor
    processor = DataProcessor(window_size=48)

    # Definir espaço de busca de hiperparâmetros
    param_grid = {
        "dropout": [0.2, 0.3, 0.4],
        "batch_size": [16, 32, 64, 128],
        "epochs": [50],
        "patience": [5],
        "layers_config": [
            [64, 32], [128, 64], [256, 128], [128, 64, 32], [256, 128, 64], [512, 256, 128],
            [64, 64], [128, 128], [256, 256], [128, 128, 128], [256, 256, 256], [512, 512, 512],
            [64, 64, 32, 32], [128, 128, 64, 64], [256, 256, 128, 128], [512, 512, 256, 256]
        ],
        "window_size": [48, 72, 96, 120]
    }

    # Inicializar GridSearch
    grid_search = GridSearch(
        model_trainer=model_trainer,
        param_grid=param_grid,
        scoring='loss',  # Métrica utilizada
        verbose=1
    )

    # Executar GridSearch
    results = grid_search.search(
        data_processor=processor,
        data_df=data_df,
        coluna_alvo=TARGET_COLUMN,
        steps_ahead=STEPS_AHEAD
    )

    # Exibir os melhores parâmetros e resultados
    print("Melhores hiperparâmetros encontrados:")
    print(results['best_params'])
    print(f"Melhor pontuação: {results['best_score']}")

    # Salvar resultados detalhados
    results['results'].to_csv("grid_search_results.csv", index=False)
    print("Resultados detalhados salvos em 'grid_search_results.csv'")
