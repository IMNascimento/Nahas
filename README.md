# Nahas

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-GNU%20GPL%20v3-blue.svg)
![Python](https://img.shields.io/badge/python-3.10.17-blue)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.18.0-orange)
![PyTorch](https://img.shields.io/badge/pytorch-2.5.1-red)

## Visão Geral

**Nahas** é uma plataforma de previsão de séries temporais financeiras baseada em aprendizado profundo, com foco em criptomoedas. O sistema integra redes neurais LSTM e Transformer em três frameworks distintos (Keras, TensorFlow e PyTorch), oferecendo uma interface web interativa para treinamento, otimização de hiperparâmetros, fine-tuning de modelos e execução de previsões em tempo real via API da Binance.

A plataforma também implementa uma normalização multi-escala **inspirada no EvoMSN** (*Evolving Multi-Scale Normalization*, Qin et al., 2024): seleção de periodicidades por transformada de Fourier, normalização por fatia, predição das estatísticas futuras e combinação ponderada das previsões. É uma variante **offline**: não implementa a otimização evolutiva *online* do método original, que alterna atualizações entre o modelo de sequência e o módulo estatístico. As menções a EvoMSN neste repositório devem ser lidas nesse sentido restrito.

---

## Funcionalidades

- **Treinamento de modelos** LSTM e Transformer via interface web
- **Suporte multi-framework**: Keras, TensorFlow e PyTorch com interface unificada
- **Grid Search automatizado** com suporte a multi-GPU para otimização de hiperparâmetros
- **Fine-tuning** de modelos pré-treinados com comparação automática de desempenho e backup
- **Previsão em tempo real** com integração à API da Binance
- **Normalização avançada**: Global, Local, EvoMSN e EvoMSN-like
- **Indicadores técnicos**: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, ADX, Estocástico, Envelopes, Fibonacci
- **Persistência de resultados** em banco de dados MySQL via ORM (Peewee)
- **Exportação de resultados** para CSV e visualizações gráficas
- **Suporte a múltiplos data sources**: Binance, Yahoo Finance e MetaTrader 5

---

## Pré-requisitos

Antes de começar, certifique-se de ter instalado:

| Dependência | Versão |
|---|---|
| Python | 3.10.17 |
| CUDA | 12.1 |
| cuDNN | 9.x |
| MySQL | 8.x |
| PyTorch | 2.5.1 |
| TensorFlow | 2.18.0 |
| Keras | 3.7.0 |

> Para treinar sem GPU, defina `USE_GPU=False` no arquivo `.env`. O desempenho será significativamente menor.

---

## Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/SophiaMind/Nahas.git
cd Nahas
```

### 2. Crie e ative o ambiente virtual

```bash
python3 -m venv src/venv
source src/venv/bin/activate        # Linux/macOS
# .\src\venv\Scripts\activate       # Windows
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt            # o que o código realmente importa
pip install -r requirements-dev.txt        # + pytest, ruff e pip-audit
pip install -r requirements-optional.txt   # + torch e MetaTrader5 (opcionais)
pip install -r requirements-full.txt       # congelamento do ambiente dos experimentos
```

O ambiente em que os experimentos da dissertação rodaram **não** tinha `torch` nem
`MetaTrader5` instalados. Escolher o framework `pytorch` na interface sem instalar os
extras levanta `ImportError` com a instrução correspondente.

### 4. Configure o ambiente

Copie o arquivo de exemplo e preencha com suas credenciais:

```bash
cp src/.env.example src/.env
```

Edite `src/.env`:

```env
# Banco de Dados (MySQL)
NAME_DB="nome_do_banco"
USER_DB="usuario"
PASSWORD_DB="senha"
HOST_DB="localhost"
PORT_DB="3306"

# E-mail (opcional, para notificações)
EMAIL_SMTP="smtp.gmail.com"
EMAIL_PORT="587"
EMAIL_USER="seu@email.com"
EMAIL_PASSWORD="sua_senha"
EMAIL_IMAP="imap.gmail.com"

# MetaTrader 5 (opcional)
LOGIN_MT5=seu_login
PASSWORD_MT5=sua_senha
SERVER_MT5='mt5.xpi.com.br:443'

# Binance (obrigatório para previsão em tempo real)
BINANCE_API_KEY=sua_chave_api
BINANCE_API_SECRET_KEY=sua_chave_secreta

# Configurações do modelo
USE_GPU=True   # True para GPU, False para CPU
SEED=22
```

### 5. Inicialize o banco de dados

```bash
cd src
python database/init_db.py
```

---

## Execução

### Interface Web (modo principal)

```bash
cd src
streamlit run web/app.py
```

Acesse a aplicação em `http://localhost:8501`.

A interface possui quatro abas:

| Aba | Descrição |
|---|---|
| **Training** | Treinamento de um modelo com configuração completa de hiperparâmetros |
| **Grid Search** | Otimização automática de hiperparâmetros via busca em grade (JSON ou UI) |
| **Fine-tuning** | Atualização de modelos pré-treinados com novos dados e comparação de desempenho |
| **Live Run** | Previsão em tempo real com dados da Binance |

---

## Estrutura do Projeto

```
Nahas/
├── requirements.txt
├── LICENSE
├── README.md
└── src/
    ├── .env.example             # Template de variáveis de ambiente
    ├── config/
    │   ├── settings.py          # Carregamento de variáveis de ambiente e configuração de GPU
    │   └── setup_db.py
    ├── data/
    │   ├── data_processing.py   # Janelamento, normalização e divisão de dados
    │   └── evomsn_normalizer.py # Normalização EvoMSN e EvoMSN-like
    ├── database/
    │   ├── init_db.py           # Criação das tabelas no banco
    │   ├── model_nahas.py       # Modelos ORM: TrainingRun, FineTuningRun, GridResult
    │   ├── model_binance.py     # Histórico de preços da Binance
    │   └── model_nocapital.py
    ├── models/
    │   ├── base/                # Classe abstrata BaseTrainer
    │   ├── keras/               # LSTMTrainer e TransformerTrainer (Keras)
    │   ├── pytorch/             # LSTMTrainer e TransformerTrainer (PyTorch)
    │   └── tensorflow/          # LSTMTrainer e TransformerTrainer (TensorFlow)
    ├── optimization/
    │   └── grid_search.py       # Grid search com suporte a multi-GPU
    ├── services/
    │   ├── model_service.py     # Orquestração: treino, fine-tuning e previsão
    │   ├── trainer_factory.py   # Factory para instanciação dinâmica de trainers
    │   ├── binance.py           # Cliente da API Binance (dados e ordens)
    │   ├── yahoofinance.py      # Fonte de dados Yahoo Finance
    │   ├── metatrader.py        # Integração MetaTrader 5
    │   └── sync_binance_price_history.py
    ├── utils/
    │   ├── technical_indicators.py  # 11+ indicadores técnicos
    │   ├── capacity_train.py        # Estimativa de memória GPU e escalonamento
    │   ├── csv_exporter.py
    │   ├── plotter.py
    │   ├── model_comparison.py
    │   └── logger.py
    ├── web/
    │   ├── app.py               # Aplicação Streamlit principal
    │   ├── components/          # Componentes de UI por aba
    │   └── style/
    └── results/
        └── train/               # Modelos e artefatos salvos
```

---

## Fluxo de Dados

```
Fonte de dados (Binance / Yahoo Finance / MetaTrader 5)
        │
        ▼
Processamento (janelamento, indicadores técnicos, normalização)
        │
        ▼
Divisão treino / validação / teste
        │
        ▼
Treinamento (Keras / TensorFlow / PyTorch via TrainerFactory)
   ├── Treino único
   ├── Grid Search (paralelo, multi-GPU)
   └── Fine-tuning (com backup e comparação)
        │
        ▼
Avaliação (RMSE, MAE, MSE, R²)
        │
        ▼
Persistência (MySQL via Peewee ORM)
        │
        ▼
Previsão em tempo real (Binance API)
```

---

## Tecnologias Utilizadas

- **Deep Learning**: TensorFlow 2.18, Keras 3.7, PyTorch 2.5.1
- **Interface Web**: Streamlit 1.45
- **Banco de Dados**: MySQL + Peewee ORM
- **Dados Financeiros**: python-binance, yfinance, MetaTrader 5
- **Data Science**: Pandas, NumPy, Scikit-learn, SciPy
- **Visualização**: Matplotlib, Seaborn
- **Ambiente**: CUDA 12.1, cuDNN 9.x

---

## Autores

- **Igor Muniz Nascimento** — Desenvolvedor Principal — [GitHub](https://github.com/IMNascimento)

---

## Reprodução dos resultados da dissertação

As correções de metodologia aplicadas depois dos experimentos mudaram os **padrões** do código.
Para reproduzir exatamente os números publicados, use a configuração abaixo:

```json
{
  "window_size": 96,
  "steps_ahead": 1,
  "train_size": 0.70,
  "validation_split": 0.15,
  "target_column": "close",
  "relevant_columns": ["close", "open", "high", "low", "volume"],
  "include_target_channel": "legacy",
  "embargo": 0,
  "seed": 1482563973,
  "normalization": {
    "strategy": "local",
    "x_mode": "minmax",
    "y_mode": "minmax_target",
    "evomsn_short_horizon_policy": "legacy",
    "evomsn_period_scaling": "legacy"
  }
}
```

As cinco sementes da seção de repetição são `1482563973`, `7`, `12345`, `98765` e `20260811`.

Com os **padrões atuais** (`equalized`, embargo positivo, `window_stats`, `per_channel`) os
números são diferentes, e é isso que se espera: os padrões são as versões corrigidas.

---

## Limitações conhecidas

Declaradas aqui de propósito, para que sejam lidas antes de serem descobertas.

- **Conjunto de features entre estratégias.** No modo `legacy`, a normalização por janela recebe a
  série do alvo como canal extra de `X` e as demais estratégias não. Isso confunde o efeito da
  normalização com o efeito daquele canal. A chave `include_target_channel` (`equalized`, `never`,
  `legacy`) e o script `src/experiments/run_confound_matrix.py` medem esse efeito. Primeira medição,
  BTCUSDT, uma semente, 10 épocas: dar o fechamento ao braço global reduz o erro dele em 34% (4701
  para 3100 de RMSE) e tirá-lo do braço local piora 17% (343 para 401). Com features iguais dos dois
  lados a vantagem da normalização por janela permanece, em 9,0 vezes com cinco canais e 11,7 vezes
  com quatro. Detalhes em [CORRECOES_AUDITORIA.md](CORRECOES_AUDITORIA.md).
- **Degenerescência multi-escala sob horizonte unitário.** Com `H = 1`, a dispersão do alvo por
  fatia é nula por construção e a saída da rede é anulada na desnormalização. A política
  `short_horizon_policy` (`window_stats`, `legacy`, `error`) trata o caso; os resultados publicados
  usam `legacy`, que reproduz o comportamento degenerado.
- **Seleção de períodos por FFT.** Sob `period_scaling="legacy"`, a média de amplitudes entre canais
  é dominada pelo canal de maior escala. O padrão `per_channel` padroniza cada canal antes da
  transformada.
- **Escolha de hiperparâmetros.** O grid pontua na **validação** por padrão. Pontuar no teste
  continua possível, mas transforma o teste em conjunto de seleção e não deve ser usado para
  relatar desempenho.
- **Tempo limite por sinal do sistema.** O mecanismo de timeout usa `SIGALRM`, que existe apenas em
  sistemas Unix e apenas na thread principal.
- **Partição única.** O protocolo usa retenção temporal fixa, sem validação progressiva.

---

## Licença

Distribuído sob a licença **GNU General Public License v3.0**. Consulte o arquivo [LICENSE](LICENSE) para mais informações.

---

## Citação

Se você utilizar este projeto em sua pesquisa, por favor cite o seguinte artigo:

```bibtex
@article{,
  author    = {},
  title     = {},
  journal   = {},
  year      = {},
  volume    = {},
  number    = {},
  pages     = {},
  doi       = {}
}
```
