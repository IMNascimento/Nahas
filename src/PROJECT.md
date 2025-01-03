# Architecture

Nahas/
│
├── src/             # Diretório principal do código-fonte
│   ├── config/      # Configurações específicas do projeto
│   │   ├──__init__.py
│   │   └── settings.py     # Configurações gerais (e.g., banco de dados), api e configurações específicas do modelo LSTM
│   │
│   │
│   ├── data/               # processamento, limpeza e gerenciamento de dados
│   │   ├──__init__.py
│   │   └── data_preprocessing.py  # Scripts de pré-processamento de dados
│   │
│   │
│   ├── models/             # Modelos de IA e banco de dados
│   │   ├── db/
│   │   │   ├──__init__.py
│   │   │   └── model_binance.py     # Definições dos modelos de banco de dados
│   │   ├── lstm_model.py    # Classe LSTM para previsão
│   │   └── __init__.py
│   │
│   │
│   ├── optmization/           # Otmizadores de parametros
│   │   ├──__init__.py
│   │   └──grid_search.py  # GridSearch para procura dos melhores hiperparametros
│   │
│   │
│   ├── services/           # Serviços de operações principais
│   │   ├──__init__.py
│   │   ├── binance.py   # Serviço para API binance
│   │   ├── metatrader.py  # Serviço para API MetaTrader
│   │   └──yahoofinance.py  # Serviço para API YahooFinance
│   │
│   │
│   ├── scripts/           # Scripts iniciais para treino execução online e busca de hiperparametros
│   │   ├──__init__.py
│   │   ├── evaluate.py   # Avaliador de modelo
│   │   ├── live_run.py   # Execução de modelo online 
│   │   ├── train.py   # treinar modelo 
│   │   └──train_grid.py  # procurar os melhores hiperparametros 
│   │
│   │
│   ├── utils/              # Utilitários e funções auxiliares
│   │   ├── logger.py         # classe para manipular logs 
│   │   ├── csv_exporter.py         # classe para exportar resultados em csv
│   │   ├── csv_to_database.py  # classe para pegar dados de um arquivo csv e passar para um banco de dados
│   │   ├── plotter.py  # Classe para gerar plots em png 
│   │   ├── validation.py  # Funções de validações de tipos de dados
│   │   ├── technical_indicators.py  # Classe geradores de indicadores técnicos financeiros
│   │   └── __init__.py
│   │
│   │
│   │
│   ├── tests/              # Testes automatizados
│   │   ├── test_bots.py           # Testes para os bots de captura
│   │   ├── test_models.py         # Testes para modelos de IA e DB
│   │   ├── test_services.py       # Testes para os serviços
│   │   ├── test_backtest.py       # Testes para a lógica de backtesting
│   │   └── __init__.py
│   │
│   └── .env            # Arquivo de configuração em caso de não tiver duplique o .env.example e preencha ele com seus dados 
│
├── docs/                   # Documentação do projeto
│   ├── README.md           # Introdução e documentação principal
│   ├── API_DOCUMENTATION.md # Documentação da API
│   └── SYSTEM_DESIGN.md    # Arquitetura e design do sistema
│
│       
└── requirements.txt     # Dependências do Python