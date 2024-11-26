project_root/
│
├── src/                    # Diretório principal do código-fonte
│   ├── data/               # Captura, limpeza e gerenciamento de dados
│   │   ├── bots/           # Bots de captura de dados da bolsa
│   │   │   ├── bot_config.py
│   │   │   ├── bot_base.py  # Classe base para bots
│   │   │   └── specific_bot.py  # Bot específico para bolsa de valores
│   │   ├── data_loader.py   # Script para carregar dados
│   │   └── data_preprocessing.py  # Scripts de pré-processamento de dados
│   │
│   ├── models/             # Modelos de IA e banco de dados
│   │   ├── lstm_model.py    # Classe LSTM para previsão
│   │   ├── db_models.py     # Definições dos modelos de banco de dados
│   │   └── __init__.py
│   │
│   ├── services/           # Serviços de operações principais
│   │   ├── prediction_service.py   # Serviço para treinar e prever com IA
│   │   ├── trading_service.py      # Serviço para operações reais e simulação
│   │   ├── backtest_service.py     # Serviço para backtesting
│   │   └── data_service.py         # Serviço de interação com dados
│   │
│   ├── tests/              # Testes automatizados
│   │   ├── test_bots.py           # Testes para os bots de captura
│   │   ├── test_models.py         # Testes para modelos de IA e DB
│   │   ├── test_services.py       # Testes para os serviços
│   │   └── test_backtest.py       # Testes para a lógica de backtesting
│   │
│   ├── utils/              # Utilitários e funções auxiliares
│   │   ├── logger.py         # Configuração e funções de logging
│   │   ├── config.py         # Configurações gerais
│   │   ├── helper_functions.py # Funções auxiliares
│   │   └── __init__.py
│   │
│   ├── api/                # API para o frontend ou integração externa
│   │   ├── endpoints/       # Definição de endpoints
│   │   │   ├── trading_api.py   # Endpoints para operações
│   │   │   ├── data_api.py      # Endpoints de dados
│   │   │   └── model_api.py     # Endpoints para interações com IA
│   │   └── __init__.py
│   │
│   └── main.py             # Ponto de entrada principal do sistema
│
├── docs/                   # Documentação do projeto
│   ├── README.md           # Introdução e documentação principal
│   ├── API_DOCUMENTATION.md # Documentação da API
│   └── SYSTEM_DESIGN.md    # Arquitetura e design do sistema
│
├── config/                 # Configurações específicas do projeto
│   ├── settings.py         # Configurações gerais (e.g., banco de dados)
│   ├── lstm_config.json    # Configurações específicas do modelo LSTM
│   └── bot_settings.json   # Configurações para bots de captura de dados
│
├── requirements.txt        # Dependências do Python
└── Dockerfile              # Dockerfile para configurar o ambiente