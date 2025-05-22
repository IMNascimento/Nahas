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


# Arquitetura de Plataforma
nova arquitetura para atender plataforma de treino rapido.

Nahas/
│
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py          # Configurações globais
│   │   └── experiment.yaml      # Configuração default/hierárquica para experimentos
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_preprocessing.py
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   ├── model_base.py        # Gerenciador de conexão e queries MySQL 
│   │   ├── model_nahas.py        # guarda os dados dos experimentos
│   │   └── model_binance.py # Modelos binance guarda cotações
│   │
│   ├── models/
│   │   ├── base/
│   │   │   ├── __init__.py
│   │   │   └── base_trainer.py      # Interface base dos trainers
│   │   ├── keras/
│   │   │   ├── __init__.py
│   │   │   ├── lstm_trainer.py
│   │   │   └── transformer_trainer.py
│   │   ├── pytorch/
│   │   │   ├── __init__.py
│   │   │   ├── lstm_trainer.py
│   │   │   └── transformer_trainer.py
│   │   ├── tensorflow/
│   │   │   ├── __init__.py
│   │   │   ├── lstm_trainer.py
│   │   │   └── transformer_trainer.py
│   │   └── __init__.py
│   │
│   ├── optmization/
│   │   ├── __init__.py
│   │   └── grid_search.py       # Para busca de hiperparâmetros automatizada
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── finetuning_service.py   
│   │   ├── grid_search_service.py   
│   │   ├── live_run_service.py   
│   │   ├── training_service.py       
│   │   ├── binance.py
│   │   ├── metatrader.py
│   │   └── yahoofinance.py
│   │
│   ├── experiment/                  # Gerenciamento de experimentos (core)
│   │   ├── __init__.py
│   │   ├── experiment_manager.py    # Cria, executa, salva e recupera experimentos
│   │   ├── experiment_runner.py     # Executor de experimentos (CLI e Web usam isso)
│   │   └── experiment_schema.py     # Esquemas de configuração dos experimentos
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py                # Sistema de logs robusto
│   │   ├── csv_exporter.py
│   │   ├── csv_to_database.py
│   │   ├── plotter.py
│   │   ├── validation.py
│   │   ├── technical_indicators.py
│   │   └── metrics.py               # Cálculo de métricas customizadas
│   │
│   │
│   ├── web/                         # Interface web local/servidor (Streamlit)
│   │   ├── __init__.py
│   │   ├── app.py                   # Streamlit App principal
│   │   ├── components/              # Componentes customizados Streamlit (ex: cards, tabelas)
│   │   │   └── ...
│   │   └── style/                   # Customização de CSS Streamlit
│   │       └── custom.css
│   │
│   ├── logs/                        # Pasta onde ficam logs locais (arquivos .log rotativos)
│   │   └── (gerado em runtime)
│   │
│   ├── scripts/                     # Scripts CLI de uso geral
│   │   ├── __init__.py
│   │   ├── train.py                 # Treina qualquer modelo (auto detecta backend)
│   │   ├── evaluate.py
│   │   ├── live_run.py
│   │   ├── train_grid.py
│   │   ├── manage_experiments.py    # CLI para consultar/reiniciar/gerenciar experimentos
│   │   └── export_results.py        # Exporta resultados do banco/csv
|   |
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_models.py
│   │   ├── test_services.py
│   │   ├── test_experiment.py
│   │   └── test_utils.py
│   │
│   └── .env
│
├── experiments/                  # Armazena pastas de resultados por experimento
│   └── {id_experimento}/
│       ├── models/               # Modelos salvos
│       ├── csv/                  # Resultados CSV
│       ├── plots/                # Gráficos gerados
│       ├── logs/                 # Logs específicos do experimento
│       └── metrics.json          # Métricas principais do experimento
│
├── docs/
│   ├── README.md
│   ├── API_DOCUMENTATION.md
│   └── SYSTEM_DESIGN.md
│
├── requirements.txt
├── .env.example
└── setup.py                     # (opcional, para empacotar como biblioteca)
