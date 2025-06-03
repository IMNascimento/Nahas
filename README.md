# Nahas

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Version](https://img.shields.io/badge/version-1.0.0-blue)

## Introdução

Nahas é uma ferramenta [descreva a principal funcionalidade ou objetivo do projeto] que oferece [benefícios principais]. Desenvolvido como um projeto open source, nosso objetivo é [explicar o objetivo principal do projeto].

## Funcionalidades

- Funcionalidade 1
- Funcionalidade 2
- Funcionalidade 3
- [Adicione outras funcionalidades importantes]

## Pré-requisitos

Antes de começar, certifique-se de ter as seguintes ferramentas instaladas:

- [Python] versão 3.10.17
- cuda 12.1
- Pytorch v 2.5.1
- tensorflow v 2.15

## Instalação

Siga as etapas abaixo para configurar o projeto em sua máquina local:

1. Clone o repositório:
    ```bash
    git clone https://github.com/SophiaMind/Nahas.git
    ```
2. Navegue até o diretório do projeto:
    ```bash
    cd Nahas\src\
    ```
3. Crie e ative o ambiente virtual:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Para Linux/MacOS
    .\venv\Scripts\activate  # Para Windows
    ```
4. Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

## Uso

Após a instalação, você pode iniciar a aplicação com o seguinte comando:

```bash
python3 scripts/train.py
```

Não se esqueça de criar o .env existe um arquivo inicial para ser copiado que é o env.example.

## Exemplos de Uso
```python
# Exemplo de código mostrando como usar a funcionalidade principal do projeto
```

## Contribuindo

Contribuições são bem-vindas! Por favor, siga as diretrizes em CONTRIBUTING.md para fazer um pull request.

## Licença

Distribuído sob a licença GNU 3. Veja LICENSE para mais informações.

## Autores

Igor Muniz Nascimento - Desenvolvedor Principal - [GitHub](https://github.com/IMNascimento)

## Agradecimentos
[Recursos ou bibliotecas que você usou]
[Qualquer outra pessoa ou organização que você queira mencionar]








atualizações 
o arquivo scripts/train já está pronto so falta testar ele pega os hiperparametros do settings se não passar nada mas
podemos passar parametros por args[]

lstm_model.py está com todos os parametros customizavel e sendo realizados por passagem até as funções de ativação. Só falta testar agora

technical indicators gera todos indicadores no dataframe seguindo os hiperparametros do settings que carrega. E ele so passa para o modelo as colunas que foram filtradas.

settings esta capturando os arquivos de chave do .env e os arquivos de configuração do modelo 

o arquivo scripts/train_grid.py está configurado para le os hiperparameter_grid e ele aceita fazer o gridsearch com a passagem de colunas de indicadores financeiros
ele só faz grid utilizando multi gpu precisa de 2 gpus testar para ve se vai funcionar normalmente.

evaluate.py ele carrega no hiperparametro o nome do modelo e tem que ficar esperto pois a passagem da janela para a normalização do dado tem que ser a mesma usada no modelo então checar isso no hiperparametros também e testar
 

