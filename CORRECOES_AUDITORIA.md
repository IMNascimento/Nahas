# Correções da auditoria de pré-publicação

Auditoria de 27/09/2026 sobre o repositório, antes de torná-lo público. O relatório
completo, com severidades e evidências, está em `auditoria_codigo_nahas.pdf` (fora do
repositório). Este arquivo registra o que mudou no código.

Suítes: `cd src && ./venv/bin/python tests/run_all.py` (29 testes).

---

## Segurança

**`.gitignore`** passou a cobrir `.env.*` com exceção de `.env.example`. O arquivo
`src/.env.remote-backup`, com credenciais reais de banco, e-mail, MetaTrader e Binance,
estava fora do padrão anterior e seria incluído por um `git add -A`. Um `.gitignore` na
raiz passou a cobrir ambientes virtuais, logs e configurações de editor.

**`services/trainer_factory.py`** troca `exec` mais `eval` por `importlib.import_module`
com `getattr`. O comportamento é o mesmo; a diferença é não montar código a partir de
string em tempo de execução.

---

## Metodologia

**Métrica de validação.** `ModelService.train` passou a prever, inverter e medir também
o conjunto de validação, e o dicionário de métricas ganhou a seção `validation`. Sem ela
a única métrica disponível para a busca de hiperparâmetros era a do teste.

**`grid_search_unified`** aceita `score_on="validation"` e usa esse valor como padrão.
Pontuar no teste continua possível, mas é seleção de modelo no conjunto de relato. A
interface web oferece as três opções, começando na validação, e `mape` entrou na lista
de métricas de ranking.

**MAPE.** `_compute_basic_metrics` passou a calcular MAPE, com piso no denominador para
não explodir quando a série passa por zero. A métrica é reportada na dissertação e não
existia na pipeline.

**Seleção de períodos no multi-escala.** `EvoMSNNormalizer` ganhou `period_scaling`:

- `per_channel` (padrão): padroniza cada canal antes da transformada de Fourier;
- `legacy`: mantém a média de amplitudes sobre o X cru.

Sem a padronização, a média entre canais é dominada pelo de maior escala, de modo que os
períodos escolhidos podem refletir o espectro do volume, e não o do preço. A chave de
configuração correspondente é `normalization.evomsn_period_scaling`.

**Política do canal do alvo em configuração salva.** `ModelService.policy_from_config`
resolve para `legacy` quando a chave `include_target_channel` está ausente, que é o caso
de todo config gravado antes de ela existir. `finetune` e `live_run_once` passaram a usar
essa resolução: com o padrão novo, um modelo antigo receberia um canal a mais do que
espera.

**`finetune` com `evomsn_like`** passou a receber `short_horizon_policy` e
`period_scaling` da configuração, em vez de cair no padrão e divergir do treino.

---

## Robustez

- **Cópia profunda** da configuração base entre combinações do grid: com cópia rasa, os
  dicionários aninhados eram compartilhados e a escrita de uma combinação vazava para as
  seguintes.
- **`run_id` próprio por combinação**, preservando o prefixo do usuário. Antes, todas as
  combinações podiam gravar na mesma pasta, inclusive em paralelo.
- **Impasse do escalonador**: `impasse_no_escalonador` detecta a situação de nada rodando
  e nada admitido. O laço antigo voltava ao início e girava em vazio consumindo um núcleo.
- **Métrica ausente**: `extrair_score` devolve `None` e a combinação é ignorada, em vez de
  levantar `TypeError` no meio do grid.
- **`universal_postprocess`** recusa contagem de timestamps diferente do número de
  previsões, em vez de cortar pelo fim e desalinhar a série em silêncio.
- **Conexões de banco em multiprocesso** passaram a registrar a falha ao fechar, em vez de
  engolir a exceção.

---

## Remoções

**`src/optimization/grid_search.py`** foi removido. A classe não era usada pelo fluxo
atual e tinha dois defeitos silenciosos: atribuía `data_processor.window_size`, que não é
propriedade da classe, de modo que a varredura de janela não tinha efeito; e inicializava
o melhor score como menos infinito sempre que a métrica não se chamasse exatamente
`loss`, escolhendo a pior configuração como melhor.

---

## Dependências

`requirements.txt` passou a listar apenas o que o código importa. O congelamento completo
do ambiente dos experimentos virou `requirements-full.txt`. Ferramentas de
desenvolvimento estão em `requirements-dev.txt`, e `torch` e `MetaTrader5`, que não
estavam instalados no ambiente dos experimentos, em `requirements-optional.txt`.

---

## Testes e CI

- `src/tests/test_auditoria_fixes.py`: 15 testes, um por correção acima.
- `src/tests/test_pipeline_integracao.py`: 3 testes que rodam a pipeline inteira nas
  quatro estratégias de normalização, com o banco substituído por dublês.
- `src/tests/run_all.py`: roda as três suítes em sequência.
- `.github/workflows/ci.yml`: lint, testes e auditoria de dependências a cada push.

### Divergência de `source` entre o banco e o sync

Encontrado durante o pré-teste, no banco real. As linhas gravadas tinham
`source = 'binance'` e o `sync_binance_price_history.py` usa `SOURCE_DEFAULT =
'binance_api'`. Como `source` entra na chave única e no filtro de
`_last_timestamp_utc`, o sync concluía que não havia histórico, baixava tudo desde 2015 e
gravava uma **segunda cópia completa** da série. Do outro lado, um treino configurado com
o `source` que não existe morria com "Nenhum dado encontrado", mensagem que culpa os
dados e não o filtro.

- `ModelService.mensagem_sem_dados` lista as combinações de `source` e `exchange` que
  existem para aquele símbolo e intervalo, com a contagem de linhas.
- `outras_fontes` no módulo de sincronização avisa, antes de baixar, que a série já existe
  sob outro `source` e que a execução vai duplicar em vez de completar.

### Política do canal do alvo gravada no config do run

`train` e `finetune` passaram a escrever `include_target_channel` no config salvo. Sem
isso, um run feito hoje com o padrão `equalized` salvava um config sem a chave, e a
inferência o leria como `legacy`, mudando o número de canais da entrada.

---

## Validação com dados reais

Pré-teste de 27/09/2026 com o histórico completo da Binance carregado em MySQL local:
BTCUSDT e ETHUSDT, candles de 1 hora, 79.767 por ativo, de 2017-08-17 a 2026-09-28,
baixados pelo próprio `sync_binance_price_history.py`. Treino em GPU com os
hiperparâmetros do apêndice da dissertação (L = 96, H = 1, LSTM [128, 64], tanh,
dropout 0,2, lote 32, Adam 1e-3, semente 1482563973) e 10 épocas, número reduzido de
propósito: o objetivo é conferir corretude, não reproduzir o valor final.

### Estratégias de normalização

| caso | canais | RMSE validação | RMSE teste | MAPE teste | excesso sobre persistência |
|---|---|---|---|---|---|
| global MinMax | 5 | 1216,14 | 3100,20 | 2,396% | +890,05% |
| local MinMax | 5 | 127,12 | 342,77 | 0,385% | +9,46% |
| multi-escala corrigido | 5 | 125,11 | 336,14 | 0,381% | +7,34% |
| multi-escala legado | 5 | 240,50 | 654,08 | 0,837% | +108,88% |
| multi-escala "like" | 5 | 125,11 | 336,14 | 0,381% | +7,34% |

Leituras:

- **A pipeline continua reproduzindo o comportamento publicado.** Com 10 épocas, o local
  MinMax dá 342,77 contra 333,77 do trabalho (500 épocas), e o global MinMax dá 3100
  contra 3431. A razão global sobre local fica em 9,0, contra 10,3 no texto.
- **Determinismo confirmado.** Duas execuções independentes da mesma configuração
  devolveram RMSE idêntico até a quarta casa decimal.
- **As correções da família multi-escala valem 49% de erro.** De 654,08 no comportamento
  legado para 336,14 com dispersão pela janela do alvo e seleção de períodos padronizada
  por canal. Isso confirma, com medida, a hipótese registrada na dissertação: o
  desempenho fraco daquela família era consequência do horizonte unitário, e não
  propriedade do método multi-escala.
- **EvoMSN e EvoMSN-like coincidem exatamente sob H = 1 com `window_stats`.** Não é erro:
  com todas as escalas degeneradas, a referência vem da janela do alvo em todas elas e os
  preditores auxiliares, única diferença entre as duas classes, nunca são chamados. Sob
  `legacy` as duas voltam a divergir. Coberto por teste.

### Matriz do confundimento

Quatro braços, mesma semente, mesmas épocas, por `experiments/run_confound_matrix.py`:

| braço | canais | alvo em X | RMSE teste | excesso sobre persistência |
|---|---|---|---|---|
| A, global legado | 4 | não | 4701,59 | +1401,45% |
| B, global equalizado | 5 | sim | 3100,20 | +890,05% |
| C, local equalizado | 5 | sim | 342,77 | +9,46% |
| D, local sem o alvo | 4 | não | 401,31 | +28,16% |

O canal extra de feature importa, e pouco: dar o fechamento ao braço global reduz o erro
dele em 34%, e tirá-lo do braço local piora 17%. Com features iguais dos dois lados, a
vantagem da normalização por janela permanece em 9,0 vezes (5 canais) e 11,7 vezes
(4 canais), contra 10,3 vezes na comparação publicada, que ficava entre as duas. A
conclusão central da dissertação sobrevive à comparação controlada.

Ressalva: uma semente, 10 épocas, um único ativo. Serve para decidir que o efeito não
inverte o resultado, não para substituir a tabela do trabalho.

### Seleção de hiperparâmetros na validação

Grid de duas combinações (taxa de aprendizado 1e-3 e 5e-4) com `score_on` no padrão novo:

```
{'LEARNING_RATE': 0.001}  -> rmse@validation = 136,68
{'LEARNING_RATE': 0.0005} -> rmse@validation = 132,38   <- escolhida
```

O `leaderboard.csv` registra `score_on = validation` em todas as linhas. O conjunto de
teste não participou da escolha.

### Inferência ao vivo

`live_run_once` executado com o config salvo pelo treino e, em seguida, com o mesmo config
sem a chave `include_target_channel`, simulando um run anterior à correção. Nos dois casos
a inferência montou a janela com o número de canais correto e rodou até a previsão.

---

## Segunda rodada: achados de 28/09

### `.env.example`: verificação de credenciais — nada vazou

`src/.env.example` é versionado desde 2023. Uma primeira triagem automática, por entropia
e por lista de palavras-guia (`seu`, `sua`, `your`, `example`, `xxx`), marcou vários campos
como possíveis credenciais reais e chegou a recomendar rotação de senhas. **Essa conclusão
estava errada.**

A inspeção direta dos valores, nas 15 versões históricas de `*.env.example` (incluindo as
cópias em `old/`, `src_old/`, `teste1/` e `teste2/`), mostrou que os campos sensíveis
continham placeholders escritos em português — `name banco`, `username banco`,
`passowrd banco`, `host banco`, `email usuario`, `senha email`. Não casavam com o padrão
de placeholder esperado, mas são descrições, não segredos.

Valores genuínos presentes:

| campo | valor | natureza |
|---|---|---|
| `EMAIL_SMTP`, `EMAIL_IMAP` | `smtp.gmail.com`, `imap.gmail.com` | hosts públicos de provedor |
| `EMAIL_PORT` | `587` | porta padrão |
| `SERVER_MT5` | hostname público da corretora | presente em 10 versões |

`LOGIN_MT5`, `PASSWORD_MT5`, `BINANCE_API_KEY` e `BINANCE_API_SECRET_KEY` foram
placeholders em todas as versões: nada capaz de movimentar dinheiro ou autenticar em conta
alguma entrou no repositório em três anos.

**Conclusão: não houve vazamento de credencial e não há nada a rotacionar.**

O arquivo foi sanitizado de todo modo, por dois motivos independentes do incidente que não
houve: placeholders descritivos podem ser confundidos com valores reais — foi exatamente o
que aconteceu nesta auditoria — e o hostname da corretora é divulgação desnecessária num
repositório público. Os 15 campos passaram a placeholders inequívocos no formato
`<senha_do_banco>`, com cabeçalho avisando que o arquivo é versionado. `HOST_DB`,
`PORT_DB`, `EMAIL_PORT` e os hosts de provedor ficaram com valores de exemplo neutros.

A observação da seção **Segurança** sobre `src/.env.remote-backup` permanece válida: esse
arquivo, que nunca foi versionado, contém credenciais reais e passou a ser coberto pelo
`.gitignore`.

### Vazamento temporal no preenchimento de indicadores

`utils/technical_indicators.py` preenchia valores ausentes com `data[col].mean()` e
`data[col].min()`, estatísticas da **série inteira**. Como o preenchimento ocorre antes da
divisão temporal, as primeiras linhas de cada indicador — o aquecimento da janela móvel,
que cai no conjunto de treino — recebiam um valor calculado a partir do futuro, teste
incluído. É a mesma classe de erro que este trabalho investiga.

Não afetou nenhum resultado publicado: `indicators_apply` está vazio em todas as
execuções do trabalho, e as colunas de preço não tinham ausências. O defeito estava no
código que seria publicado.

Regra atual: preço e volume usam forward fill, que só olha para trás; colunas de
indicador mantêm os NaN de aquecimento e são descartadas pelo `dropna()` do chamador.
Qualquer preenchimento ali seria look-ahead — backward fill copia o futuro, e média ou
mínimo globais idem. Coberto por teste de causalidade: alterar a metade final da série
não altera nenhum valor da metade inicial.

### Rótulo enganoso no baseline de saída constante

`utils/baselines.py` reportava `constant_zero_equals_persistence`, sugerindo que prever
zero no espaço normalizado equivale à persistência. A equivalência vale **apenas** para
`y_mode='relative_last'`, que divide pelo último valor observado. Em `minmax_target` — o
modo usado na matriz do confundimento — o zero corresponde ao **mínimo da janela**, e em
`zscore_target`, à média: baselines diferentes e piores. Em pré-teste com dados reais a
diferença chegou a 12.528 em preço, com RMSE de 3.115 reportado sob um nome que insinuava
persistência.

O relatório passa a trazer `equals_persistence` (booleano, com tolerância relativa à
escala do alvo) e um campo `meaning` dizendo o que o zero significa naquele modo. Coberto
por teste nos três modos.

## O que permanece em aberto

1. **Matriz do confundimento com mais sementes e no ETH.** A execução acima usou uma
   semente e um ativo. O efeito medido é pequeno frente à diferença entre famílias, mas o
   número que entrar em publicação deveria vir de pelo menos três sementes nos dois
   ativos.
2. **Nova busca de hiperparâmetros** pontuando na validação, com orçamento de épocas
   completo, para publicar uma escolha que não passou pelo teste.
3. **Reexecução completa da família multi-escala** com as correções, em 500 épocas, para
   substituir os números dessa família nas tabelas do trabalho.
4. **Decidir o que fazer com a divergência entre o texto e o código** quanto à busca
   preliminar de hiperparâmetros: a dissertação diz que foi feita apenas na validação, e a
   ferramenta publicada só passou a permitir isso agora.
