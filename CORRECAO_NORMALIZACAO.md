# Correção do confundimento entre estratégias de normalização

## Resumo

A comparação entre as estratégias de normalização (`global`, `local`, `evomsn`)
não era controlada: **os braços do experimento recebiam conjuntos de features
diferentes**. O braço `local` recebia a própria série do alvo como canal extra
de `X`; `global` e `evomsn`, não.

Como o canal do alvo é exatamente o que torna a persistência trivialmente
expressável, a diferença reportada entre normalizações mistura dois efeitos:

1. o efeito da normalização (o que se pretendia medir);
2. o efeito de ter `close[t-96..t-1]` como feature (o que não se pretendia).

Esta branch corrige isso, mantendo um caminho explícito para reproduzir os
resultados antigos.

---

## O que estava acontecendo

Com o config usado nos experimentos — `relevant_columns = ['close','open','high','low','volume']`,
`target_column = 'close'` — a coluna alvo é removida das features em
`DataProcessor.create_windows`:

```python
features = data.drop(columns=[coluna_alvo]).values   # X = [open, high, low, volume]
```

E, em `ModelService.train`, o alvo voltava para `X` **apenas** em um dos braços:

```python
if norm_strategy == "local" and self._y_mode_needs_target_in_X(y_mode):
    X_all = np.concatenate([X_all, tw_all], axis=2)   # só aqui
```

| Estratégia | Canais em `X` | Tem `close`? |
|---|---|---|
| `global` | open, high, low, volume | não |
| `local` (`relative_last`, `*_target`) | open, high, low, volume, **close** | **sim** |
| `evomsn` | open, high, low, volume | não |

### O mecanismo

Em `y_mode='relative_last'`:

```python
ref = X[:, -1, [target_idx]]          # close[t-1]
yn  = (y - ref) / (np.abs(ref) + eps)
```

No espaço normalizado, **prever `yn = 0` é identicamente a persistência**. A rede
não precisa aprender a copiar a última observação: basta emitir uma constante
próxima de zero. Isso explica R² ≈ 0,999 com erro no piso da persistência, e
explica por que arquiteturas que aprendem saídas quase constantes com mais
facilidade chegam mais perto desse piso.

O teste `test_zero_normalizado_e_exatamente_persistencia` verifica a identidade
numericamente (diferença máxima 0,0).

### O que NÃO estava errado

Auditado e correto — o efeito observado não é artefato de indexação:

- `create_windows`: para a amostra *j*, `X` cobre `[j, j+ws-1]` e `y = target[j+ws]`.
- `_build_target_windows`: termina em `target[j+ws-1]`, a barra anterior ao alvo.
- Scaler global ajustado **apenas no treino**, reaplicado em validação/teste.

---

## O que mudou

### `src/data/data_processing.py`

- **`split_indices(n_total, train_size, validation_size, embargo)`** — novo.
  Retorna as fatias temporais, aplicáveis identicamente a `X`, `y`, timestamps e
  janelas do alvo. `split_data` passa a usá-lo e aceita `embargo`.
  O split ocorre depois do janelamento, então as últimas janelas de treino
  compartilhavam até `window_size - 1` barras com as primeiras de validação.
- **`normalize_local(..., target_windows=...)`** — as estatísticas de `y` podem
  vir de um array externo em vez de um canal de `X`. Isso desacopla *o que
  normaliza `y`* de *o que o modelo enxerga*, que é o que permite comparar
  normalizações com as mesmas features. Sem o argumento, o comportamento antigo
  é preservado.

### `src/services/model_service.py`

- **`include_target_channel`** — nova chave de config:
  - `"equalized"` (padrão): inclui o canal do alvo em **todas** as estratégias;
  - `"never"`: não inclui em nenhuma;
  - `"legacy"`: comportamento anterior, para reproduzir os resultados publicados.
- **`embargo`** — nova chave. Padrão `window_size + steps_ahead - 1`
  (`0` quando `include_target_channel="legacy"`).
- Normalização local passa a receber `target_windows` nos três caminhos
  (treino, fine-tuning e inferência).
- **Bugfix no fine-tuning**: usava `target_idx=0` sem o alvo em `X`, ou seja,
  normalizava o alvo pela janela de **`open`** em vez de `close`. Modelos
  fine-tunados usavam referência de normalização diferente do modelo de origem.
- Inferência (`live_run_once`) segue a mesma política do treino — se divergir,
  o shape de entrada não bate com o modelo salvo.

### `src/utils/baselines.py` (novo)

- `persistence_forecast`, `causal_moving_average`
- `constant_output_forecast` — o que o modelo preveria emitindo constante no
  espaço normalizado, passando pela mesma inversão
- `mimicry_diagnostics` — compara **retorno previsto x retorno realizado**;
  `corr_returns ≈ 0` indica emulador de persistência, sem sinal
- `diebold_mariano` — variância de longo prazo por Newey-West, correção de
  Harvey-Leybourne-Newbold
- `evaluate_against_baselines` — junta tudo

### `src/tests/test_normalization_fixes.py` (novo)

```bash
cd src && ./venv/bin/python tests/test_normalization_fixes.py
```

---

## Matriz de experimentos a rodar

| # | `include_target_channel` | Estratégias | O que isola |
|---|---|---|---|
| 1 | `legacy` | todas | reproduz os números atuais da dissertação |
| 2 | `equalized` | global, local, evomsn | **efeito da normalização** com features iguais |
| 3 | `never` | global, local, evomsn | normalização sem dar o alvo ao modelo |

A comparação 1 × 2 quantifica quanto da vantagem reportada da normalização local
vinha do canal extra de feature, e não da normalização.

Em todos os casos, reportar junto: RMSE da persistência, RMSE da média móvel
causal, estatística DM e `corr_returns`.

## Reprodutibilidade

Resultados anteriores continuam reproduzíveis:

```json
{ "include_target_channel": "legacy", "embargo": 0 }
```

---

# Correção da degenerescência multi-escala sob horizonte unitário

A variante multi-escala (`evomsn` / `evomsn_like`) não estava apenas imprecisa:
**o backbone era inoperante.** A Seção 5.3 da dissertação já documenta o
diagnóstico; esta parte corrige a implementação.

## O mecanismo

As estatísticas ϕ, ξ do alvo são calculadas por fatia de `y`. Com `H = 1`, o
preenchimento replica o único valor até completar o período, então a fatia é um
vetor constante:

- `xi_true = std(vetor constante) = 0` exatamente;
- o alvo do backbone, `(Ys - phi)/(xi + eps)`, é **identicamente zero**;
- o regressor de ξ̂ é ajustado sobre zeros e devolve ~1e-14;
- em `y = ỹ·(ξ̂ + ε) + ϕ̂`, a saída da rede é multiplicada por ~ε e some.

Medido no repositório (`H=1`, `L=96`, períodos `[96, 48, 24]`):

| | `legacy` | `window_stats` |
|---|---|---|
| desvio padrão do alvo do backbone | 1,3e-06 | 1,26 |
| ξ̂ mediano | 5,1e-15 | 2,19 |
| efeito de somar 1 à saída da rede | 1,0e-08 | 2,63 |
| (desvio padrão do alvo real) | 5,19 | 5,19 |

Em `legacy`, mudar a saída da rede em um desvio padrão altera a previsão em
1e-08, contra um alvo de escala 5,19 — oito ordens de grandeza abaixo. A previsão
é ϕ̂ puro. É por isso que LSTM e Transformer davam resultados idênticos.

## `short_horizon_policy`

Nova opção de `EvoMSNNormalizer`, exposta em
`normalization.evomsn_short_horizon_policy`:

- **`"window_stats"` (padrão)** — nas escalas em que `H < p`, usa as estatísticas
  da última fatia da janela do alvo como referência de normalização. São causais,
  bem definidas e computáveis na inferência, o que dispensa o preditor
  estatístico nessas escalas e devolve à rede sua contribuição. **Afasta-se de
  (QIN et al., 2024), que pressupõe `H ≥ p`** — deve ser descrito como adaptação,
  não como reprodução.
- **`"legacy"`** — mantém o comportamento degenerado, com aviso em log.
  Reproduz os números publicados.
- **`"error"`** — recusa a execução, para não rodar em silêncio um regime inválido.

Quando `H ≥ p` nenhuma escala degenera e as políticas coincidem exatamente
(verificado em teste).

## Outras correções

- `EvoMSNNormalizer.fit(..., target_windows=...)` — as estatísticas do alvo vêm
  da janela do alvo, não de um canal arbitrário de `X`.
- `degeneracy_report` — exposto após o `fit` e registrado em log: períodos,
  quais escalas degeneram, política ativa e desvio padrão do alvo por escala.
- **Bugfix em `EvoMSNLikeNormalizer`**: `_predict_future_stats` fazia média sobre
  **todos os canais**, misturando volume (~1e0) com preço (~1e4). Passa a usar o
  canal do alvo.
- `EvoMSNMeta` ganhou campos com default e `load()` tolera metas antigos.

## O que isso significa para a dissertação

A Seção 5.3 continua correta e não precisa ser reescrita — ela descreve o
comportamento da implementação avaliada. O que muda é que agora existe uma
correção, e vale acrescentar que a família multi-escala pode ser reavaliada sob
`window_stats`, separando "o método não serve para horizonte curto" de "a
implementação anulava o backbone". São afirmações diferentes, e a segunda é a que
os números atuais sustentam.

---

# Como rodar a matriz do confundimento

```bash
cd src
# 1) confira os configs sem tocar no banco
./venv/bin/python experiments/run_confound_matrix.py \
    --base-config results/train/b0d13639/hiperparams/config_BTCUSDT_1h_USDT_BIN_b0d13639.json \
    --dry-run

# 2) rode (exige banco acessivel)
./venv/bin/python experiments/run_confound_matrix.py \
    --base-config results/train/b0d13639/hiperparams/config_BTCUSDT_1h_USDT_BIN_b0d13639.json \
    --framework keras --model-type lstm --seeds 1
```

Saída: tabela com RMSE, R², RMSE da persistência, excesso percentual, estatística
DM e `corr_returns` por braço, mais a leitura direta de **quanto da distância
entre global e local é fechada apenas por acrescentar o canal do alvo ao global**.

Todo treino passa a reportar os baselines automaticamente — em log e no
`metrics.json`, sob as chaves `baselines` e `setup`.
