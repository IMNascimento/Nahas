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
