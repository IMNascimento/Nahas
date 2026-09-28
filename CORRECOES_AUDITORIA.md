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

---

## O que permanece em aberto

1. **Matriz do confundimento** (`src/experiments/run_confound_matrix.py`): quantifica
   quanto da vantagem da normalização por janela vinha do canal extra de feature. Precisa
   de banco e de GPU.
2. **Nova busca de hiperparâmetros** pontuando na validação, para publicar uma escolha
   que não passou pelo teste.
3. **Reexecução da família multi-escala** com `period_scaling="per_channel"`, para medir o
   efeito da correção na seleção de períodos.
