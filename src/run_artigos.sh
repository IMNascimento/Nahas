#!/usr/bin/env bash
# Fila das grades dos artigos no servidor. Rodar dentro de um tmux:
#   tmux new -s artigos "bash run_artigos.sh"
# Cada grade grava results/grid/<grid_id>/ (leaderboard, grid_provenance.json e
# uma pasta por execucao com provenance.json). Logs em logs/<grid_id>.log.
#
# Idempotente: grades com grid_summary.json sao puladas; ao final, toda grade
# com combinacoes falhas (ex.: timeout) e reexecutada so nessas combinacoes
# (run_grid.py --rerun-failed), na mesma pasta.
set -uo pipefail
cd "$(dirname "$0")"
source ../venv/bin/activate
mkdir -p logs
# Limite por treino (worker). Transformer+EvoMSN com 4 jobs em 2 GPUs passa de 2 h.
export NAHAS_TRAIN_TIMEOUT_S=${NAHAS_TRAIN_TIMEOUT_S:-21600}
# Threads de CPU por worker. Sem isso cada um dos 4 workers abre pools com os 20
# nucleos (TF, BLAS, OpenMP): carga 24 em 20 nucleos e pacote a 90 C. O treino em
# si roda na GPU; a CPU so alimenta o laco do Keras e o pre-processamento.
NT=${NAHAS_CPU_THREADS:-5}
export OMP_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT MKL_NUM_THREADS=$NT
export TF_NUM_INTRAOP_THREADS=$NT TF_NUM_INTEROP_THREADS=2
FILA=(
  configs/artigos/artigo1_principal_H1.json
  configs/artigos/artigo1_principal_H6.json
  configs/artigos/artigo1_principal_H24.json
  configs/artigos/artigo2_degenerescencia.json
)
for cfg in "${FILA[@]}"; do
  nome=$(basename "$cfg" .json)
  if [[ -f "results/grid/$nome/grid_summary.json" ]]; then
    echo "[FILA] $nome ja concluido, pulando" | tee -a logs/fila.log
    continue
  fi
  echo "[FILA] $(date -Iseconds) inicio $nome" | tee -a logs/fila.log
  python run_grid.py "$cfg" > "logs/$nome.log" 2>&1
  echo "[FILA] $(date -Iseconds) fim $nome (exit $?)" | tee -a logs/fila.log
done
# Reexecucao das combinacoes que falharam
for cfg in "${FILA[@]}"; do
  nome=$(basename "$cfg" .json)
  [[ -f "results/grid/$nome/grid_summary.json" ]] || continue
  nfalhas=$(python -c "import json,sys; s=json.load(open('results/grid/$nome/grid_summary.json')); print(sum(1 for r in s['summary'] if not r.get('run_id')))")
  if [[ "$nfalhas" -gt 0 ]]; then
    echo "[FILA] $(date -Iseconds) reexecutando $nfalhas falha(s) de $nome" | tee -a logs/fila.log
    python run_grid.py "$cfg" --rerun-failed > "logs/${nome}_rerun_$(date +%Y%m%d_%H%M).log" 2>&1
    echo "[FILA] $(date -Iseconds) fim reexecucao $nome (exit $?)" | tee -a logs/fila.log
  fi
done
echo "[FILA] $(date -Iseconds) FILA CONCLUIDA" | tee -a logs/fila.log
