#!/usr/bin/env bash
# Fila das grades dos artigos no servidor. Rodar dentro de um tmux:
#   tmux new -s artigos "bash run_artigos.sh"
# Cada grade grava results/grid/<grid_id>/ (leaderboard, grid_provenance.json e
# uma pasta por execucao com provenance.json). Logs em logs/<grid_id>.log.
set -uo pipefail
cd "$(dirname "$0")"
source ../venv/bin/activate
mkdir -p logs
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
echo "[FILA] $(date -Iseconds) FILA CONCLUIDA" | tee -a logs/fila.log
