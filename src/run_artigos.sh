#!/usr/bin/env bash
# Fila das grades dos artigos no servidor. Rodar dentro de um tmux:
#   tmux new -s artigos "bash run_artigos.sh"
#
# A fila e lida de configs/artigos/fila.txt, uma linha por item, e RELIDA a cada
# item: da para acrescentar ou reordenar linhas enquanto roda. Dois tipos de linha:
#   configs/artigos/<grid>.json          grade: python run_grid.py <cfg>; falhas sao
#                                        refeitas em seguida (--rerun-failed)
#   cmd <nome> <comando ...>             comando arbitrario (ex.: run_streaming.py)
# Linhas vazias e iniciadas por # sao ignoradas. Um item esta concluido quando existe
# results/fila_done/<nome>; grades tambem precisam de results/grid/<nome>/grid_summary.json.
# Cada grade grava results/grid/<grid_id>/ (leaderboard, grid_provenance.json e uma
# pasta por execucao com provenance.json). Logs em logs/<nome>.log e logs/fila.log.
set -uo pipefail
cd "$(dirname "$0")"
source ../venv/bin/activate
mkdir -p logs results/fila_done
# Limite por treino (worker). Transformer+EvoMSN com 4 jobs em 2 GPUs passa de 2 h.
export NAHAS_TRAIN_TIMEOUT_S=${NAHAS_TRAIN_TIMEOUT_S:-21600}
# Threads de CPU por worker. Sem isso cada um dos 4 workers abre pools com os 20
# nucleos (TF, BLAS, OpenMP): carga 24 em 20 nucleos e pacote a 90 C.
NT=${NAHAS_CPU_THREADS:-5}
export OMP_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT MKL_NUM_THREADS=$NT
export TF_NUM_INTRAOP_THREADS=$NT TF_NUM_INTEROP_THREADS=2
FILA_TXT=configs/artigos/fila.txt
log() { echo "[FILA] $(date -Iseconds) $*" | tee -a logs/fila.log; }
n_falhas() { python -c "import json; s=json.load(open('results/grid/$1/grid_summary.json')); print(sum(1 for r in s['summary'] if not r.get('run_id')))"; }
# imprime a primeira linha ainda nao concluida da fila (ou nada)
proximo() {
  local l nome
  while IFS= read -r l || [[ -n "$l" ]]; do
    l="${l%%#*}"; l="$(echo "$l" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    [[ -z "$l" ]] && continue
    if [[ "$l" == cmd\ * ]]; then nome=$(awk '{print $2}' <<<"$l"); else nome=$(basename "$l" .json); fi
    [[ -f "results/fila_done/$nome" ]] && continue
    echo "$l"; return 0
  done < "$FILA_TXT"
  return 1
}
while item=$(proximo); do
  if [[ "$item" == cmd\ * ]]; then
    nome=$(awk '{print $2}' <<<"$item"); comando=$(cut -d' ' -f3- <<<"$item")
    log "inicio $nome: $comando"
    bash -c "$comando" > "logs/$nome.log" 2>&1; rc=$?
    log "fim $nome (exit $rc)"
    if [[ $rc -eq 0 ]]; then touch "results/fila_done/$nome"; else log "ERRO em $nome; marcando como concluido para nao travar a fila (reexecute a mao)"; touch "results/fila_done/$nome"; fi
    continue
  fi
  cfg="$item"; nome=$(basename "$cfg" .json)
  if [[ ! -f "$cfg" ]]; then log "config $cfg nao existe; pulando"; touch "results/fila_done/$nome"; continue; fi
  if [[ -f "results/grid/$nome/grid_summary.json" ]]; then
    log "$nome ja tem grid_summary.json"
  else
    log "inicio $nome"
    python run_grid.py "$cfg" > "logs/$nome.log" 2>&1
    log "fim $nome (exit $?)"
  fi
  if [[ -f "results/grid/$nome/grid_summary.json" ]]; then
    nfalhas=$(n_falhas "$nome")
    if [[ "$nfalhas" -gt 0 ]]; then
      log "reexecutando $nfalhas falha(s) de $nome"
      python run_grid.py "$cfg" --rerun-failed > "logs/${nome}_rerun_$(date +%Y%m%d_%H%M).log" 2>&1
      log "fim reexecucao $nome (exit $?)"
    fi
    touch "results/fila_done/$nome"
  else
    log "ERRO: $nome terminou sem grid_summary.json; marcando como concluido para nao travar a fila"
    touch "results/fila_done/$nome"
  fi
done
log "FILA CONCLUIDA (nada pendente em $FILA_TXT)"
