#!/usr/bin/env bash
# MySQL do Nahas no servidor (aiserver), em Docker, com TUDO dentro de uma pasta so,
# para ser removido com um comando quando o servidor voltar a servir o vLLM.
#
#   Pasta:      /mnt/models-ssd/nahas-mysql/   (data/ = dados do MySQL, dumps/ = backups .sql.gz)
#   Container:  nahas-mysql (mysql:8.0), porta 127.0.0.1:3306, banco nocapital, usuario Igor
#   .env:       src/.env com HOST_DB="127.0.0.1" (o antigo, apontando para o notebook, esta em src/.env.bak_notebook)
#
#   sudo bash nahas_mysql_servidor.sh status   # esta rodando? quantas linhas?
#   sudo bash nahas_mysql_servidor.sh stop     # para (dados ficam)
#   sudo bash nahas_mysql_servidor.sh start    # sobe de novo com os mesmos dados
#   sudo bash nahas_mysql_servidor.sh backup   # dumps/nocapital_<data>.sql.gz
#   sudo bash nahas_mysql_servidor.sh restore dumps/nocapital_20261026.sql.gz
#   sudo bash nahas_mysql_servidor.sh remove   # apaga container, imagem e a pasta inteira (pede confirmacao)
set -euo pipefail
PASTA=/mnt/models-ssd/nahas-mysql
NOME=nahas-mysql
SENHA="Igor@1234"
case "${1:-status}" in
  start)
    if docker ps -a --format '{{.Names}}' | grep -qx "$NOME"; then docker start "$NOME"; else
      mkdir -p "$PASTA/data" "$PASTA/dumps"
      docker run -d --name "$NOME" --restart unless-stopped -p 127.0.0.1:3306:3306 \
        -e MYSQL_ROOT_PASSWORD="$SENHA" -e MYSQL_DATABASE=nocapital -e MYSQL_USER=Igor -e MYSQL_PASSWORD="$SENHA" \
        -v "$PASTA/data:/var/lib/mysql" mysql:8.0
    fi ;;
  stop)   docker stop "$NOME" ;;
  status) docker ps -a --format '{{.Names}} {{.Status}} {{.Ports}}' | grep "$NOME" || echo "container $NOME nao existe"
          docker exec "$NOME" mysql -uIgor -p"$SENHA" nocapital -e "SELECT symbol, COUNT(*) n, MAX(timestamp) ultimo FROM price_history GROUP BY 1; SELECT COUNT(*) treinos FROM trainingrun;" 2>/dev/null || true
          du -sh "$PASTA" ;;
  backup) mkdir -p "$PASTA/dumps"; f="$PASTA/dumps/nocapital_$(date +%Y%m%d_%H%M).sql.gz"
          docker exec "$NOME" mysqldump -uroot -p"$SENHA" --single-transaction --quick --databases nocapital 2>/dev/null | gzip > "$f"; ls -la "$f" ;;
  restore) zcat "$2" | docker exec -i "$NOME" mysql -uroot -p"$SENHA" 2>/dev/null; echo "restaurado de $2" ;;
  remove)
    read -r -p "Apagar o container $NOME, a imagem mysql:8.0 e $PASTA (dados e dumps)? [digite SIM] " ok
    [[ "$ok" == "SIM" ]] || { echo "cancelado"; exit 1; }
    docker rm -f "$NOME" 2>/dev/null || true
    docker rmi mysql:8.0 2>/dev/null || true
    rm -rf "$PASTA"
    echo "removido. Lembre de voltar src/.env (HOST_DB) se o banco for ficar em outro lugar." ;;
  *) echo "uso: $0 {start|stop|status|backup|restore <dump.sql.gz>|remove}"; exit 1 ;;
esac
