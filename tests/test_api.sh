#!/usr/bin/env bash
set -euo pipefail
BASE_URL="${1:-http://localhost:8000}"
OUTPUT_DIR="${2:-reports/http-smoke}"
mkdir -p "$OUTPUT_DIR"
curl --fail --silent --show-error "$BASE_URL/health" > "$OUTPUT_DIR/health.json"
curl --fail --silent --show-error -X POST "$BASE_URL/predict" \
  -F 'sales=@data/demo_sales.csv' -F 'stock=@data/demo_stock.csv' -o "$OUTPUT_DIR/forecast.csv"
curl --fail --silent --show-error "$BASE_URL/predict/all?format=csv" -o "$OUTPUT_DIR/all.csv"
cmp "$OUTPUT_DIR/forecast.csv" "$OUTPUT_DIR/all.csv"
curl --fail --silent --show-error "$BASE_URL/predict/all?format=json" -o "$OUTPUT_DIR/all.json"
curl --fail --silent --show-error "$BASE_URL/predict?article=char_article_0" -o "$OUTPUT_DIR/article.csv"
status=$(curl --silent --output "$OUTPUT_DIR/error.json" --write-out '%{http_code}' "$BASE_URL/predict")
test "$status" = 400
echo 'HTTP API smoke passed'
