#!/usr/bin/env bash
set -euo pipefail

PORT=${PORT:-8000}
OTEL_ENDPOINT=${OTEL_EXPORTER_OTLP_ENDPOINT:-http://localhost:4318}
STREAM=${STREAM:-0}
MODEL=${MODEL:-gpt-5.4-mini}

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required"
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required"
  exit 1
fi

tmp_dir=$(mktemp -d)
cleanup() {
  docker rm -f llm-bridge-otel >/dev/null 2>&1 || true
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

cat >"${tmp_dir}/otel-collector.yaml" <<'EOF'
receivers:
  otlp:
    protocols:
      http:
        endpoint: 0.0.0.0:4318
exporters:
  debug:
    verbosity: detailed
service:
  pipelines:
    traces:
      receivers: [otlp]
      exporters: [debug]
EOF

docker run --rm -d --name llm-bridge-otel \
  -p 4318:4318 \
  -v "${tmp_dir}/otel-collector.yaml":/etc/otelcol/config.yaml \
  otel/opentelemetry-collector:latest \
  --config /etc/otelcol/config.yaml

sleep 2

if [ "${STREAM}" = "1" ]; then
  payload='{"model":"'"${MODEL}"'","stream":true,"messages":[{"role":"user","content":"ping"}]}'
else
  payload='{"model":"'"${MODEL}"'","messages":[{"role":"user","content":"ping"}]}'
fi

echo "Sending request to http://localhost:${PORT}/v1/chat/completions"
echo "OTEL_EXPORTER_OTLP_ENDPOINT=${OTEL_ENDPOINT}"
if ! curl -sS -N -X POST "http://localhost:${PORT}/v1/chat/completions" \
  -H "content-type: application/json" \
  -d "${payload}" >/dev/null; then
  echo "request failed; confirm server is running"
fi

sleep 6
echo "Collector output:"
docker logs llm-bridge-otel || true
