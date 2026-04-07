# OTel Tracing Quickstart

This project exports traces via OTLP/HTTP. Defaults:

- `OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318`
- `OTEL_SERVICE_NAME=llm_api`

## Start the server

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \
OTEL_SERVICE_NAME=llm_api \
cargo run --bin http_server
```

## Verify traces locally (collector + sample request)

Use the script below to spin up a local collector and send a request.

```bash
./scripts/verify_otel_trace.sh
```

For a streaming request (so `response.write` appears), set `STREAM=1`:

```bash
STREAM=1 ./scripts/verify_otel_trace.sh
```

## Expected span tree

- `http.request`
  - `llm.chat` (round 1)
  - `tool.calls`
    - `tool.call`
  - `llm.chat` (round 2)
  - `response.write` (streaming only)
