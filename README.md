# yomo-llm-api

OpenAI-compatible HTTP server and Rust crate for chat completions with tool support.

## Quick start

1) Ensure `config.toml` exists in the repo root.
2) Run the server:

```bash
cargo run --bin http_server
```

Enable the mock weather server tool:

```bash
ENABLE_MOCK_GET_WEATHER=1 cargo run --bin http_server
```

Default endpoint: `POST /v1/chat/completions` on port `8000`.

## Test

```bash
cargo test
```

Single integration test case:

```bash
cargo test --test openai_chat -- openai_chat::basic
```

## Example request

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5.4-mini",
    "messages": [{"role": "user", "content": "Ping"}]
  }'
```

## Notes

- Integration fixtures live under `tests/fixtures/openai-chat-cases/`.
- Server reads `config.toml` at runtime.
- Tool merge and test guidance: `docs/tool_merge_and_testing.md`.
