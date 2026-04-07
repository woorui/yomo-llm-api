## Tool merge cases

Server tools come from `ServerToolRegistry::list`. Client tools come from the request body `tools`.
The server merges them before calling the model. The merge behavior is name-based.

Cases:
1) Server only
   - Request has no `tools`
   - Result: all server tools are injected into the request

2) Client only
   - Server registry returns empty list
   - Result: request tools are used as-is

3) Same name in both server and client
   - Result: server tool wins; client tool with the same name is ignored

4) Mixed names (some match, some unique)
   - Result: matching names use server version; unmatched client tools are kept
   - Server-only tools not present in the request are appended

5) No tools anywhere
   - Result: tools remain `null`

Tool execution rules in agent loop:
- If all tool calls are server tools, the server executes them and continues the loop.
- If any client tool call exists, the loop returns the client tool calls to the caller.
- In streaming mode, client tool deltas are pushed immediately; server tool deltas are buffered.

## Verification checklist

- Tool merge rules: server only, client only, same-name collision, mixed names, no tools.
- Tool execution flow: server tools run on the server; client tools return tool_calls to the caller.
- Tool message order: assistant tool_calls message precedes tool result messages.
- Streaming behavior: client tool deltas stream immediately; server tool deltas are not emitted.
- Error mapping: invalid requests return invalid_request_error; provider failures return provider_error.
- Compatibility: requests without tools behave the same as before.
- Usage accumulation: multi-round calls return summed usage in the final response.
- Max calls: loop stops at max_calls and returns a response.

## Start the server

1) Ensure `config.toml` exists in the repo root.
2) Start server:

```bash
cargo run --bin http_server
```

Enable mock weather server tool:

```bash
ENABLE_MOCK_GET_WEATHER=1 cargo run --bin http_server
```

Save server logs to a file (recommended for usage checks):

```bash
RUST_LOG=debug ENABLE_MOCK_GET_WEATHER=1 cargo run --bin http_server > server.log 2>&1
```

Note: do not run integration tests concurrently. The server logs are shared and will interleave.

Default port: `8000`
Default endpoint: `POST /v1/chat/completions`

## Curl test cases

### Basic non-stream request

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5.4-mini",
    "messages": [{"role": "user", "content": "Ping"}]
  }'
```

### Streaming request

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5.4-mini",
    "stream": true,
    "messages": [{"role": "user", "content": "Ping"}]
  }'
```

### Server tool only (mock weather tool)

Start with `ENABLE_MOCK_GET_WEATHER=1`.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5.4-mini",
    "messages": [{"role": "user", "content": "Weather in Beijing And Shanghai"}],
    "tool_choice": "auto"
  }'
```

### Client tool only

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5.4-mini",
    "messages": [{"role": "user", "content": "Weather in Beijing"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather by city",
        "parameters": {
          "type": "object",
          "properties": {"location": {"type": "string"}},
          "required": ["location"]
        }
      }
    }],
    "tool_choice": "auto"
  }'
```

### Mixed tools (server + client)

Start with `ENABLE_MOCK_GET_WEATHER=1`.
The client defines one extra tool. The server tool wins on name collisions.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5.4-mini",
    "messages": [{"role": "user", "content": "Weather in Beijing"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "client_only_tool",
        "description": "Client side tool",
        "parameters": {
          "type": "object",
          "properties": {"foo": {"type": "string"}},
          "required": ["foo"]
        }
      }
    }],
    "tool_choice": "auto"
  }'
```

## Test method summary

1) Start the server with log capture:

```bash
RUST_LOG=debug ENABLE_MOCK_GET_WEATHER=1 cargo run --bin http_server > server.log 2>&1
```

2) Run the curl cases in order (no concurrency).

3) Validate results:

- Non-stream response content is returned.
- Stream response emits deltas and ends with `data: [DONE]`.
- Tool merge behavior matches the rules above.
- Server tool calls execute on server; client tools return tool_calls to caller.
- Tool message order is assistant tool_calls followed by tool messages.
- Errors map to `invalid_request_error` or `provider_error` when expected.

4) Verify usage accumulation (multi-round):

- In `server.log`, locate `llm chat(#N), usage=...` lines.
- Sum per-round `input_tokens`/`output_tokens`/`total_tokens`.
- Compare with the final response `usage` for that request.
