# Agent Guidelines

This repo is a Rust crate named `llm_api` with an HTTP server binary and
fixture-driven tests. Follow existing patterns and keep changes minimal and
idiomatic.

If Cursor/Copilot rules exist, they should be followed, but none were found in
`.cursor/rules/`, `.cursorrules`, or `.github/copilot-instructions.md`.

## Build / Lint / Test Commands

### Build
- `cargo build` (debug)
- `cargo build --release`
- `cargo run --bin http_server` (runs the HTTP server)
- `cargo check`

### Format
- `cargo fmt` (format all Rust files)
- `cargo fmt -- --check` (CI-friendly check)

### Lint
- `cargo clippy` (all targets)
- `cargo clippy --all-targets --all-features -D warnings` (strict)

### Tests
- `cargo test` (all unit + integration tests)
- `cargo test --test openai_chat` (integration test harness)
- `cargo test trimmed_base_url` (single unit test by name)

### Run a Single Integration Test Case
The integration test uses `libtest-mimic` and exposes cases as
`openai_chat::<case-name>`.
- `cargo test --test openai_chat -- openai_chat::basic` (example)
- `cargo test --test openai_chat -- openai_chat::` (filter by prefix)

### Notes
- Test fixtures live under `tests/fixtures/openai-chat-cases/`.
- The HTTP server reads `config.toml` at runtime.

## Code Style Guidelines

### Project Structure
- Library modules are under `src/` and re-exported in `src/lib.rs`.
- The HTTP server binary is `src/bin/http_server.rs`.
- Integration tests use `libtest-mimic` in `tests/openai_chat.rs`.

### Formatting
- Use `rustfmt` defaults; do not hand-format.
- Keep line width reasonable; allow rustfmt to wrap.

### Imports
- Group standard library, external crates, then crate modules.
- Keep `use` lists small and explicit; avoid glob imports.
- Preserve existing ordering patterns when editing a file.

### Naming
- Types: `UpperCamelCase` (structs/enums).
- Functions/variables: `snake_case`.
- Constants: `SCREAMING_SNAKE_CASE`.
- Prefer descriptive names over abbreviations.

### Types and Serde
- Use `serde` attributes for JSON shape (`rename`, `rename_all`, `skip_serializing_if`).
- Use `Option<T>` for optional fields; omit with `skip_serializing_if`.
- Preserve public API shapes in `openai_types` and `provider`.

### Errors and Result Handling
- Use typed errors (`ConfigError`, `ClientError`, `ChatError`) for library APIs.
- Implement `Display` and `Error` for custom errors.
- Use `anyhow::Error` only at binary boundaries (server entrypoints).
- Add contextual error messages with `anyhow::Context` for I/O/serialization.
- Map errors to OpenAI-compatible responses in the server layer only.

### Async and Streams
- Prefer `async fn` for I/O-bound code and `try_stream!` for SSE streaming.
- Treat stream parsing as incremental; buffer by line as in `openai_client`.
- Avoid blocking calls inside async functions.

### Logging
- Use `log` macros (`info!`, `error!`) in the server binary only.
- Keep logs concise and include model/request context where available.

### HTTP / API Conventions
- OpenAI-compatible endpoint is `/v1/chat/completions`.
- Validate request shape early and return `invalid_request_error` for user input.
- For non-2xx upstream responses, parse OpenAI error bodies when possible.

### Data Mapping
- Keep mapping functions (`map_openai_*`) pure and deterministic.
- Preserve OpenAI response fields, including `usage` and finish reason mapping.
- Avoid changing serialized field names without updating serde attributes.

### Tests
- Add new fixtures under `tests/fixtures/openai-chat-cases/<case>/`.
- Each case should include `request.json` and `response.txt`.
- Stream cases are represented as SSE `data:` lines in `response.txt`.

### Config / Secrets
- `config.toml` can include provider API keys; do not commit real secrets.
- Prefer environment-based overrides when possible.

### General Practices
- Keep changes small and localized.
- Avoid introducing new dependencies unless necessary.
- Add comments only for non-obvious logic.
- Preserve public API stability unless explicitly requested.
