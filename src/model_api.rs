use std::{collections::HashMap, io, sync::Arc, time::Duration};

use axum::{
    body::{Body, Bytes},
    http::{HeaderMap, HeaderName, Request, Response, StatusCode},
    response::IntoResponse,
    Json,
};
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{error, info, warn, Span, field};

use crate::config::ProviderConfig;
use crate::endpoint::Endpoint;
use crate::provider_registry::{ByModel, SelectionError, SelectionResult, SelectionStrategy};

pub const SUPPORTED_ENDPOINT_PATHS: [&str; 9] = crate::endpoint::SUPPORTED_ENDPOINT_PATHS;

const MODEL_API_TIMEOUT_SECS: u64 = 300;

#[derive(Clone)]
pub struct ModelApi {
    client: Client,
    bindings_by_endpoint: HashMap<Endpoint, Vec<EndpointBinding>>,
    usage_handler: Arc<dyn UsageHandler + Send + Sync>,
    selection_strategy: Arc<dyn SelectionStrategy<()>>,
}

#[derive(Clone, Debug)]
struct EndpointBinding {
    endpoint: Endpoint,
    target_url: String,
    api_key: Option<String>,
    model_id: String,
}

impl ModelApi {
    pub fn new(
        providers: &HashMap<String, ProviderConfig>,
    ) -> Result<Self, String> {
        let client = Client::builder()
            .timeout(Duration::from_secs(MODEL_API_TIMEOUT_SECS))
            .build()
            .map_err(|e| format!("build reqwest client failed: {e}"))?;

        let mut bindings_by_endpoint: HashMap<Endpoint, Vec<EndpointBinding>> = HashMap::new();

        for provider in providers.values() {
            let base_url = provider
                .params
                .get("base_url")
                .ok_or_else(|| format!("provider {} missing base_url", provider.id))?;
            let api_key = provider.params.get("api_key").cloned();
            for endpoint_key in &provider.endpoints {
                let endpoint = Endpoint::from_key(endpoint_key)
                    .ok_or_else(|| format!("unknown endpoint key: {endpoint_key}"))?;
                let endpoint_path = endpoint.path();
                let target_url = build_target_url(base_url, endpoint_path)?;
                let bindings = bindings_by_endpoint.entry(endpoint).or_default();
                bindings.push(EndpointBinding {
                    endpoint,
                    target_url,
                    api_key: api_key.clone(),
                    model_id: provider.model_id.clone(),
                });
            }
        }

        let selection_strategy = Arc::new(ByModel::new(providers.clone()));

        Ok(Self {
            client,
            bindings_by_endpoint,
            usage_handler: Arc::new(LogUsageHandler),
            selection_strategy,
        })
    }

    pub async fn handle(&self, req: Request<Body>) -> Response<Body> {
        let req_path = req.uri().path().to_string();
        let method = req.method().clone();
        let incoming_headers = req.headers().clone();
        let span = tracing::info_span!(
            "http.request",
            http.method = %method,
            http.route = %req_path,
            http.status_code = field::Empty,
            model = field::Empty,
            usage = field::Empty,
            error_message = field::Empty,
        );
        let _guard = span.enter();

        let body = match axum::body::to_bytes(req.into_body(), usize::MAX).await {
            Ok(b) => b,
            Err(err) => {
                error!(path = req_path, "read request body failed: {err}");
                span.record("http.status_code", StatusCode::BAD_REQUEST.as_u16() as i64);
                span.record("error_message", "failed to read request body");
                return error_response(StatusCode::BAD_REQUEST, "failed to read request body");
            }
        };

        let endpoint = match Endpoint::from_path(&req_path) {
            Some(endpoint) => endpoint,
            None => {
                let model = extract_model_from_request(&incoming_headers, &body)
                    .unwrap_or_default();
                let message = format!("model {model} is not supported");
                span.record("http.status_code", StatusCode::BAD_REQUEST.as_u16() as i64);
                span.record("error_message", message.as_str());
                return error_response(StatusCode::BAD_REQUEST, &message);
            }
        };
        let request_model_id = extract_model_from_request(&incoming_headers, &body);
        let selection = match self.selection_strategy.select(
            endpoint,
            request_model_id.as_deref(),
            &(),
        )
        {
            Ok(selection) => selection,
            Err(SelectionError::EndpointNotConfigured) => {
                let model = request_model_id.clone().unwrap_or_default();
                let message = format!("model {model} is not supported");
                span.record("http.status_code", StatusCode::BAD_REQUEST.as_u16() as i64);
                span.record("error_message", message.as_str());
                return error_response(StatusCode::BAD_REQUEST, &message);
            }
            Err(SelectionError::ModelNotConfigured) => {
                span.record("http.status_code", StatusCode::BAD_REQUEST.as_u16() as i64);
                span.record("error_message", "model not configured for endpoint");
                return error_response(
                    StatusCode::BAD_REQUEST,
                    "model not configured for endpoint",
                );
            }
            Err(SelectionError::ModelAmbiguous) => {
                span.record("http.status_code", StatusCode::BAD_REQUEST.as_u16() as i64);
                span.record(
                    "error_message",
                    "multiple configs match endpoint; include model or add default",
                );
                return error_response(
                    StatusCode::BAD_REQUEST,
                    "multiple configs match endpoint; include model or add default",
                );
            }
        };
        span.record(
            "model",
            field::display(request_model_id.as_deref().unwrap_or(&selection.model_id)),
        );
        let binding = match self.binding_for_selection(endpoint, &selection) {
            Some(binding) => binding,
            None => {
                span.record("http.status_code", StatusCode::BAD_REQUEST.as_u16() as i64);
                span.record("error_message", "model not configured for endpoint");
                return error_response(
                    StatusCode::BAD_REQUEST,
                    "model not configured for endpoint",
                );
            }
        };

        let body = ensure_request_model(&incoming_headers, body, &binding.model_id);
        let request_model_id = request_model_id.or_else(|| Some(selection.model_id.clone()));

        let mut outbound = self
            .client
            .request(method, &binding.target_url)
            .body(body.clone());

        outbound = copy_request_headers(outbound, &incoming_headers);

        if let Some(api_key) = &binding.api_key {
            let has_auth = incoming_headers.contains_key("authorization");
            if !has_auth {
                outbound = outbound.bearer_auth(api_key);
            }
        }

        let upstream = match outbound.send().await {
            Ok(resp) => resp,
            Err(err) => {
                error!(
                    path = binding.endpoint.path(),
                    "upstream request failed: {err}"
                );
                span.record("http.status_code", StatusCode::BAD_GATEWAY.as_u16() as i64);
                span.record("error_message", "upstream request failed");
                return error_response(StatusCode::BAD_GATEWAY, "upstream request failed");
            }
        };

        let status = upstream.status();
        let response_headers = upstream.headers().clone();
        let content_type = response_headers
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        let mut builder = Response::builder().status(status);
        for (name, value) in &response_headers {
            if skip_response_header(name) {
                continue;
            }
            builder = builder.header(name, value);
        }

        if content_type.contains("text/event-stream") {
            let stream = stream_sse_response(
                upstream,
                binding.endpoint,
                status,
                request_model_id.clone(),
                Arc::clone(&self.usage_handler),
                span.clone(),
            );
            span.record("http.status_code", status.as_u16() as i64);
            return builder.body(Body::from_stream(stream)).unwrap_or_else(|_| {
                span.record("http.status_code", StatusCode::INTERNAL_SERVER_ERROR.as_u16() as i64);
                span.record("error_message", "failed to build response");
                error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "failed to build response",
                )
            });
        }

        let response_body = match upstream.bytes().await {
            Ok(b) => b,
            Err(err) => {
                error!(
                    path = binding.endpoint.path(),
                    "read upstream body failed: {err}"
                );
                span.record("http.status_code", StatusCode::BAD_GATEWAY.as_u16() as i64);
                span.record("error_message", "failed to read upstream body");
                return error_response(StatusCode::BAD_GATEWAY, "failed to read upstream body");
            }
        };

        let usage = extract_usage(binding.endpoint, &content_type, &response_body);
        record_usage_on_span(&span, &usage);
        dispatch_usage(
            Arc::clone(&self.usage_handler),
            binding.endpoint,
            status,
            request_model_id.clone(),
            usage,
        );
        span.record("http.status_code", status.as_u16() as i64);

        builder.body(Body::from(response_body)).unwrap_or_else(|_| {
            span.record("http.status_code", StatusCode::INTERNAL_SERVER_ERROR.as_u16() as i64);
            span.record("error_message", "failed to build response");
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "failed to build response",
            )
        })
    }

    fn binding_for_selection(
        &self,
        endpoint: Endpoint,
        selection: &SelectionResult,
    ) -> Option<EndpointBinding> {
        self.bindings_by_endpoint
            .get(&endpoint)
            .and_then(|bindings| {
                bindings
                    .iter()
                    .find(|binding| binding.model_id.eq_ignore_ascii_case(&selection.model_id))
                    .cloned()
            })
    }
}

#[derive(Serialize)]
struct ErrorBody {
    error: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "endpoint", rename_all = "snake_case")]
enum Usage {
    Messages(MessagesUsage),
    Responses(ResponsesUsage),
    ChatCompletions(ChatCompletionsUsage),
    Embeddings(EmbeddingsUsage),
    Rerank(RerankUsage),
    AudioSpeech(AudioSpeechUsage),
    AudioTranscriptions(AudioTranscriptionsUsage),
    Images(ImagesUsage),
    Unknown(UnknownUsage),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MessagesUsage {
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    cache_creation_input_tokens: Option<u64>,
    cache_read_input_tokens: Option<u64>,
    cache_creation: Option<MessagesCacheCreation>,
    inference_geo: Option<String>,
    service_tier: Option<String>,
    server_tool_use: Option<MessagesServerToolUse>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MessagesCacheCreation {
    #[serde(rename = "ephemeral_5m_input_tokens")]
    ephemeral_5m_input_tokens: Option<u64>,
    #[serde(rename = "ephemeral_1h_input_tokens")]
    ephemeral_1h_input_tokens: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MessagesServerToolUse {
    web_search_requests: Option<u64>,
    web_fetch_requests: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ResponsesUsage {
    input_tokens: Option<u64>,
    input_tokens_details: Option<ResponsesInputTokensDetails>,
    output_tokens: Option<u64>,
    output_tokens_details: Option<ResponsesOutputTokensDetails>,
    total_tokens: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ResponsesInputTokensDetails {
    cached_tokens: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ResponsesOutputTokensDetails {
    reasoning_tokens: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChatCompletionsUsage {
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    total_tokens: Option<u64>,
    prompt_tokens_details: Option<ChatCompletionsPromptTokensDetails>,
    completion_tokens_details: Option<ChatCompletionsCompletionTokensDetails>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChatCompletionsPromptTokensDetails {
    cached_tokens: Option<u64>,
    audio_tokens: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChatCompletionsCompletionTokensDetails {
    reasoning_tokens: Option<u64>,
    audio_tokens: Option<u64>,
    accepted_prediction_tokens: Option<u64>,
    rejected_prediction_tokens: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EmbeddingsUsage {
    prompt_tokens: Option<u64>,
    total_tokens: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RerankUsage {
    input_tokens: Option<f64>,
    output_tokens: Option<f64>,
    cached_tokens: Option<f64>,
    billed_units: Option<RerankBilledUnits>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RerankBilledUnits {
    images: Option<f64>,
    input_tokens: Option<f64>,
    image_tokens: Option<f64>,
    output_tokens: Option<f64>,
    search_units: Option<f64>,
    classifications: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AudioSpeechUsage {
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    total_tokens: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AudioTranscriptionsUsage {
    #[serde(rename = "type")]
    usage_type: Option<String>,
    input_tokens: Option<u64>,
    input_token_details: Option<AudioTranscriptionsInputTokenDetails>,
    output_tokens: Option<u64>,
    total_tokens: Option<u64>,
    seconds: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AudioTranscriptionsInputTokenDetails {
    audio_tokens: Option<u64>,
    text_tokens: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ImagesUsage {
    total_tokens: Option<u64>,
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    input_tokens_details: Option<ImagesInputTokensDetails>,
    output_tokens_details: Option<ImagesOutputTokensDetails>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ImagesInputTokensDetails {
    text_tokens: Option<u64>,
    image_tokens: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ImagesOutputTokensDetails {
    text_tokens: Option<u64>,
    image_tokens: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
struct UnknownUsage {
    raw: Value,
}

fn build_target_url(base_url: &str, endpoint_path: &str) -> Result<String, String> {
    if base_url.is_empty() {
        return Err("base_url cannot be empty".to_string());
    }

    if base_url.ends_with(endpoint_path) {
        return Ok(base_url.to_string());
    }

    let trimmed = base_url.trim_end_matches('/');
    Ok(format!("{trimmed}{endpoint_path}"))
}

fn copy_request_headers(
    mut req: reqwest::RequestBuilder,
    headers: &HeaderMap,
) -> reqwest::RequestBuilder {
    for (name, value) in headers {
        if skip_request_header(name) {
            continue;
        }
        req = req.header(name, value);
    }
    req
}

fn skip_request_header(name: &HeaderName) -> bool {
    matches!(
        name.as_str().to_ascii_lowercase().as_str(),
        "host" | "content-length" | "connection" | "transfer-encoding"
    )
}

fn skip_response_header(name: &HeaderName) -> bool {
    matches!(
        name.as_str().to_ascii_lowercase().as_str(),
        "content-length" | "connection" | "transfer-encoding"
    )
}

fn ensure_request_model(headers: &HeaderMap, body: Bytes, target_model: &str) -> Bytes {
    let content_type = headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default();

    if !content_type.contains("application/json") {
        return body;
    }

    let mut json = match serde_json::from_slice::<Value>(&body) {
        Ok(v) => v,
        Err(_) => return body,
    };

    let obj = match json.as_object_mut() {
        Some(o) => o,
        None => return body,
    };

    let current_model = obj.get("model").and_then(|value| value.as_str());
    if current_model == Some(target_model) {
        return body;
    }

    obj.insert(
        "model".to_string(),
        Value::String(target_model.to_string()),
    );

    serde_json::to_vec(&json).map(Bytes::from).unwrap_or(body)
}

fn extract_model_from_request(headers: &HeaderMap, body: &Bytes) -> Option<String> {
    let content_type = headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default();

    if !content_type.contains("application/json") {
        return None;
    }

    let value: Value = serde_json::from_slice(body).ok()?;
    value
        .get("model")
        .and_then(|v| v.as_str())
        .map(ToString::to_string)
}

trait UsageHandler {
    fn handle(
        &self,
        endpoint: Endpoint,
        status: reqwest::StatusCode,
        request_model: Option<String>,
        usage: Option<Usage>,
    );
}

struct LogUsageHandler;

impl UsageHandler for LogUsageHandler {
    fn handle(
        &self,
        endpoint: Endpoint,
        status: reqwest::StatusCode,
        request_model: Option<String>,
        usage: Option<Usage>,
    ) {
        if status.is_success() {
            info!(
                endpoint = endpoint.path(),
                model = request_model.as_deref().unwrap_or(""),
                usage = ?usage,
                "proxy request succeeded"
            );
        } else {
            warn!(
                endpoint = endpoint.path(),
                status = status.as_u16(),
                model = request_model.as_deref().unwrap_or(""),
                usage = ?usage,
                "proxy request failed"
            );
        }
    }
}

fn stream_sse_response(
    upstream: reqwest::Response,
    endpoint: Endpoint,
    status: reqwest::StatusCode,
    request_model: Option<String>,
    usage_handler: Arc<dyn UsageHandler + Send + Sync>,
    span: Span,
) -> impl futures_util::Stream<Item = Result<Bytes, io::Error>> {
    let mut stream = upstream.bytes_stream();
    let mut tracker = SseUsageAccumulator::new(endpoint, status, request_model, usage_handler);

    async_stream::stream! {
        while let Some(item) = stream.next().await {
            match item {
                Ok(chunk) => {
                    tracker.ingest(&chunk);
                    yield Ok(chunk);
                }
                Err(err) => {
                    tracker.log(None);
                    yield Err(io::Error::new(io::ErrorKind::Other, err));
                    return;
                }
            }
        }
        let usage = tracker.latest_usage.take();
        tracker.log(usage.clone());
        record_usage_on_span(&span, &usage);
    }
}

fn extract_usage(endpoint: Endpoint, content_type: &str, body: &[u8]) -> Option<Usage> {
    if content_type.contains("text/event-stream") {
        return extract_usage_from_sse(endpoint, body);
    }
    extract_usage_from_json_bytes(endpoint, body)
}

fn dispatch_usage(
    handler: Arc<dyn UsageHandler + Send + Sync>,
    endpoint: Endpoint,
    status: reqwest::StatusCode,
    request_model: Option<String>,
    usage: Option<Usage>,
) {
    tokio::spawn(async move {
        handler.handle(endpoint, status, request_model, usage);
    });
}

fn record_usage_on_span(span: &Span, usage: &Option<Usage>) {
    let Some(usage) = usage else {
        return;
    };
    if let Ok(value) = serde_json::to_string(usage) {
        span.record("usage", &value.as_str());
    }
}

struct SseUsageAccumulator {
    endpoint: Endpoint,
    status: reqwest::StatusCode,
    request_model: Option<String>,
    usage_handler: Arc<dyn UsageHandler + Send + Sync>,
    buffer: Vec<u8>,
    latest_usage: Option<Usage>,
}

impl SseUsageAccumulator {
    fn new(
        endpoint: Endpoint,
        status: reqwest::StatusCode,
        request_model: Option<String>,
        usage_handler: Arc<dyn UsageHandler + Send + Sync>,
    ) -> Self {
        Self {
            endpoint,
            status,
            request_model,
            usage_handler,
            buffer: Vec::new(),
            latest_usage: None,
        }
    }

    fn ingest(&mut self, chunk: &Bytes) {
        self.buffer.extend_from_slice(chunk);
        while let Some(pos) = self.buffer.iter().position(|b| *b == b'\n') {
            let line = self.buffer.drain(..=pos).collect::<Vec<u8>>();
            let line = trim_line_end(&line);
            if line.is_empty() {
                continue;
            }
            if let Some(payload) = line.strip_prefix(b"data:") {
                let payload = trim_left_spaces(payload);
                if payload == b"[DONE]" {
                    continue;
                }
                if let Ok(json) = serde_json::from_slice::<Value>(payload) {
                    if let Some(found) = extract_usage_from_json_value(self.endpoint, &json) {
                        self.latest_usage = accumulate_usage(self.latest_usage.take(), found);
                    }
                }
            }
        }
    }

    fn log(&self, usage: Option<Usage>) {
        let usage = usage.or_else(|| self.latest_usage.clone());
        dispatch_usage(
            Arc::clone(&self.usage_handler),
            self.endpoint,
            self.status,
            self.request_model.clone(),
            usage,
        );
    }
}

fn trim_line_end(line: &[u8]) -> &[u8] {
    let mut end = line.len();
    while end > 0 {
        let byte = line[end - 1];
        if byte == b'\n' || byte == b'\r' {
            end -= 1;
            continue;
        }
        break;
    }
    &line[..end]
}

fn trim_left_spaces(input: &[u8]) -> &[u8] {
    let mut start = 0usize;
    while start < input.len() && input[start] == b' ' {
        start += 1;
    }
    &input[start..]
}

fn extract_usage_from_sse(endpoint: Endpoint, body: &[u8]) -> Option<Usage> {
    let text = std::str::from_utf8(body).ok()?;
    let mut latest = None;

    for line in text.lines() {
        let line = line.trim();
        if !line.starts_with("data:") {
            continue;
        }

        let data = line.trim_start_matches("data:").trim();
        if data.is_empty() || data == "[DONE]" {
            continue;
        }

        if let Ok(json) = serde_json::from_str::<Value>(data) {
            if let Some(record) = extract_usage_from_json_value(endpoint, &json) {
                latest = accumulate_usage(latest, record);
            }
        }
    }

    latest
}

fn accumulate_usage(current: Option<Usage>, incoming: Usage) -> Option<Usage> {
    match current {
        None => Some(incoming),
        Some(existing) => Some(merge_usage(existing, incoming)),
    }
}

fn merge_usage(existing: Usage, incoming: Usage) -> Usage {
    match (existing, incoming) {
        (Usage::Messages(mut left), Usage::Messages(right)) => {
            left.input_tokens = sum_opt(left.input_tokens, right.input_tokens);
            left.output_tokens = sum_opt(left.output_tokens, right.output_tokens);
            left.cache_creation_input_tokens = sum_opt(
                left.cache_creation_input_tokens,
                right.cache_creation_input_tokens,
            );
            left.cache_read_input_tokens =
                sum_opt(left.cache_read_input_tokens, right.cache_read_input_tokens);
            left.cache_creation = merge_messages_cache_creation(
                left.cache_creation,
                right.cache_creation,
            );
            left.server_tool_use =
                merge_messages_server_tool_use(left.server_tool_use, right.server_tool_use);
            if left.inference_geo.is_none() {
                left.inference_geo = right.inference_geo;
            }
            if left.service_tier.is_none() {
                left.service_tier = right.service_tier;
            }
            Usage::Messages(left)
        }
        (Usage::Responses(mut left), Usage::Responses(right)) => {
            left.input_tokens = sum_opt(left.input_tokens, right.input_tokens);
            left.output_tokens = sum_opt(left.output_tokens, right.output_tokens);
            left.total_tokens = sum_opt(left.total_tokens, right.total_tokens);
            left.input_tokens_details =
                merge_response_input_details(left.input_tokens_details, right.input_tokens_details);
            left.output_tokens_details = merge_response_output_details(
                left.output_tokens_details,
                right.output_tokens_details,
            );
            Usage::Responses(left)
        }
        (Usage::ChatCompletions(mut left), Usage::ChatCompletions(right)) => {
            left.prompt_tokens = sum_opt(left.prompt_tokens, right.prompt_tokens);
            left.completion_tokens = sum_opt(left.completion_tokens, right.completion_tokens);
            left.total_tokens = sum_opt(left.total_tokens, right.total_tokens);
            left.prompt_tokens_details = merge_chat_prompt_tokens_details(
                left.prompt_tokens_details,
                right.prompt_tokens_details,
            );
            left.completion_tokens_details = merge_chat_completion_tokens_details(
                left.completion_tokens_details,
                right.completion_tokens_details,
            );
            Usage::ChatCompletions(left)
        }
        (Usage::Embeddings(mut left), Usage::Embeddings(right)) => {
            left.prompt_tokens = sum_opt(left.prompt_tokens, right.prompt_tokens);
            left.total_tokens = sum_opt(left.total_tokens, right.total_tokens);
            Usage::Embeddings(left)
        }
        (Usage::Rerank(mut left), Usage::Rerank(right)) => {
            left.input_tokens = sum_opt_f64(left.input_tokens, right.input_tokens);
            left.output_tokens = sum_opt_f64(left.output_tokens, right.output_tokens);
            left.cached_tokens = sum_opt_f64(left.cached_tokens, right.cached_tokens);
            left.billed_units = merge_rerank_billed_units(left.billed_units, right.billed_units);
            Usage::Rerank(left)
        }
        (Usage::AudioSpeech(mut left), Usage::AudioSpeech(right)) => {
            left.input_tokens = sum_opt(left.input_tokens, right.input_tokens);
            left.output_tokens = sum_opt(left.output_tokens, right.output_tokens);
            left.total_tokens = sum_opt(left.total_tokens, right.total_tokens);
            Usage::AudioSpeech(left)
        }
        (Usage::AudioTranscriptions(mut left), Usage::AudioTranscriptions(right)) => {
            left.input_tokens = sum_opt(left.input_tokens, right.input_tokens);
            left.output_tokens = sum_opt(left.output_tokens, right.output_tokens);
            left.total_tokens = sum_opt(left.total_tokens, right.total_tokens);
            left.input_token_details = merge_audio_transcriptions_input_details(
                left.input_token_details,
                right.input_token_details,
            );
            left.seconds = sum_opt_f64(left.seconds, right.seconds);
            if left.usage_type.is_none() {
                left.usage_type = right.usage_type;
            }
            Usage::AudioTranscriptions(left)
        }
        (Usage::Images(mut left), Usage::Images(right)) => {
            left.total_tokens = sum_opt(left.total_tokens, right.total_tokens);
            left.input_tokens = sum_opt(left.input_tokens, right.input_tokens);
            left.output_tokens = sum_opt(left.output_tokens, right.output_tokens);
            left.input_tokens_details =
                merge_image_input_details(left.input_tokens_details, right.input_tokens_details);
            left.output_tokens_details = merge_image_output_details(
                left.output_tokens_details,
                right.output_tokens_details,
            );
            Usage::Images(left)
        }
        (_, incoming) => incoming,
    }
}

fn merge_response_input_details(
    left: Option<ResponsesInputTokensDetails>,
    right: Option<ResponsesInputTokensDetails>,
) -> Option<ResponsesInputTokensDetails> {
    match (left, right) {
        (None, None) => None,
        (Some(l), None) => Some(l),
        (None, Some(r)) => Some(r),
        (Some(mut l), Some(r)) => {
            l.cached_tokens = sum_opt(l.cached_tokens, r.cached_tokens);
            Some(l)
        }
    }
}

fn merge_response_output_details(
    left: Option<ResponsesOutputTokensDetails>,
    right: Option<ResponsesOutputTokensDetails>,
) -> Option<ResponsesOutputTokensDetails> {
    match (left, right) {
        (None, None) => None,
        (Some(l), None) => Some(l),
        (None, Some(r)) => Some(r),
        (Some(mut l), Some(r)) => {
            l.reasoning_tokens = sum_opt(l.reasoning_tokens, r.reasoning_tokens);
            Some(l)
        }
    }
}

fn merge_image_input_details(
    left: Option<ImagesInputTokensDetails>,
    right: Option<ImagesInputTokensDetails>,
) -> Option<ImagesInputTokensDetails> {
    match (left, right) {
        (None, None) => None,
        (Some(l), None) => Some(l),
        (None, Some(r)) => Some(r),
        (Some(mut l), Some(r)) => {
            l.text_tokens = sum_opt(l.text_tokens, r.text_tokens);
            l.image_tokens = sum_opt(l.image_tokens, r.image_tokens);
            Some(l)
        }
    }
}

fn merge_image_output_details(
    left: Option<ImagesOutputTokensDetails>,
    right: Option<ImagesOutputTokensDetails>,
) -> Option<ImagesOutputTokensDetails> {
    match (left, right) {
        (None, None) => None,
        (Some(l), None) => Some(l),
        (None, Some(r)) => Some(r),
        (Some(mut l), Some(r)) => {
            l.text_tokens = sum_opt(l.text_tokens, r.text_tokens);
            l.image_tokens = sum_opt(l.image_tokens, r.image_tokens);
            Some(l)
        }
    }
}

fn merge_messages_cache_creation(
    left: Option<MessagesCacheCreation>,
    right: Option<MessagesCacheCreation>,
) -> Option<MessagesCacheCreation> {
    match (left, right) {
        (None, None) => None,
        (Some(l), None) => Some(l),
        (None, Some(r)) => Some(r),
        (Some(mut l), Some(r)) => {
            l.ephemeral_5m_input_tokens =
                sum_opt(l.ephemeral_5m_input_tokens, r.ephemeral_5m_input_tokens);
            l.ephemeral_1h_input_tokens =
                sum_opt(l.ephemeral_1h_input_tokens, r.ephemeral_1h_input_tokens);
            Some(l)
        }
    }
}

fn merge_messages_server_tool_use(
    left: Option<MessagesServerToolUse>,
    right: Option<MessagesServerToolUse>,
) -> Option<MessagesServerToolUse> {
    match (left, right) {
        (None, None) => None,
        (Some(l), None) => Some(l),
        (None, Some(r)) => Some(r),
        (Some(mut l), Some(r)) => {
            l.web_search_requests = sum_opt(l.web_search_requests, r.web_search_requests);
            l.web_fetch_requests = sum_opt(l.web_fetch_requests, r.web_fetch_requests);
            Some(l)
        }
    }
}

fn merge_chat_prompt_tokens_details(
    left: Option<ChatCompletionsPromptTokensDetails>,
    right: Option<ChatCompletionsPromptTokensDetails>,
) -> Option<ChatCompletionsPromptTokensDetails> {
    match (left, right) {
        (None, None) => None,
        (Some(l), None) => Some(l),
        (None, Some(r)) => Some(r),
        (Some(mut l), Some(r)) => {
            l.cached_tokens = sum_opt(l.cached_tokens, r.cached_tokens);
            l.audio_tokens = sum_opt(l.audio_tokens, r.audio_tokens);
            Some(l)
        }
    }
}

fn merge_chat_completion_tokens_details(
    left: Option<ChatCompletionsCompletionTokensDetails>,
    right: Option<ChatCompletionsCompletionTokensDetails>,
) -> Option<ChatCompletionsCompletionTokensDetails> {
    match (left, right) {
        (None, None) => None,
        (Some(l), None) => Some(l),
        (None, Some(r)) => Some(r),
        (Some(mut l), Some(r)) => {
            l.reasoning_tokens = sum_opt(l.reasoning_tokens, r.reasoning_tokens);
            l.audio_tokens = sum_opt(l.audio_tokens, r.audio_tokens);
            l.accepted_prediction_tokens =
                sum_opt(l.accepted_prediction_tokens, r.accepted_prediction_tokens);
            l.rejected_prediction_tokens =
                sum_opt(l.rejected_prediction_tokens, r.rejected_prediction_tokens);
            Some(l)
        }
    }
}

fn merge_audio_transcriptions_input_details(
    left: Option<AudioTranscriptionsInputTokenDetails>,
    right: Option<AudioTranscriptionsInputTokenDetails>,
) -> Option<AudioTranscriptionsInputTokenDetails> {
    match (left, right) {
        (None, None) => None,
        (Some(l), None) => Some(l),
        (None, Some(r)) => Some(r),
        (Some(mut l), Some(r)) => {
            l.audio_tokens = sum_opt(l.audio_tokens, r.audio_tokens);
            l.text_tokens = sum_opt(l.text_tokens, r.text_tokens);
            Some(l)
        }
    }
}

fn merge_rerank_billed_units(
    left: Option<RerankBilledUnits>,
    right: Option<RerankBilledUnits>,
) -> Option<RerankBilledUnits> {
    match (left, right) {
        (None, None) => None,
        (Some(l), None) => Some(l),
        (None, Some(r)) => Some(r),
        (Some(mut l), Some(r)) => {
            l.images = sum_opt_f64(l.images, r.images);
            l.input_tokens = sum_opt_f64(l.input_tokens, r.input_tokens);
            l.image_tokens = sum_opt_f64(l.image_tokens, r.image_tokens);
            l.output_tokens = sum_opt_f64(l.output_tokens, r.output_tokens);
            l.search_units = sum_opt_f64(l.search_units, r.search_units);
            l.classifications = sum_opt_f64(l.classifications, r.classifications);
            Some(l)
        }
    }
}

fn sum_opt(left: Option<u64>, right: Option<u64>) -> Option<u64> {
    match (left, right) {
        (Some(l), Some(r)) => Some(l + r),
        (Some(l), None) => Some(l),
        (None, Some(r)) => Some(r),
        (None, None) => None,
    }
}

fn sum_opt_f64(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (left, right) {
        (Some(l), Some(r)) => Some(l + r),
        (Some(l), None) => Some(l),
        (None, Some(r)) => Some(r),
        (None, None) => None,
    }
}

fn extract_usage_from_json_bytes(endpoint: Endpoint, body: &[u8]) -> Option<Usage> {
    let json: Value = serde_json::from_slice(body).ok()?;
    extract_usage_from_json_value(endpoint, &json)
}

fn extract_usage_from_json_value(endpoint: Endpoint, value: &Value) -> Option<Usage> {
    match endpoint {
        Endpoint::Rerank => {
            if let Some(meta) = value.get("meta").and_then(Value::as_object) {
                if let Some(usage_value) = extract_rerank_usage(meta) {
                    return Some(match serde_json::from_value::<RerankUsage>(usage_value.clone()) {
                        Ok(usage) => Usage::Rerank(usage),
                        Err(_) => Usage::Unknown(UnknownUsage { raw: usage_value }),
                    });
                }
            }
        }
        _ => {
            if let Some(usage_value) = find_usage_value(value) {
                return Some(match endpoint {
                    Endpoint::Messages => {
                        parse_usage_variant::<MessagesUsage, _>(usage_value, Usage::Messages)
                    }
                    Endpoint::Responses => {
                        parse_usage_variant::<ResponsesUsage, _>(usage_value, Usage::Responses)
                    }
                    Endpoint::ChatCompletions => parse_usage_variant::<ChatCompletionsUsage, _>(
                        usage_value,
                        Usage::ChatCompletions,
                    ),
                    Endpoint::Embeddings => {
                        parse_usage_variant::<EmbeddingsUsage, _>(usage_value, Usage::Embeddings)
                    }
                    Endpoint::AudioSpeech => {
                        parse_usage_variant::<AudioSpeechUsage, _>(usage_value, Usage::AudioSpeech)
                    }
                    Endpoint::AudioTranscriptions => {
                        parse_usage_variant::<AudioTranscriptionsUsage, _>(
                            usage_value,
                            Usage::AudioTranscriptions,
                        )
                    }
                    Endpoint::ImagesGenerations | Endpoint::ImagesEdits => {
                        parse_usage_variant::<ImagesUsage, _>(usage_value, Usage::Images)
                    }
                    Endpoint::Rerank => Usage::Unknown(UnknownUsage { raw: usage_value }),
                });
            }
        }
    }

    match value {
        Value::Object(map) => {
            for child in map.values() {
                if let Some(found) = extract_usage_from_json_value(endpoint, child) {
                    return Some(found);
                }
            }
            None
        }
        Value::Array(items) => {
            for item in items {
                if let Some(found) = extract_usage_from_json_value(endpoint, item) {
                    return Some(found);
                }
            }
            None
        }
        _ => None,
    }
}

fn extract_rerank_usage(meta: &serde_json::Map<String, Value>) -> Option<Value> {
    let mut usage = serde_json::Map::new();

    if let Some(tokens) = meta.get("tokens").and_then(Value::as_object) {
        if let Some(input_tokens) = tokens.get("input_tokens") {
            usage.insert("input_tokens".to_string(), input_tokens.clone());
        }
        if let Some(output_tokens) = tokens.get("output_tokens") {
            usage.insert("output_tokens".to_string(), output_tokens.clone());
        }
    }

    if let Some(cached_tokens) = meta.get("cached_tokens") {
        usage.insert("cached_tokens".to_string(), cached_tokens.clone());
    }

    if let Some(billed_units) = meta.get("billed_units") {
        usage.insert("billed_units".to_string(), billed_units.clone());
    }

    if usage.is_empty() {
        None
    } else {
        Some(Value::Object(usage))
    }
}

fn parse_usage_variant<T, F>(usage_value: Value, make: F) -> Usage
where
    T: for<'de> Deserialize<'de>,
    F: FnOnce(T) -> Usage,
{
    match serde_json::from_value::<T>(usage_value.clone()) {
        Ok(typed) => make(typed),
        Err(_) => Usage::Unknown(UnknownUsage { raw: usage_value }),
    }
}

fn find_usage_value(value: &Value) -> Option<Value> {
    if let Some(usage) = value.get("usage") {
        return Some(usage.clone());
    }

    match value {
        Value::Object(map) => {
            for child in map.values() {
                if let Some(found) = find_usage_value(child) {
                    return Some(found);
                }
            }
            None
        }
        Value::Array(items) => {
            for item in items {
                if let Some(found) = find_usage_value(item) {
                    return Some(found);
                }
            }
            None
        }
        _ => None,
    }
}

fn error_response(status: StatusCode, msg: impl Into<String>) -> Response<Body> {
    (status, Json(ErrorBody { error: msg.into() })).into_response()
}
