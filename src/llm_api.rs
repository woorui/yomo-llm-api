use std::fmt;
use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::Context;
use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use log::{error, info};
use opentelemetry::trace::TraceContextExt;
use tracing::{Span, field, info_span, Instrument};
use tracing_opentelemetry::OpenTelemetrySpanExt;
use yomo::tool_mgr::ToolMgr;
use yomo::metadata_mgr::MetadataMgr;
use yomo::auth::Auth;

use crate::metadata::Metadata;
use crate::agent_loop::{
    AgentLoopConfig, AgentLoopResult, run_agent_loop,
};
use crate::tool_invoker::ToolInvoker;
use crate::openai_http_mapping::{
    map_chat_error, map_openai_response, openai_error_response, stream_openai_chunks,
    validate_openai_request,
};
use crate::openai_types::ChatCompletionRequest;
use crate::provider_registry::ProviderRegistry;
use crate::provider_registry::ByModel;

pub struct LlmApiState<A, M> {
    pub registry: Arc<ProviderRegistry<M>>,
    pub tool_mgr: Arc<dyn ToolMgr<A, M>>,
    pub tool_invoker: Arc<dyn ToolInvoker<M>>,
    pub metadata_mgr: Arc<dyn MetadataMgr<A, M>>,
    pub auth: Arc<dyn Auth<A>>,
    pub header_extractor: Arc<dyn HeaderExtractor>,
}

impl<A, M> Clone for LlmApiState<A, M> {
    fn clone(&self) -> Self {
        Self {
            registry: Arc::clone(&self.registry),
            tool_mgr: Arc::clone(&self.tool_mgr),
            tool_invoker: Arc::clone(&self.tool_invoker),
            metadata_mgr: Arc::clone(&self.metadata_mgr),
            auth: Arc::clone(&self.auth),
            header_extractor: Arc::clone(&self.header_extractor),
        }
    }
}

pub trait HeaderExtractor: Send + Sync {
    fn credential(&self, headers: &HeaderMap) -> String;
    fn extension(&self, headers: &HeaderMap) -> String;
}

#[derive(Default)]
pub struct DefaultHeaderExtractor;

impl HeaderExtractor for DefaultHeaderExtractor {
    fn credential(&self, headers: &HeaderMap) -> String {
        headers
            .get("authorization")
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.strip_prefix("Bearer "))
            .unwrap_or("")
            .to_string()
    }

    fn extension(&self, headers: &HeaderMap) -> String {
        headers
            .get("X-Extension")
            .and_then(|value| value.to_str().ok())
            .unwrap_or("")
            .to_string()
    }
}


pub async fn run_http_server<A>(
    addr: SocketAddr,
    state: LlmApiState<A, Metadata>,
) -> Result<(), anyhow::Error>
where
    A: Send + Sync + 'static,
{
    let app = Router::new()
        .route("/v1/chat/completions", post(handle_chat_completions::<A>))
        .with_state(state);

    info!("listening on {addr}");
    axum::serve(
        tokio::net::TcpListener::bind(addr)
            .await
            .context("bind failed")?,
        app,
    )
    .await
    .context("server failed")?;
    Ok(())
}

pub fn build_provider_registry<M>(config_path: &str) -> Result<ProviderRegistry<M>, anyhow::Error> {
    let config = crate::config::Config::load(config_path)
        .and_then(|config| ProviderRegistry::from_config(&config, Arc::new(ByModel)))
        .map_err(|err| anyhow::anyhow!("failed to load config {config_path}: {err}"))?;
    Ok(config)
}

async fn handle_chat_completions<A>(
    State(state): State<LlmApiState<A, Metadata>>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse
where
    A: Send + Sync + 'static,
{
    let credential = state.header_extractor.credential(&headers);
    let auth_info = match state.auth.authenticate(&credential).await {
        Ok(auth_info) => auth_info,
        Err(err) => {
            error!("authentication failed: {err}");
            return openai_error_response(
                StatusCode::UNAUTHORIZED,
                &err.to_string(),
                Some("authentication_error"),
            );
        }
    };
    let extension = state.header_extractor.extension(&headers);
    let mut metadata = match state.metadata_mgr.new_from_extension(&auth_info, &extension) {
        Ok(metadata) => metadata,
        Err(err) => {
            error!("metadata build failed: {err}");
            return openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &err.to_string(),
                Some("server_error"),
            );
        }
    };
    let root_span = info_span!(
        "http.request",
        http.method = "POST",
        http.route = "/v1/chat/completions",
        http.status_code = field::Empty,
        model = field::Empty,
        streaming = field::Empty,
    );
    let otel_trace_id = root_span
        .context()
        .span()
        .span_context()
        .trace_id()
        .to_string();
    metadata.trace_id = otel_trace_id;

    let trace_id = metadata.trace_id.clone();
    let extension = metadata.extension.clone();
    info!("chat request received {}", metadata);
    let metadata_for_error = metadata.clone();
    match handle_chat_completions_inner::<A, Metadata>(
        state,
        metadata,
        trace_id,
        extension,
        body,
        root_span.clone(),
    )
        .instrument(root_span.clone())
        .await
    {
        Ok(response) => response,
        Err(err) => {
            root_span.record("http.status_code", StatusCode::INTERNAL_SERVER_ERROR.as_u16() as i64);
            error!("chat completion failed: {err} {}", metadata_for_error);
            openai_error_response(StatusCode::INTERNAL_SERVER_ERROR, "internal error", None)
        }
    }
}


async fn handle_chat_completions_inner<A, M>(
    state: LlmApiState<A, M>,
    metadata: M,
    trace_id: String,
    extension: String,
    body: Bytes,
    root_span: Span,
) -> Result<Response, anyhow::Error>
where
    A: Send + Sync + 'static,
    M: fmt::Display + Clone + Send + Sync + 'static,
{
    let mut request: ChatCompletionRequest =
        serde_json::from_slice(&body).context("invalid json body")?;
    let stream = request.stream.unwrap_or(false);
    root_span.record("model", field::display(&request.model));
    root_span.record("streaming", stream);
    if stream {
        match &mut request.stream_options {
            Some(options) => {
                options.include_usage = true;
            }
            None => {
                request.stream_options = Some(crate::openai_types::StreamOptions {
                    include_usage: true,
                    include_obfuscation: None,
                });
            }
        }
    }
    info!(
        "chat request parsed: model={}, stream={} {}",
        request.model, stream, metadata
    );
    if let Err(message) = validate_openai_request(&request) {
        error!(
            "chat request invalid: model={}, error={} {}",
            request.model, message, metadata
        );
        root_span.record("http.status_code", StatusCode::BAD_REQUEST.as_u16() as i64);
        return Ok(openai_error_response(
            StatusCode::BAD_REQUEST,
            &message,
            Some("invalid_request_error"),
        ));
    }

    let provider = match state.registry.select(&request.model, &metadata) {
        Ok(entry) => Arc::clone(&entry.provider),
        Err(err) => {
            error!(
                "chat selection failed: model={}, error={} {}",
                request.model, err, metadata
            );
            root_span.record("http.status_code", StatusCode::BAD_REQUEST.as_u16() as i64);
            return Ok(openai_error_response(
                StatusCode::BAD_REQUEST,
                &err.to_string(),
                Some("invalid_request_error"),
            ));
        }
    };

    let model_for_log = request.model.clone();
    let request_model = request.model.clone();
    let metadata_for_log = metadata.clone();
    let loop_result = run_agent_loop(
        provider,
        request,
        Arc::clone(&state.tool_mgr),
        Arc::clone(&state.tool_invoker),
        metadata,
        trace_id.clone(),
        extension,
        AgentLoopConfig::default(),
        root_span.clone(),
    )
    .await;

    match loop_result {
        Ok(AgentLoopResult::NonStream(response)) => {
            info!(
                "chat request success: model={} {}",
                response.model, metadata_for_log
            );
            let mapped = map_openai_response(response);
            let payload = serde_json::to_vec(&mapped).context("serialize response")?;
            root_span.record("http.status_code", StatusCode::OK.as_u16() as i64);
            Ok(Response::builder()
                .status(StatusCode::OK)
                .body(Body::from(payload))
                .expect("build response"))
        }
        Ok(AgentLoopResult::Stream { events }) => {
            root_span.record("http.status_code", StatusCode::OK.as_u16() as i64);
            let sse = stream_openai_chunks(events, trace_id, request_model, root_span.clone());
            let body = Body::from_stream(sse);
            Ok(Response::builder()
                .status(StatusCode::OK)
                .body(body)
                .expect("build response"))
        }
        Err(err) => {
            error!(
                "chat request failed: model={}, error={} {}",
                model_for_log, err, metadata_for_log
            );
            let response = map_chat_error(err);
            root_span.record("http.status_code", response.status().as_u16() as i64);
            Ok(response)
        }
    }
}
