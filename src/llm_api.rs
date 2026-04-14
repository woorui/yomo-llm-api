use std::collections::HashMap;
use std::fmt;
use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::Context;
use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, Request, StatusCode};
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
use crate::provider::Provider;
use crate::config::ProviderConfig;
use crate::endpoint::Endpoint;
use crate::model_api::ModelApi;
use crate::provider_registry::{RouteSelectionError, SelectionStrategy};

pub struct LlmApiState<A, M> {
    pub providers: Arc<HashMap<String, ProviderConfig>>,
    pub agent_providers: Arc<HashMap<String, Arc<dyn Provider>>>,
    pub model_api: Arc<ModelApi>,
    pub route_strategy: Arc<dyn SelectionStrategy<M>>,
    pub tool_mgr: Arc<dyn ToolMgr<A, M>>,
    pub tool_invoker: Arc<dyn ToolInvoker<M>>,
    pub metadata_mgr: Arc<dyn MetadataMgr<A, M>>,
    pub auth: Arc<dyn Auth<A>>,
    pub header_extractor: Arc<dyn HeaderExtractor>,
}

impl<A, M> Clone for LlmApiState<A, M> {
    fn clone(&self) -> Self {
        Self {
            providers: Arc::clone(&self.providers),
            agent_providers: Arc::clone(&self.agent_providers),
            model_api: Arc::clone(&self.model_api),
            route_strategy: Arc::clone(&self.route_strategy),
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
        .route("/v1/messages", post(handle_messages::<A>))
        .route("/v1/responses", post(handle_responses::<A>))
        .route("/v1/embeddings", post(handle_embeddings::<A>))
        .route("/v1/rerank", post(handle_rerank::<A>))
        .route("/v1/audio/speech", post(handle_audio_speech::<A>))
        .route(
            "/v1/audio/transcriptions",
            post(handle_audio_transcriptions::<A>),
        )
        .route(
            "/v1/images/generations",
            post(handle_images_generations::<A>),
        )
        .route("/v1/images/edits", post(handle_images_edits::<A>))
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

async fn handle_messages<A>(
    State(state): State<LlmApiState<A, Metadata>>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse
where
    A: Send + Sync + 'static,
{
    handle_endpoint_proxy(
        State(state),
        headers,
        body,
        Endpoint::Messages.path(),
    )
    .await
}

async fn handle_responses<A>(
    State(state): State<LlmApiState<A, Metadata>>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse
where
    A: Send + Sync + 'static,
{
    handle_endpoint_proxy(
        State(state),
        headers,
        body,
        Endpoint::Responses.path(),
    )
    .await
}

async fn handle_embeddings<A>(
    State(state): State<LlmApiState<A, Metadata>>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse
where
    A: Send + Sync + 'static,
{
    handle_endpoint_proxy(
        State(state),
        headers,
        body,
        Endpoint::Embeddings.path(),
    )
    .await
}

async fn handle_rerank<A>(
    State(state): State<LlmApiState<A, Metadata>>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse
where
    A: Send + Sync + 'static,
{
    handle_endpoint_proxy(State(state), headers, body, Endpoint::Rerank.path()).await
}

async fn handle_audio_speech<A>(
    State(state): State<LlmApiState<A, Metadata>>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse
where
    A: Send + Sync + 'static,
{
    handle_endpoint_proxy(
        State(state),
        headers,
        body,
        Endpoint::AudioSpeech.path(),
    )
    .await
}

async fn handle_audio_transcriptions<A>(
    State(state): State<LlmApiState<A, Metadata>>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse
where
    A: Send + Sync + 'static,
{
    handle_endpoint_proxy(
        State(state),
        headers,
        body,
        Endpoint::AudioTranscriptions.path(),
    )
    .await
}

async fn handle_images_generations<A>(
    State(state): State<LlmApiState<A, Metadata>>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse
where
    A: Send + Sync + 'static,
{
    handle_endpoint_proxy(
        State(state),
        headers,
        body,
        Endpoint::ImagesGenerations.path(),
    )
    .await
}

async fn handle_images_edits<A>(
    State(state): State<LlmApiState<A, Metadata>>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse
where
    A: Send + Sync + 'static,
{
    handle_endpoint_proxy(
        State(state),
        headers,
        body,
        Endpoint::ImagesEdits.path(),
    )
    .await
}

async fn handle_endpoint_proxy<A>(
    State(state): State<LlmApiState<A, Metadata>>,
    headers: HeaderMap,
    body: Bytes,
    path: &'static str,
) -> impl IntoResponse
where
    A: Send + Sync + 'static,
{
    if let Err(response) = authenticate_request(&state, &headers).await {
        return response;
    }

    let proxy = match Request::builder()
        .method("POST")
        .uri(path)
        .body(Body::from(body))
    {
        Ok(proxy) => proxy,
        Err(err) => {
            error!("build endpoint proxy request failed: {err}");
            return openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "failed to build proxy request",
                Some("server_error"),
            );
        }
    };
    let mut proxy = proxy;
    *proxy.headers_mut() = headers;
    state.model_api.handle(proxy).await
}

async fn authenticate_request<A>(
    state: &LlmApiState<A, Metadata>,
    headers: &HeaderMap,
) -> Result<Metadata, Response>
where
    A: Send + Sync + 'static,
{
    let credential = state.header_extractor.credential(headers);
    let auth_info = match state.auth.authenticate(&credential).await {
        Ok(auth_info) => auth_info,
        Err(err) => {
            error!("authentication failed: {err}");
            return Err(openai_error_response(
                StatusCode::UNAUTHORIZED,
                &err.to_string(),
                Some("authentication_error"),
            ));
        }
    };
    let extension = state.header_extractor.extension(headers);
    let metadata = match state.metadata_mgr.new_from_extension(&auth_info, &extension) {
        Ok(metadata) => metadata,
        Err(err) => {
            error!("metadata build failed: {err}");
            return Err(openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &err.to_string(),
                Some("server_error"),
            ));
        }
    };
    Ok(metadata)
}


async fn handle_chat_completions<A>(
    State(state): State<LlmApiState<A, Metadata>>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse
where
    A: Send + Sync + 'static,
{
    let mut metadata = match authenticate_request(&state, &headers).await {
        Ok(metadata) => metadata,
        Err(response) => return response,
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
        headers,
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
    headers: HeaderMap,
    body: Bytes,
    root_span: Span,
) -> Result<Response, anyhow::Error>
where
    A: Send + Sync + 'static,
    M: fmt::Display + Clone + Send + Sync + 'static,
{
    let mut request: ChatCompletionRequest =
        serde_json::from_slice(&body).context("invalid json body")?;

    let server_tools = state
        .tool_mgr
        .list_tools(&metadata)
        .await
        .map_err(|err| anyhow::anyhow!("tool manager error: {err}"))?;

    if server_tools.is_empty() {
        let stream = request.stream.unwrap_or(false);
        root_span.record("model", field::display(&request.model));
        root_span.record("streaming", stream);
        let proxy = Request::builder()
            .method("POST")
            .uri(Endpoint::ChatCompletions.path())
            .body(Body::from(body))
            .context("build model api request")?;
        let mut proxy = proxy;
        *proxy.headers_mut() = headers;
        return Ok(state.model_api.handle(proxy).await);
    }

    let request_model_id = if request.model.trim().is_empty() {
        None
    } else {
        Some(request.model.clone())
    };
    let route = match state.route_strategy.select(
        Endpoint::ChatCompletions,
        request_model_id.as_deref(),
        &metadata,
    )
    {
        Ok(route) => route,
        Err(RouteSelectionError::EndpointNotConfigured) => {
            root_span.record("http.status_code", StatusCode::NOT_FOUND.as_u16() as i64);
            return Ok(openai_error_response(
                StatusCode::NOT_FOUND,
                "endpoint not configured in file",
                Some("invalid_request_error"),
            ));
        }
        Err(RouteSelectionError::ModelNotConfigured) => {
            root_span.record("http.status_code", StatusCode::BAD_REQUEST.as_u16() as i64);
            return Ok(openai_error_response(
                StatusCode::BAD_REQUEST,
                "model not configured for endpoint",
                Some("invalid_request_error"),
            ));
        }
        Err(RouteSelectionError::ModelAmbiguous) => {
            root_span.record("http.status_code", StatusCode::BAD_REQUEST.as_u16() as i64);
            return Ok(openai_error_response(
                StatusCode::BAD_REQUEST,
                "multiple configs match endpoint; include model or add default",
                Some("invalid_request_error"),
            ));
        }
    };

    request.model = route.model.clone();
    let stream = request.stream.unwrap_or(false);
    root_span.record(
        "model",
        field::display(request_model_id.as_deref().unwrap_or(&route.model_id)),
    );
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
        "chat request parsed: model_id={}, model={}, stream={} {}",
        request_model_id
            .as_deref()
            .unwrap_or(&route.model_id),
        route.model,
        stream,
        metadata
    );
    if let Err(message) = validate_openai_request(&request) {
        error!(
            "chat request invalid: model_id={}, error={} {}",
            request_model_id
                .as_deref()
                .unwrap_or(&route.model_id),
            message,
            metadata
        );
        root_span.record("http.status_code", StatusCode::BAD_REQUEST.as_u16() as i64);
        return Ok(openai_error_response(
            StatusCode::BAD_REQUEST,
            &message,
            Some("invalid_request_error"),
        ));
    }

    let provider = match state.agent_providers.get(&route.provider_id) {
        Some(provider) => Arc::clone(provider),
        None => {
            error!(
                "chat selection failed: provider_id={}, model_id={} {}",
                route.provider_id,
                route.model_id,
                metadata
            );
            root_span.record("http.status_code", StatusCode::BAD_REQUEST.as_u16() as i64);
            return Ok(openai_error_response(
                StatusCode::BAD_REQUEST,
                "provider does not support agent loop",
                Some("invalid_request_error"),
            ));
        }
    };

    let model_for_log = route.model_id.clone();
    let request_model = route.model.clone();
    let metadata_for_log = metadata.clone();
    let loop_result = run_agent_loop::<A, M>(
        provider,
        request,
        server_tools,
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
