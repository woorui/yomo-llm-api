use anyhow::Context;
use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::{State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use log::{error, info};
use std::net::SocketAddr;
use std::sync::Arc;

use crate::context::{RequestContext, TraceContext};
use crate::agent_loop::{
    AgentLoopConfig, AgentLoopResult, ServerToolInvoker, ServerToolRegistry, ToolError,
    run_agent_loop,
};
use crate::openai_http_mapping::{
    map_chat_error, map_openai_response, openai_error_response, stream_openai_chunks,
    validate_openai_request,
};
use crate::openai_types::ChatCompletionRequest;
use crate::provider_registry::ProviderRegistry;
use crate::provider_registry::ByModel;

#[derive(Clone)]
pub struct AppState<Ctx> {
    pub registry: Arc<ProviderRegistry>,
    pub tool_registry: Arc<dyn ServerToolRegistry<Ctx = Ctx>>,
    pub tool_invoker: Arc<dyn ServerToolInvoker<Ctx = Ctx>>,
    pub request_context_builder: Arc<dyn RequestContext<Ctx = Ctx>>,
}


pub struct EmptyToolRegistry;

impl ServerToolRegistry for EmptyToolRegistry {
    type Ctx = TraceContext;

    fn list(&self, _ctx: &Self::Ctx) -> Vec<crate::openai_types::ToolDefinition> {
        Vec::new()
    }
}

pub struct EmptyToolInvoker;

impl ServerToolInvoker for EmptyToolInvoker {
    type Ctx = TraceContext;

    fn invoke(
        &self,
        _ctx: &Self::Ctx,
        name: &str,
        _args: serde_json::Value,
    ) -> Result<serde_json::Value, ToolError> {
        Err(ToolError {
            code: "unsupported_tool".to_string(),
            message: format!("no server tool invoker configured for {name}"),
        })
    }
}

pub async fn run_http_server(
    addr: SocketAddr,
    state: AppState<TraceContext>,
) -> Result<(), anyhow::Error> {
    let app = Router::new()
        .route("/v1/chat/completions", post(handle_chat_completions))
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

pub fn build_provider_registry(config_path: &str) -> Result<ProviderRegistry, anyhow::Error> {
    let config = crate::config::Config::load(config_path)
        .and_then(|config| ProviderRegistry::from_config(&config, Arc::new(ByModel)))
        .map_err(|err| anyhow::anyhow!("failed to load config {config_path}: {err}"))?;
    Ok(config)
}

async fn handle_chat_completions(
    State(state): State<AppState<TraceContext>>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    let ctx = state.request_context_builder.build_from_headers(&headers);

    info!("chat request received {}", ctx);
    match handle_chat_completions_inner(state, ctx, body).await {
        Ok(response) => response,
        Err(err) => {
            error!("chat completion failed: {err} {}", ctx);
            openai_error_response(StatusCode::INTERNAL_SERVER_ERROR, "internal error", None)
        }
    }
}

async fn handle_chat_completions_inner(
    state: AppState<TraceContext>,
    ctx: TraceContext,
    body: Bytes,
) -> Result<Response, anyhow::Error> {
    let mut request: ChatCompletionRequest =
        serde_json::from_slice(&body).context("invalid json body")?;
    let stream = request.stream.unwrap_or(false);
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
        request.model, stream, ctx
    );
    if let Err(message) = validate_openai_request(&request) {
        error!(
            "chat request invalid: model={}, error={} {}",
            request.model, message, ctx
        );
        return Ok(openai_error_response(
            StatusCode::BAD_REQUEST,
            &message,
            Some("invalid_request_error"),
        ));
    }

    let provider = match state.registry.select(&request.model, &ctx) {
        Ok(entry) => Arc::clone(&entry.provider),
        Err(err) => {
            error!(
                "chat selection failed: model={}, error={} {}",
                request.model, err, ctx
            );
            return Ok(openai_error_response(
                StatusCode::BAD_REQUEST,
                &err.to_string(),
                Some("invalid_request_error"),
            ));
        }
    };

    let model_for_log = request.model.clone();
    let ctx_for_log = ctx.clone();
    let loop_result = run_agent_loop(
        provider,
        request,
        Arc::clone(&state.tool_registry),
        Arc::clone(&state.tool_invoker),
        ctx,
        AgentLoopConfig::default(),
    )
    .await;

    match loop_result {
        Ok(AgentLoopResult::NonStream(response)) => {
            info!("chat request success: model={} {}", response.model, ctx_for_log);
            let mapped = map_openai_response(response);
            let payload = serde_json::to_vec(&mapped).context("serialize response")?;
            Ok(Response::builder()
                .status(StatusCode::OK)
                .body(Body::from(payload))
                .expect("build response"))
        }
        Ok(AgentLoopResult::Stream { events }) => {
            let sse = stream_openai_chunks(events, ctx_for_log.trace_id.clone());
            let body = Body::from_stream(sse);
            Ok(Response::builder()
                .status(StatusCode::OK)
                .body(body)
                .expect("build response"))
        }
        Err(err) => {
            error!(
                "chat request failed: model={}, error={} {}",
                model_for_log, err, ctx_for_log
            );
            Ok(map_chat_error(err))
        }
    }
}
