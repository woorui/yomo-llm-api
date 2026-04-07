use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::Context;
use log::{error, info};
use opentelemetry::global;
use opentelemetry::KeyValue;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::{Resource, trace as sdktrace};
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::runtime::Tokio;
use tracing::subscriber::set_global_default;
use tracing_subscriber::prelude::*;
use tracing_subscriber::EnvFilter;

use llm_api::agent_loop::{ServerToolInvoker, ServerToolRegistry};
use llm_api::context::{RequestContext, TraceContext, TraceContextBuilder};
use llm_api::llm_api::{
    AppState, EmptyToolInvoker, EmptyToolRegistry, build_provider_registry, run_http_server,
};
use llm_api::mock_get_weather::{WeatherToolInvoker, WeatherToolRegistry};

#[tokio::main]
async fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    if let Err(err) = init_tracing() {
        error!("otel tracing init failed: {err}");
    }
    let config_path = "config.toml".to_string();
    let port = "8000".to_string();
    let addr: SocketAddr = format!("0.0.0.0:{port}").parse().expect("invalid PORT");

    let registry = match build_provider_registry(&config_path) {
        Ok(registry) => registry,
        Err(err) => {
            error!("{err}");
            return;
        }
    };

    let enable_weather_tool = std::env::var("ENABLE_MOCK_GET_WEATHER")
        .ok()
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE"))
        .unwrap_or(false);

    let (tool_registry, tool_invoker) = if enable_weather_tool {
        info!("weather tool enabled via ENABLE_MOCK_GET_WEATHER");
        (
            Arc::new(WeatherToolRegistry::<TraceContext>::default())
                as Arc<dyn ServerToolRegistry<Ctx = TraceContext>>,
            Arc::new(WeatherToolInvoker::<TraceContext>::default())
                as Arc<dyn ServerToolInvoker<Ctx = TraceContext>>,
        )
    } else {
        (
            Arc::new(EmptyToolRegistry) as Arc<dyn ServerToolRegistry<Ctx = TraceContext>>,
            Arc::new(EmptyToolInvoker) as Arc<dyn ServerToolInvoker<Ctx = TraceContext>>,
        )
    };

    let state = AppState {
        registry: Arc::new(registry),
        tool_registry,
        tool_invoker,
        request_context_builder: Arc::new(TraceContextBuilder::default())
            as Arc<dyn RequestContext<Ctx = TraceContext>>,
    };

    if let Err(err) = run_http_server(addr, state).await {
        error!("{err}");
    }
}

fn init_tracing() -> Result<(), anyhow::Error> {
    let endpoint = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT")
        .unwrap_or_else(|_| "http://localhost:4318".to_string());
    let service_name =
        std::env::var("OTEL_SERVICE_NAME").unwrap_or_else(|_| "llm_api".to_string());
    global::set_text_map_propagator(TraceContextPropagator::new());
    let exporter = opentelemetry_otlp::new_exporter().http().with_endpoint(endpoint);
    let tracer = opentelemetry_otlp::new_pipeline()
        .tracing()
        .with_exporter(exporter)
        .with_trace_config(
            sdktrace::config().with_resource(Resource::new(vec![KeyValue::new(
                "service.name",
                service_name,
            )])),
        )
        .install_batch(Tokio)
        .context("init otlp tracing")?;
    let otel_layer = tracing_opentelemetry::layer()
        .with_tracer(tracer)
        .with_location(false)
        .with_threads(false);
    let subscriber = tracing_subscriber::registry()
        .with(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with(otel_layer);
    set_global_default(subscriber).context("set tracing subscriber")?;
    Ok(())
}
