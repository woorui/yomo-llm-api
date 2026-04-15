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

use llm_api::tool_invoker::ToolInvoker;
use llm_api::metadata::{Metadata, MetadataBuilder};
use llm_api::tool_invoker::ConnToolInvoker;
use llm_api::llm_api::{
    DefaultHeaderExtractor, LlmApiState, run_http_server,
};
use llm_api::config::Config;
use llm_api::model_api::ModelApi;
use llm_api::providers::openai::build_openai_provider;
use llm_api::provider_registry::ByModel;
use llm_api::mock_get_weather::{WeatherToolInvoker, WeatherToolMgr};
use yomo::auth::AuthImpl;
use yomo::tool_mgr::ToolMgrImpl;

#[tokio::main]
async fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    if let Err(err) = init_tracing() {
        error!("otel tracing init failed: {err}");
    }
    let config_path = "config.toml".to_string();
    let port = "8000".to_string();
    let addr: SocketAddr = format!("0.0.0.0:{port}").parse().expect("invalid PORT");

    let config = match Config::load(&config_path).and_then(|config| {
        config.validate()?;
        Ok(config)
    }) {
        Ok(config) => config,
        Err(err) => {
            error!("failed to load config {config_path}: {err}");
            return;
        }
    };
    let providers = config
        .providers
        .into_iter()
        .map(|provider| (provider.id.clone(), provider))
        .collect::<std::collections::HashMap<_, _>>();
    let mut agent_providers = std::collections::HashMap::new();
    for (id, provider) in &providers {
        if provider.provider_type != "openai" {
            continue;
        }
        match build_openai_provider(&provider.params) {
            Ok(built) => {
                agent_providers.insert(id.clone(), Arc::new(built) as Arc<dyn llm_api::provider::Provider>);
            }
            Err(err) => {
                error!("failed to build provider {id}: {err}");
                return;
            }
        }
    }
    let model_api = match ModelApi::new(&providers) {
        Ok(model_api) => model_api,
        Err(err) => {
            error!("failed to init endpoint proxy: {err}");
            return;
        }
    };
    let selection_strategy = Arc::new(ByModel::new(providers.clone()));

    let enable_weather_tool = std::env::var("ENABLE_MOCK_GET_WEATHER")
        .ok()
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE"))
        .unwrap_or(false);

    let (tool_mgr, tool_invoker) = if enable_weather_tool {
        info!("weather tool enabled via ENABLE_MOCK_GET_WEATHER");
        (
            Arc::new(WeatherToolMgr::<Metadata>::default())
                as Arc<dyn yomo::tool_mgr::ToolMgr<(), Metadata>>,
            Arc::new(WeatherToolInvoker::<Metadata>::default())
                as Arc<dyn ToolInvoker<Metadata>>,
        )
    } else {
        let (tool_tx, _tool_rx) = tokio::sync::mpsc::unbounded_channel();
        let connector = yomo::connector::MemoryConnector::new(tool_tx, 64 * 1024);
        (
            Arc::new(ToolMgrImpl::new()) as Arc<dyn yomo::tool_mgr::ToolMgr<(), Metadata>>,
            Arc::new(ConnToolInvoker::<Metadata, _, _, _>::new(Arc::new(connector)))
                as Arc<dyn ToolInvoker<Metadata>>,
        )
    };

    let state = LlmApiState {
        providers: Arc::new(providers),
        agent_providers: Arc::new(agent_providers),
        model_api: Arc::new(model_api),
        selection_strategy,
        tool_mgr,
        tool_invoker,
        metadata_mgr: Arc::new(MetadataBuilder::default())
            as Arc<dyn yomo::metadata_mgr::MetadataMgr<(), Metadata>>,
        auth: Arc::new(AuthImpl::new(None)),
        header_extractor: Arc::new(DefaultHeaderExtractor::default()),
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
