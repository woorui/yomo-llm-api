use log::{error, info};
use std::net::SocketAddr;
use std::sync::Arc;

use llm_api::agent_loop::{ServerToolInvoker, ServerToolRegistry};
use llm_api::llm_api::{
    AppState, EmptyToolInvoker, EmptyToolRegistry, ToolContext, build_provider_registry,
    run_http_server,
};
use llm_api::mock_get_weather::{WeatherToolInvoker, WeatherToolRegistry};

#[tokio::main]
async fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
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
            Arc::new(WeatherToolRegistry::<ToolContext>::default())
                as Arc<dyn ServerToolRegistry<Ctx = ToolContext>>,
            Arc::new(WeatherToolInvoker::<ToolContext>::default())
                as Arc<dyn ServerToolInvoker<Ctx = ToolContext>>,
        )
    } else {
        (
            Arc::new(EmptyToolRegistry) as Arc<dyn ServerToolRegistry<Ctx = ToolContext>>,
            Arc::new(EmptyToolInvoker) as Arc<dyn ServerToolInvoker<Ctx = ToolContext>>,
        )
    };

    let state = AppState {
        registry: Arc::new(registry),
        tool_registry,
        tool_invoker,
    };

    if let Err(err) = run_http_server(addr, state).await {
        error!("{err}");
    }
}
