use serde_json::json;
use std::marker::PhantomData;

use crate::agent_loop::{ServerToolInvoker, ServerToolRegistry, ToolError};
use crate::openai_types::{FunctionDefinition, ToolDefinition};

#[derive(Clone)]
pub struct WeatherToolRegistry<Ctx> {
    _marker: PhantomData<Ctx>,
}

impl<Ctx> Default for WeatherToolRegistry<Ctx> {
    fn default() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<Ctx: Send + Sync> ServerToolRegistry for WeatherToolRegistry<Ctx> {
    type Ctx = Ctx;

    fn list(&self, _ctx: &Self::Ctx) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            r#type: "function".to_string(),
            function: FunctionDefinition {
                name: "get_weather".to_string(),
                description: Some("Get weather by city".to_string()),
                strict: None,
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }),
            },
        }]
    }
}

#[derive(Clone)]
pub struct WeatherToolInvoker<Ctx> {
    _marker: PhantomData<Ctx>,
}

impl<Ctx> Default for WeatherToolInvoker<Ctx> {
    fn default() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<Ctx: Send + Sync> ServerToolInvoker for WeatherToolInvoker<Ctx> {
    type Ctx = Ctx;

    fn invoke(
        &self,
        _ctx: &Self::Ctx,
        name: &str,
        args: serde_json::Value,
    ) -> Result<serde_json::Value, ToolError> {
        if name != "get_weather" {
            return Err(ToolError {
                code: "unknown_tool".to_string(),
                message: format!("unsupported tool: {name}"),
            });
        }

        let location = args
            .get("location")
            .and_then(|value| value.as_str())
            .unwrap_or("unknown")
            .to_string();

        Ok(json!({
            "location": location,
            "temperature_c": 21,
            "condition": "sunny"
        }))
    }
}
