use std::collections::HashMap;
use serde_json::json;
use std::marker::PhantomData;

use anyhow::Result;
use async_trait::async_trait;
use crate::tool_invoker::ToolInvoker;
use yomo::types::{RequestHeaders, ToolRequest, ToolResponse};
use yomo::tool_mgr::ToolMgr;

#[derive(Clone)]
pub struct WeatherToolMgr<Ctx> {
    _marker: PhantomData<Ctx>,
}

impl<Ctx> Default for WeatherToolMgr<Ctx> {
    fn default() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

#[async_trait]
impl<Ctx: Send + Sync> ToolMgr<(), Ctx> for WeatherToolMgr<Ctx> {
    async fn upsert_tool(&self, _tool_name: String, _schema: String, _auth_info: &()) -> Result<()> {
        Ok(())
    }

    async fn list_tools(&self, _metadata: &Ctx) -> Result<HashMap<String, String>> {
        let schema = json!({
            "description": "Get weather by city",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        })
        .to_string();
        Ok(HashMap::from([("get_weather".to_string(), schema)]))
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

#[async_trait]
impl<Metadata: Send + Sync> ToolInvoker<Metadata> for WeatherToolInvoker<Metadata> {
    async fn invoke(
        &self,
        _metadata: &Metadata,
        headers: RequestHeaders,
        request: ToolRequest,
    ) -> ToolResponse {
        if headers.name != "get_weather" {
            return ToolResponse {
                result: None,
                error_msg: Some(format!("unsupported tool: {}", headers.name)),
            };
        }

        let args: serde_json::Value = serde_json::from_str(&request.args).unwrap_or_default();
        let location = args
            .get("location")
            .and_then(|value| value.as_str())
            .unwrap_or("unknown")
            .to_string();

        let result = json!({
            "location": location,
            "temperature_c": 21,
            "condition": "sunny"
        })
        .to_string();

        ToolResponse {
            result: Some(result),
            error_msg: None,
        }
    }
}
