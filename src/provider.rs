use futures_core::Stream;
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::pin::Pin;

use crate::openai_types::ChatCompletionRequest;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: i32,
    pub output_tokens: i32,
    pub total_tokens: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub name: String,
    pub description: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedResponse {
    pub request_id: String,
    pub model: String,
    pub output_text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    pub finish_reason: FinishReason,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UnifiedEvent {
    ResponseCreated {
        id: String,
        model: String,
        created_at: String,
    },
    ResponseInProgress {
        id: String,
        model: String,
        created_at: String,
    },
    OutputItemAdded {
        id: String,
        item_type: String,
    },
    OutputItemDone {
        id: String,
        item_type: String,
    },
    ContentPartAdded {
        item_id: String,
        part_type: String,
    },
    ContentPartDelta {
        item_id: String,
        part_type: String,
        delta: String,
    },
    ContentPartDone {
        item_id: String,
        part_type: String,
    },
    ThinkingDelta {
        id: String,
        delta: String,
    },
    ThinkingDone {
        id: String,
        summary: Option<String>,
    },
    ToolCallDelta {
        id: String,
        name: String,
        arguments_delta: String,
    },
    ToolCallDone {
        id: String,
        name: String,
        arguments: String,
    },
    MessageStart {
        id: String,
        role: String,
    },
    MessageDelta {
        id: String,
        delta: String,
    },
    MessageStop {
        id: String,
        stop_reason: Option<String>,
    },
    Usage {
        usage: Usage,
    },
    Completed {
        finish_reason: Option<String>,
        usage: Option<Usage>,
    },
    Failed {
        code: String,
        message: String,
    },
    Cancelled {
        reason: String,
    },
}

#[derive(Debug)]
pub enum AgentError {
    InvalidRequest(String),
    InvalidResponse(String),
    ProviderErr(String),
}

impl std::fmt::Display for AgentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentError::InvalidRequest(message) => write!(f, "invalid request: {message}"),
            AgentError::InvalidResponse(message) => write!(f, "invalid response: {message}"),
            AgentError::ProviderErr(message) => write!(f, "provider error: {message}"),
        }
    }
}

impl std::error::Error for AgentError {}

pub trait Provider: Send + Sync {
    fn model(&self) -> &str;

    fn complete<'a>(
        &'a self,
        request: ChatCompletionRequest,
    ) -> Pin<Box<dyn Future<Output = Result<UnifiedResponse, AgentError>> + Send + 'a>>;

    fn stream<'a>(
        &'a self,
        request: ChatCompletionRequest,
    ) -> Pin<Box<dyn Stream<Item = Result<UnifiedEvent, AgentError>> + Send + 'a>>;
}
