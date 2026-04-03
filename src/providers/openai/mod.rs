use async_stream::try_stream;
use futures_core::Stream;
use futures_util::StreamExt;
use std::pin::Pin;

use crate::error::ConfigError;
use crate::openai_client;
use crate::openai_http_mapping::validate_openai_request;
use crate::openai_types::ChatCompletionRequest;
use crate::provider::{ChatError, Provider, UnifiedEvent, UnifiedResponse};

pub mod mapper;

#[derive(Clone)]
pub struct OpenAIProvider {
    client: openai_client::Client,
    model: String,
}

impl OpenAIProvider {
    pub fn new(client: openai_client::Client, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }
}

impl Provider for OpenAIProvider {
    fn model(&self) -> &str {
        &self.model
    }

    fn complete<'a>(
        &'a self,
        request: ChatCompletionRequest,
    ) -> Pin<Box<dyn futures_core::Future<Output = Result<UnifiedResponse, ChatError>> + Send + 'a>>
    {
        Box::pin(async move {
            validate_request(&request)?;
            ensure_model_matches(&self.model, &request.model)?;
            let response = self
                .client
                .chat_completions(request)
                .await
                .map_err(|err| ChatError::ProviderErr(err.to_string()))?;

            mapper::map_response(response)
        })
    }

    fn stream<'a>(
        &'a self,
        request: ChatCompletionRequest,
    ) -> Pin<Box<dyn Stream<Item = Result<UnifiedEvent, ChatError>> + Send + 'a>> {
        Box::pin(try_stream! {
            validate_request(&request)?;
            ensure_model_matches(&self.model, &request.model)?;
            let stream = self
                .client
                .chat_completions_stream(request)
                .await
                .map_err(|err| ChatError::ProviderErr(err.to_string()))?;
            futures_util::pin_mut!(stream);
            let mut state = mapper::StreamMapState::default();

            while let Some(item) = stream.next().await {
                let chunk = item.map_err(|err| ChatError::ProviderErr(err.to_string()))?;
                for event in mapper::map_stream_chunk(chunk, &mut state) {
                    yield event;
                }
            }
        })
    }
}

pub fn build_openai_provider(
    model: String,
    params: &std::collections::HashMap<String, String>,
) -> Result<OpenAIProvider, ConfigError> {
    let api_key = params
        .get("api_key")
        .ok_or_else(|| ConfigError::InvalidProvider("api_key is required".to_string()))?;
    let mut config = openai_client::Config::new(api_key.to_string());
    if let Some(base_url) = params.get("base_url") {
        config = config.base_url(base_url.to_string());
    }
    let client = openai_client::Client::new(config)
        .map_err(|err| ConfigError::InvalidProvider(err.to_string()))?;
    Ok(OpenAIProvider::new(client, model))
}

fn validate_request(request: &ChatCompletionRequest) -> Result<(), ChatError> {
    validate_openai_request(request)
        .map_err(|message| ChatError::InvalidRequest(message))
}

fn ensure_model_matches(model: &str, request_model: &str) -> Result<(), ChatError> {
    if model != request_model {
        return Err(ChatError::InvalidRequest(format!(
            "request model {request_model} does not match provider model {model}"
        )));
    }
    Ok(())
}
