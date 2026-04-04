use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::pin::Pin;

use futures_core::Stream;
use futures_util::{future::join_all, StreamExt};
use serde_json::Value;
use log::debug;

use crate::openai_types::{ChatCompletionRequest, Content, Message, Role, ToolDefinition};
use crate::provider::{
    ChatError, Provider, ToolCall as ProviderToolCall, UnifiedEvent, UnifiedResponse, Usage,
};
use crate::providers::openai::mapper::ensure_tool_call_id;

#[derive(Debug, Clone)]
pub struct ToolError {
    pub code: String,
    pub message: String,
}

impl fmt::Display for ToolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl Error for ToolError {}

pub trait ServerToolRegistry: Send + Sync {
    type Ctx;
    fn list(&self, ctx: &Self::Ctx) -> Vec<ToolDefinition>;
}

pub trait ServerToolInvoker: Send + Sync {
    type Ctx;
    fn invoke(&self, ctx: &Self::Ctx, name: &str, args: Value) -> Result<Value, ToolError>;
}

pub struct AgentLoopConfig {
    pub max_calls: usize,
}

impl Default for AgentLoopConfig {
    fn default() -> Self {
        Self { max_calls: 14 }
    }
}

pub enum AgentLoopResult {
    NonStream(UnifiedResponse),
    Stream {
        events: Pin<Box<dyn Stream<Item = Result<UnifiedEvent, ChatError>> + Send>>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolSource {
    Server,
    Client,
}

struct ToolMaps {
    merged_tools: Option<Vec<ToolDefinition>>,
    source_map: HashMap<String, ToolSource>,
}

pub async fn run_agent_loop<Ctx>(
    provider: std::sync::Arc<dyn Provider>,
    request: ChatCompletionRequest,
    registry: std::sync::Arc<dyn ServerToolRegistry<Ctx = Ctx>>,
    invoker: std::sync::Arc<dyn ServerToolInvoker<Ctx = Ctx>>,
    ctx: Ctx,
    config: AgentLoopConfig,
) -> Result<AgentLoopResult, ChatError>
where
    Ctx: fmt::Display + Send + Sync + 'static,
{
    let ctx = std::sync::Arc::new(ctx);
    if request.stream.unwrap_or(false) {
        run_agent_loop_stream(provider, request, registry, invoker, ctx, config).await
    } else {
        run_agent_loop_nonstream(provider, request, registry, invoker, ctx, config).await
    }
}

async fn run_agent_loop_nonstream<Ctx>(
    provider: std::sync::Arc<dyn Provider>,
    mut request: ChatCompletionRequest,
    registry: std::sync::Arc<dyn ServerToolRegistry<Ctx = Ctx>>,
    invoker: std::sync::Arc<dyn ServerToolInvoker<Ctx = Ctx>>,
    ctx: std::sync::Arc<Ctx>,
    config: AgentLoopConfig,
) -> Result<AgentLoopResult, ChatError>
where
    Ctx: fmt::Display + Send + Sync + 'static,
{
    let mut call_count = 0usize;
    let mut total_usage = Usage {
        input_tokens: 0,
        output_tokens: 0,
        total_tokens: 0,
        cached_tokens: None,
        reasoning_tokens: None,
    };
    loop {
        let tool_maps = build_tool_maps(&request, registry.as_ref(), ctx.as_ref());
        request.tools = tool_maps.merged_tools.clone();
        if call_count > 0 {
            request.tool_choice = None;
        }

        let mut response = provider
            .complete(request.clone())
            .await
            .map_err(|err| ChatError::ProviderErr(err.to_string()))?;
        add_usage(&mut total_usage, &response.usage);
        call_count += 1;
        debug!("llm chat(#{call_count}), usage={:?} {}", response.usage, ctx.as_ref());

        if call_count >= config.max_calls {
            response.usage = total_usage;
            return Ok(AgentLoopResult::NonStream(response));
        }

        let mut tool_calls = response.tool_calls.take().unwrap_or_default();
        ensure_provider_call_ids(&response.request_id, &mut tool_calls);
        if tool_calls.is_empty() {
            response.tool_calls = None;
            response.usage = total_usage;
            return Ok(AgentLoopResult::NonStream(response));
        }

        let (server_calls, client_calls) = split_tool_calls(&tool_calls, &tool_maps.source_map);
        if server_calls.is_empty() {
            response.tool_calls = Some(client_calls);
            response.usage = total_usage;
            return Ok(AgentLoopResult::NonStream(response));
        }
        if !client_calls.is_empty() {
            response.tool_calls = Some(client_calls);
            response.usage = total_usage;
            return Ok(AgentLoopResult::NonStream(response));
        }

        let request_id = response.request_id.clone();
        let mut next_messages = Vec::new();
        next_messages.push(build_assistant_tool_call_message(&request_id, &server_calls));
        next_messages.extend(build_tool_messages::<Ctx>(
            &request_id,
            &server_calls,
            invoker.clone(),
            ctx.clone(),
        )
        .await?);
        request.messages.extend(next_messages);
    }
}

async fn run_agent_loop_stream<Ctx>(
    provider: std::sync::Arc<dyn Provider>,
    mut request: ChatCompletionRequest,
    registry: std::sync::Arc<dyn ServerToolRegistry<Ctx = Ctx>>,
    invoker: std::sync::Arc<dyn ServerToolInvoker<Ctx = Ctx>>,
    ctx: std::sync::Arc<Ctx>,
    config: AgentLoopConfig,
) -> Result<AgentLoopResult, ChatError>
where
    Ctx: fmt::Display + Send + Sync + 'static,
{
    let stream = async_stream::try_stream! {
        let mut call_count = 0usize;
        let mut total_usage = Usage {
            input_tokens: 0,
            output_tokens: 0,
            total_tokens: 0,
            cached_tokens: None,
            reasoning_tokens: None,
        };
        loop {
            let tool_maps = build_tool_maps(&request, registry.as_ref(), ctx.as_ref());
            request.tools = tool_maps.merged_tools.clone();
            if call_count > 0 {
                request.tool_choice = None;
            }

            let mut provider_stream = provider.stream(request.clone());

            let usage_offset = total_usage.clone();

            let mut tool_calls: Vec<ProviderToolCall> = Vec::new();
            let mut tool_call_index: HashMap<String, usize> = HashMap::new();
            let mut usage = None;
            let mut finish_reason = None;
            let mut saw_tool_call = false;
            let mut emitted_client_tool = false;

            while let Some(item) = provider_stream.next().await {
                let event = item?;
                match &event {
                    UnifiedEvent::ToolCallDelta { id, name, arguments_delta } => {
                        saw_tool_call = true;
                        let index = *tool_call_index.entry(id.clone()).or_insert_with(|| {
                            let current = tool_calls.len();
                            tool_calls.push(ProviderToolCall {
                                id: Some(id.clone()),
                                name: name.clone(),
                                description: String::new(),
                                arguments: String::new(),
                            });
                            current
                        });
                        let is_server = matches!(
                            tool_maps
                                .source_map
                                .get(name)
                                .copied()
                                .unwrap_or(ToolSource::Client),
                            ToolSource::Server
                        );
                        if is_server {
                            if let Some(call) = tool_calls.get_mut(index) {
                                call.name = name.clone();
                                call.arguments.push_str(arguments_delta);
                            }
                        } else {
                            if let Some(call) = tool_calls.get_mut(index) {
                                call.name = name.clone();
                                call.arguments.push_str(arguments_delta);
                            }
                            emitted_client_tool = true;
                            yield UnifiedEvent::ToolCallDelta {
                                id: id.clone(),
                                name: name.clone(),
                                arguments_delta: arguments_delta.clone(),
                            };
                        }
                    }
                    UnifiedEvent::ToolCallDone { id, name, arguments } => {
                        saw_tool_call = true;
                        let index = *tool_call_index.entry(id.clone()).or_insert_with(|| {
                            let current = tool_calls.len();
                            tool_calls.push(ProviderToolCall {
                                id: Some(id.clone()),
                                name: name.clone(),
                                description: String::new(),
                                arguments: String::new(),
                            });
                            current
                        });
                        let is_server = matches!(
                            tool_maps
                                .source_map
                                .get(name)
                                .copied()
                                .unwrap_or(ToolSource::Client),
                            ToolSource::Server
                        );
                        if is_server {
                            if let Some(call) = tool_calls.get_mut(index) {
                                call.name = name.clone();
                                call.arguments = arguments.clone();
                            }
                        } else {
                            if let Some(call) = tool_calls.get_mut(index) {
                                call.name = name.clone();
                                call.arguments = arguments.clone();
                            }
                            emitted_client_tool = true;
                            yield UnifiedEvent::ToolCallDone {
                                id: id.clone(),
                                name: name.clone(),
                                arguments: arguments.clone(),
                            };
                        }
                    }
                    UnifiedEvent::Usage { usage: chunk_usage } => {
                        usage = Some(chunk_usage.clone());
                    }
                    UnifiedEvent::Completed { finish_reason: reason, usage: chunk_usage } => {
                        finish_reason = reason.clone();
                        if let Some(chunk_usage) = chunk_usage {
                            usage = Some(chunk_usage.clone());
                        }
                    }
                    _ => {}
                }

                if !saw_tool_call {
                    match event {
                        UnifiedEvent::Usage { usage: chunk_usage } => {
                            let usage = add_usage_cloned(&usage_offset, &chunk_usage);
                            yield UnifiedEvent::Usage { usage };
                        }
                        UnifiedEvent::Completed { finish_reason, usage: chunk_usage } => {
                            let usage = chunk_usage.map(|chunk_usage| {
                                add_usage_cloned(&usage_offset, &chunk_usage)
                            });
                            yield UnifiedEvent::Completed { finish_reason, usage };
                        }
                        _ => {
                            yield event;
                        }
                    }
                }
            }

            call_count += 1;
            if let Some(current_usage) = &usage {
                add_usage(&mut total_usage, current_usage);
                debug!("llm chat(#{call_count}), usage={:?} {}", current_usage, ctx.as_ref());
            }
            if call_count >= config.max_calls {
                if !tool_calls.is_empty() {
                    if !emitted_client_tool {
                        let events = build_client_tool_events(
                            &Some(total_usage.clone()),
                            &finish_reason,
                            &tool_calls,
                        );
                        for event in events {
                            yield event;
                        }
                    } else if let Some(completed) =
                        build_completed_event(&Some(total_usage.clone()), &finish_reason)
                    {
                        yield completed;
                    }
                }
                break;
            }

            if tool_calls.is_empty() {
                break;
            }

            let (server_calls, client_calls) = split_tool_calls(&tool_calls, &tool_maps.source_map);
            if server_calls.is_empty() {
                if !emitted_client_tool {
                    let events = build_client_tool_events(
                        &Some(total_usage.clone()),
                        &finish_reason,
                        &client_calls,
                    );
                    for event in events {
                        yield event;
                    }
                } else if let Some(completed) =
                    build_completed_event(&Some(total_usage.clone()), &finish_reason)
                {
                    yield completed;
                }
                break;
            }
            if !client_calls.is_empty() {
                if !emitted_client_tool {
                    let events = build_client_tool_events(
                        &Some(total_usage.clone()),
                        &finish_reason,
                        &client_calls,
                    );
                    for event in events {
                        yield event;
                    }
                } else if let Some(completed) =
                    build_completed_event(&Some(total_usage.clone()), &finish_reason)
                {
                    yield completed;
                }
                break;
            }

            let request_id = request.model.clone();
            let mut tool_messages = Vec::new();
            tool_messages.push(build_assistant_tool_call_message(&request_id, &server_calls));
            tool_messages.extend(build_tool_messages::<Ctx>(
                &request_id,
                &server_calls,
                invoker.clone(),
                ctx.clone(),
            )
            .await?);
            request.messages.extend(tool_messages);
        }
    };

    let boxed_stream: Pin<Box<dyn Stream<Item = Result<UnifiedEvent, ChatError>> + Send>> =
        Box::pin(stream);
    Ok(AgentLoopResult::Stream {
        events: boxed_stream,
    })
}

fn build_tool_maps<R>(request: &ChatCompletionRequest, registry: &R, ctx: &R::Ctx) -> ToolMaps
where
    R: ServerToolRegistry + ?Sized,
{
    let server_tools = registry.list(ctx);
    let mut source_map = HashMap::new();
    let mut merged = Vec::new();
    let mut server_lookup = HashMap::new();
    let mut server_seen = HashMap::new();

    for tool in server_tools {
        source_map.insert(tool.function.name.clone(), ToolSource::Server);
        server_lookup.insert(tool.function.name.clone(), tool);
    }

    if let Some(request_tools) = &request.tools {
        for tool in request_tools {
            if let Some(server_tool) = server_lookup.get(&tool.function.name) {
                merged.push(server_tool.clone());
                server_seen.insert(tool.function.name.clone(), true);
                source_map.insert(tool.function.name.clone(), ToolSource::Server);
            } else {
                source_map.insert(tool.function.name.clone(), ToolSource::Client);
                merged.push(tool.clone());
            }
        }
    }

    for (name, tool) in server_lookup {
        if !server_seen.contains_key(&name) {
            merged.push(tool);
        }
    }

    ToolMaps {
        merged_tools: if merged.is_empty() {
            None
        } else {
            Some(merged)
        },
        source_map,
    }
}

fn split_tool_calls(
    calls: &[ProviderToolCall],
    source_map: &HashMap<String, ToolSource>,
) -> (Vec<ProviderToolCall>, Vec<ProviderToolCall>) {
    let mut server = Vec::new();
    let mut client = Vec::new();
    for call in calls {
        match source_map
            .get(&call.name)
            .copied()
            .unwrap_or(ToolSource::Client)
        {
            ToolSource::Server => server.push(call.to_owned()),
            ToolSource::Client => client.push(call.to_owned()),
        }
    }
    (server, client)
}

fn ensure_provider_call_ids(request_id: &str, calls: &mut [ProviderToolCall]) {
    for (index, call) in calls.iter_mut().enumerate() {
        if call.id.is_none() {
            call.id = Some(ensure_tool_call_id(request_id, index));
        }
    }
}

async fn build_tool_messages<Ctx>(
    request_id: &str,
    calls: &[ProviderToolCall],
    invoker: std::sync::Arc<dyn ServerToolInvoker<Ctx = Ctx>>,
    ctx: std::sync::Arc<Ctx>,
) -> Result<Vec<Message>, ChatError>
where
    Ctx: Send + Sync + 'static,
{
    let request_id = request_id.to_string();
    let tasks = calls.iter().cloned().enumerate().map(|(index, call)| {
        let invoker = invoker.clone();
        let ctx = ctx.clone();
        let request_id = request_id.clone();
        tokio::task::spawn_blocking(move || {
            let tool_call_id = call
                .id
                .clone()
                .unwrap_or_else(|| ensure_tool_call_id(&request_id, index));
            let args: Value = serde_json::from_str(&call.arguments)
                .map_err(|err| ChatError::InvalidRequest(format!("invalid tool arguments: {err}")))?;
            let result = invoker
                .invoke(ctx.as_ref(), &call.name, args)
                .map_err(|err| ChatError::ProviderErr(err.to_string()))?;
            let content = serde_json::to_string(&result)
                .map_err(|err| ChatError::InvalidResponse(format!("invalid tool result: {err}")))?;
            Ok(Message {
                role: Role::Tool,
                content: Content::Text(content),
                tool_call_id: Some(tool_call_id),
                tool_calls: None,
            })
        })
    });

    let results = join_all(tasks).await;
    let mut messages = Vec::with_capacity(results.len());
    for result in results {
        let message = result
            .map_err(|err| ChatError::ProviderErr(format!("tool task join error: {err}")))??;
        messages.push(message);
    }
    Ok(messages)
}

fn build_assistant_tool_call_message(
    request_id: &str,
    calls: &[ProviderToolCall],
) -> Message {
    let tool_calls = calls
        .iter()
        .enumerate()
        .map(|(index, call)| crate::openai_types::ToolCall {
            id: call.id.clone().or_else(|| Some(ensure_tool_call_id(request_id, index))),
            r#type: Some("function".to_string()),
            function: crate::openai_types::ToolCallFunction {
                name: call.name.clone(),
                arguments: call.arguments.clone(),
                description: Some(call.description.clone()),
            },
        })
        .collect();

    Message {
        role: Role::Assistant,
        content: Content::Text("Tool call".to_string()),
        tool_call_id: None,
        tool_calls: Some(tool_calls),
    }
}

fn build_client_tool_events(
    usage: &Option<Usage>,
    finish_reason: &Option<String>,
    calls: &[ProviderToolCall],
) -> Vec<UnifiedEvent> {
    let usage = usage.clone().unwrap_or(Usage {
        input_tokens: 0,
        output_tokens: 0,
        total_tokens: 0,
        cached_tokens: None,
        reasoning_tokens: None,
    });
    let finish_reason = finish_reason
        .clone()
        .or_else(|| Some("tool_calls".to_string()));

    let mut events = Vec::new();
    for call in calls {
        events.push(UnifiedEvent::ToolCallDone {
            id: call.id.clone().unwrap_or_default(),
            name: call.name.clone(),
            arguments: call.arguments.clone(),
        });
    }
    events.push(UnifiedEvent::Completed {
        finish_reason,
        usage: Some(usage),
    });
    events
}

fn build_completed_event(
    usage: &Option<Usage>,
    finish_reason: &Option<String>,
) -> Option<UnifiedEvent> {
    let usage = usage.clone().unwrap_or(Usage {
        input_tokens: 0,
        output_tokens: 0,
        total_tokens: 0,
        cached_tokens: None,
        reasoning_tokens: None,
    });
    let finish_reason = finish_reason
        .clone()
        .or_else(|| Some("tool_calls".to_string()));
    Some(UnifiedEvent::Completed {
        finish_reason,
        usage: Some(usage),
    })
}

fn add_usage(total: &mut Usage, delta: &Usage) {
    total.input_tokens += delta.input_tokens;
    total.output_tokens += delta.output_tokens;
    total.total_tokens += delta.total_tokens;
    if total.cached_tokens.is_some() || delta.cached_tokens.is_some() {
        total.cached_tokens =
            Some(total.cached_tokens.unwrap_or(0) + delta.cached_tokens.unwrap_or(0));
    }
    if total.reasoning_tokens.is_some() || delta.reasoning_tokens.is_some() {
        total.reasoning_tokens =
            Some(total.reasoning_tokens.unwrap_or(0) + delta.reasoning_tokens.unwrap_or(0));
    }
}

fn add_usage_cloned(total: &Usage, delta: &Usage) -> Usage {
    let mut usage = total.clone();
    add_usage(&mut usage, delta);
    usage
}
