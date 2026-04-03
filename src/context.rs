use axum::http::HeaderMap;

pub trait RequestContext: Send + Sync {
    type Ctx;
    fn build_from_headers(&self, headers: &HeaderMap) -> Self::Ctx;
}

#[derive(Clone, Debug)]
pub struct TraceContext {
    pub trace_id: String,
}

#[derive(Clone, Debug, Default)]
pub struct TraceContextBuilder;

impl RequestContext for TraceContextBuilder {
    type Ctx = TraceContext;

    fn build_from_headers(&self, headers: &HeaderMap) -> Self::Ctx {
        let trace_id = headers
            .get("x-trace-id")
            .and_then(|value| value.to_str().ok())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("")
            .to_string();

        TraceContext { trace_id }
    }
}
