use axum::http::HeaderMap;
use std::fmt;

pub trait RequestContext: Send + Sync {
    type Ctx;
    fn build_from_headers(&self, headers: &HeaderMap) -> Self::Ctx;
}

#[derive(Clone, Debug)]
pub struct TraceContext {
    pub trace_id: String,
}

impl fmt::Display for TraceContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "trace_id={}", self.trace_id)
    }
}

#[derive(Clone, Debug, Default)]
pub struct TraceContextBuilder;

impl RequestContext for TraceContextBuilder {
    type Ctx = TraceContext;

    fn build_from_headers(&self, headers: &HeaderMap) -> Self::Ctx {
        let _ = headers;
        TraceContext {
            trace_id: String::new(),
        }
    }
}
