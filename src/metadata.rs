use std::fmt;

use anyhow::Result;
use yomo::metadata_mgr::MetadataMgr;

#[derive(Clone, Debug)]
pub struct Metadata {
    pub trace_id: String,
    pub extension: String,
}

impl fmt::Display for Metadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "trace_id={}, extension={}",
            self.trace_id, self.extension
        )
    }
}

#[derive(Clone, Debug, Default)]
pub struct MetadataBuilder;

impl MetadataMgr<(), Metadata> for MetadataBuilder {
    fn new_from_extension(&self, _auth_info: &(), _extension: &str) -> Result<Metadata> {
        Ok(Metadata {
            trace_id: String::new(),
            extension: _extension.to_string(),
        })
    }
}
