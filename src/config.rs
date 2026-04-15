use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;
use std::path::Path;

use crate::endpoint::Endpoint;

#[derive(Debug)]
pub enum ConfigError {
    Load(String),
    InvalidProvider(String),
    UnknownProviderType(String),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::Load(message) => write!(f, "config load error: {message}"),
            ConfigError::InvalidProvider(message) => write!(f, "invalid provider: {message}"),
            ConfigError::UnknownProviderType(name) => {
                write!(f, "unknown provider type: {name}")
            }
        }
    }
}

impl Error for ConfigError {}
#[derive(Debug, Deserialize)]
pub struct Config {
    pub providers: Vec<ProviderConfig>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ProviderConfig {
    pub id: String,
    #[serde(rename = "type")]
    pub provider_type: String,
    pub model_id: String,
    #[serde(default)]
    pub default: bool,
    #[serde(default)]
    pub endpoints: Vec<String>,
    #[serde(default)]
    pub params: HashMap<String, String>,
}

impl Config {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let config = config::Config::builder()
            .add_source(config::File::from(path))
            .build()
            .map_err(|err| ConfigError::Load(err.to_string()))?;
        let parsed: Self = config
            .try_deserialize()
            .map_err(|err| ConfigError::Load(err.to_string()))?;
        Ok(parsed)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.providers.is_empty() {
            return Err(ConfigError::InvalidProvider(
                "providers list is empty".to_string(),
            ));
        }

        let mut ids = std::collections::HashSet::new();
        let mut model_ids = HashSet::new();
        let mut endpoint_defaults: HashMap<String, String> = HashMap::new();
        let mut endpoint_counts: HashMap<String, usize> = HashMap::new();
        for provider in &self.providers {
            if provider.id.trim().is_empty() {
                return Err(ConfigError::InvalidProvider("id is required".to_string()));
            }
            if !ids.insert(provider.id.clone()) {
                return Err(ConfigError::InvalidProvider(format!(
                    "duplicate provider id: {}",
                    provider.id
                )));
            }
            if provider.provider_type.trim().is_empty() {
                return Err(ConfigError::InvalidProvider(format!(
                    "provider type is required for {}",
                    provider.id
                )));
            }
            if provider.model_id.trim().is_empty() {
                return Err(ConfigError::InvalidProvider(format!(
                    "model_id is required for {}",
                    provider.id
                )));
            }
            let normalized_model_id = provider.model_id.to_ascii_lowercase();
            if !model_ids.insert(normalized_model_id) {
                return Err(ConfigError::InvalidProvider(format!(
                    "duplicate model_id: {}",
                    provider.model_id
                )));
            }
            if provider.endpoints.is_empty() {
                return Err(ConfigError::InvalidProvider(format!(
                    "endpoints is required for {}",
                    provider.id
                )));
            }
            for endpoint in &provider.endpoints {
                if Endpoint::from_key(endpoint).is_none() {
                    return Err(ConfigError::InvalidProvider(format!(
                        "unknown endpoint key: {endpoint}"
                    )));
                }
                *endpoint_counts.entry(endpoint.clone()).or_insert(0) += 1;
                if provider.default {
                    if let Some(existing) = endpoint_defaults.get(endpoint) {
                        return Err(ConfigError::InvalidProvider(format!(
                            "endpoint {endpoint} has multiple default providers: {existing}, {}",
                            provider.id
                        )));
                    }
                    endpoint_defaults.insert(endpoint.clone(), provider.id.clone());
                }
            }
        }

        for (endpoint, count) in endpoint_counts {
            if count > 1 && !endpoint_defaults.contains_key(&endpoint) {
                return Err(ConfigError::InvalidProvider(format!(
                    "endpoint {endpoint} must have a default provider"
                )));
            }
        }

        Ok(())
    }
}
