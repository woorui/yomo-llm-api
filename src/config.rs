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
    #[serde(default)]
    pub endpoint: HashMap<String, EndpointRouteSet>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ProviderConfig {
    pub id: String,
    #[serde(rename = "type")]
    pub provider_type: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub default: bool,
    #[serde(default)]
    pub params: HashMap<String, String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct EndpointRouteSet {
    pub routes: Vec<EndpointRoute>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct EndpointRoute {
    pub model_id: String,
    pub model: String,
    pub provider_id: String,
    #[serde(default)]
    pub default: bool,
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
        }

        self.validate_endpoint_routes()?;

        Ok(())
    }

    pub fn endpoint_map(&self) -> Result<HashMap<Endpoint, EndpointRouteSet>, ConfigError> {
        let mut map = HashMap::new();
        for (key, routes) in &self.endpoint {
            let endpoint = Endpoint::from_key(key).ok_or_else(|| {
                ConfigError::InvalidProvider(format!("unknown endpoint key: {key}"))
            })?;
            map.insert(endpoint, routes.clone());
        }
        Ok(map)
    }

    fn validate_endpoint_routes(&self) -> Result<(), ConfigError> {
        let mut models: HashMap<String, String> = HashMap::new();

        for (key, routes) in &self.endpoint {
            if Endpoint::from_key(key).is_none() {
                return Err(ConfigError::InvalidProvider(format!(
                    "unknown endpoint key: {key}"
                )));
            }
            if routes.routes.is_empty() {
                return Err(ConfigError::InvalidProvider(format!(
                    "endpoint {key} routes is empty"
                )));
            }
            let mut model_ids = HashSet::new();
            let mut default_count = 0;
            for route in &routes.routes {
                if route.model_id.trim().is_empty() {
                    return Err(ConfigError::InvalidProvider(format!(
                        "endpoint {key} model_id is required"
                    )));
                }
                if route.model.trim().is_empty() {
                    return Err(ConfigError::InvalidProvider(format!(
                        "endpoint {key} model is required"
                    )));
                }
                if route.provider_id.trim().is_empty() {
                    return Err(ConfigError::InvalidProvider(format!(
                        "endpoint {key} provider_id is required"
                    )));
                }
                let normalized_model_id = route.model_id.to_ascii_lowercase();
                if !model_ids.insert(normalized_model_id.clone()) {
                    return Err(ConfigError::InvalidProvider(format!(
                        "endpoint {key} duplicate model_id: {}",
                        route.model_id
                    )));
                }
                let normalized_model = route.model.to_ascii_lowercase();
                if let Some(existing) = models.get(&normalized_model) {
                    if existing != &normalized_model_id {
                        return Err(ConfigError::InvalidProvider(format!(
                            "model {} is mapped by multiple model_id",
                            route.model
                        )));
                    }
                } else {
                    models.insert(normalized_model, normalized_model_id.clone());
                }
                if route.default {
                    default_count += 1;
                }
                if !self
                    .providers
                    .iter()
                    .any(|provider| provider.id == route.provider_id)
                {
                    return Err(ConfigError::InvalidProvider(format!(
                        "endpoint {key} references unknown provider_id {}",
                        route.provider_id
                    )));
                }
            }
            if routes.routes.len() > 1 && default_count != 1 {
                return Err(ConfigError::InvalidProvider(format!(
                    "endpoint {key} must have exactly one default route"
                )));
            }
        }
        Ok(())
    }
}
