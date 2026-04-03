use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

use crate::error::ConfigError;
#[derive(Debug, Deserialize)]
pub struct Config {
    pub providers: Vec<ProviderConfig>,
}

#[derive(Debug, Deserialize)]
pub struct ProviderConfig {
    pub id: String,
    #[serde(rename = "type")]
    pub provider_type: String,
    pub model: String,
    #[serde(default)]
    pub default: bool,
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
        let mut default_count = 0;
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
            if provider.model.trim().is_empty() {
                return Err(ConfigError::InvalidProvider(format!(
                    "model is required for {}",
                    provider.id
                )));
            }
            if provider.default {
                default_count += 1;
            }
        }

        if default_count == 0 {
            return Err(ConfigError::InvalidProvider(
                "default provider is required".to_string(),
            ));
        }
        if default_count > 1 {
            return Err(ConfigError::InvalidProvider(
                "multiple default providers are not allowed".to_string(),
            ));
        }

        Ok(())
    }
}
