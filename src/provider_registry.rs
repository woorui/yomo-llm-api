use std::collections::HashMap;
use std::sync::Arc;

use crate::config::Config;
use crate::config::ConfigError;
use crate::provider::Provider;
use crate::providers::openai::build_openai_provider;

pub type ProviderId = String;

#[derive(Clone)]
pub struct ProviderEntry {
    pub id: String,
    pub provider_type: String,
    pub model: String,
    pub provider: Arc<dyn Provider>,
}

pub struct ProviderRegistry<M> {
    providers: HashMap<ProviderId, ProviderEntry>,
    strategy: Arc<dyn SelectionStrategy<M>>,
    default_provider_id: ProviderId,
}

pub trait SelectionStrategy<M>: Send + Sync {
    fn select(
        &self,
        model: &str,
        metadata: &M,
        registry: &ProviderRegistry<M>,
    ) -> Result<String, anyhow::Error>;
}

#[derive(Default)]
pub struct ByModel;

impl<M> SelectionStrategy<M> for ByModel {
    fn select(
        &self,
        model: &str,
        _metadata: &M,
        registry: &ProviderRegistry<M>,
    ) -> Result<String, anyhow::Error> {
        if model.trim().is_empty() {
            return Err(anyhow::anyhow!("model is required"));
        }

        for entry in registry.providers().values() {
            if entry.model == model {
                return Ok(entry.id.clone());
            }
        }

        Err(anyhow::anyhow!("unsupported model {model}"))
    }
}

pub type StrategyRef<M> = Arc<dyn SelectionStrategy<M>>;

impl<M> ProviderRegistry<M> {
    pub fn from_config(
        config: &Config,
        strategy: Arc<dyn SelectionStrategy<M>>,
    ) -> Result<Self, ConfigError> {
        config.validate()?;

        let mut providers: HashMap<ProviderId, ProviderEntry> = HashMap::new();
        let mut default_provider_id: Option<ProviderId> = None;
        for item in &config.providers {
            if item.default {
                default_provider_id = Some(item.id.clone());
            }
            let provider = match item.provider_type.as_str() {
                "openai" => build_openai_provider(item.model.clone(), &item.params)?,
                other => return Err(ConfigError::UnknownProviderType(other.to_string())),
            };

            let entry = ProviderEntry {
                id: item.id.clone(),
                provider_type: item.provider_type.clone(),
                model: item.model.clone(),
                provider: Arc::new(provider),
            };
            providers.insert(item.id.clone(), entry);
        }

        let default_provider_id = default_provider_id.ok_or_else(|| {
            ConfigError::InvalidProvider("default provider is missing".to_string())
        })?;

        Ok(Self::new(providers, strategy, default_provider_id))
    }

    pub fn new(
        providers: HashMap<ProviderId, ProviderEntry>,
        strategy: Arc<dyn SelectionStrategy<M>>,
        default_provider_id: ProviderId,
    ) -> Self {
        Self {
            providers,
            strategy,
            default_provider_id,
        }
    }

    pub fn select(&self, model: &str, metadata: &M) -> Result<&ProviderEntry, anyhow::Error> {
        let selected = self.strategy.select(model, metadata, self).ok();
        let id = selected
            .as_ref()
            .and_then(|id| self.providers.get_key_value(id).map(|(key, _)| key))
            .unwrap_or(&self.default_provider_id);

        self.providers
            .get(id)
            .ok_or_else(|| anyhow::anyhow!("unsupported model {model}"))
    }

    pub fn providers(&self) -> &HashMap<ProviderId, ProviderEntry> {
        &self.providers
    }
}
