use std::collections::HashMap;
use std::sync::Arc;

use crate::config::Config;
use crate::config::ConfigError;
use crate::config::ProviderConfig;
use crate::endpoint::Endpoint;
use crate::provider::Provider;
use crate::providers::openai::build_openai_provider;

pub type ProviderId = String;

#[derive(Clone)]
pub struct ProviderEntry {
    pub id: String,
    pub provider_type: String,
    pub model_id: String,
    pub endpoints: Vec<String>,
    pub provider: Arc<dyn Provider>,
}

pub struct ProviderRegistry<M> {
    providers: HashMap<ProviderId, ProviderEntry>,
    strategy: Arc<dyn SelectionStrategy<M>>,
}

#[derive(Clone, Debug)]
pub struct SelectionResult {
    pub model_id: String,
    pub provider_id: String,
}

#[derive(Debug)]
pub enum SelectionError {
    EndpointNotConfigured,
    ModelNotConfigured,
    ModelAmbiguous,
}

pub trait SelectionStrategy<M>: Send + Sync {
    fn select(
        &self,
        endpoint: Endpoint,
        model_id: Option<&str>,
        metadata: &M,
    ) -> Result<SelectionResult, SelectionError>;
}

#[derive(Clone)]
pub struct ByModel {
    providers: HashMap<String, ProviderConfig>,
}

impl ByModel {
    pub fn new(providers: HashMap<String, ProviderConfig>) -> Self {
        Self { providers }
    }
}

impl<M> SelectionStrategy<M> for ByModel {
    fn select(
        &self,
        endpoint: Endpoint,
        model_id: Option<&str>,
        _metadata: &M,
    ) -> Result<SelectionResult, SelectionError> {
        let endpoint_key = endpoint.key();
        let mut candidates = Vec::new();
        for provider in self.providers.values() {
            if provider
                .endpoints
                .iter()
                .any(|item| item.eq_ignore_ascii_case(endpoint_key))
            {
                candidates.push(provider);
            }
        }

        if candidates.is_empty() {
            return Err(SelectionError::EndpointNotConfigured);
        }

        if let Some(model_id) = model_id {
            if let Some(provider) = candidates
                .iter()
                .find(|provider| provider.model_id.eq_ignore_ascii_case(model_id))
            {
                return Ok(SelectionResult {
                    model_id: provider.model_id.clone(),
                    provider_id: provider.id.clone(),
                });
            }
            return Err(SelectionError::ModelNotConfigured);
        }

        let mut defaults = candidates.iter().filter(|provider| provider.default);
        let Some(provider) = defaults.next() else {
            return Err(SelectionError::ModelAmbiguous);
        };
        if defaults.next().is_some() {
            return Err(SelectionError::ModelAmbiguous);
        }

        Ok(SelectionResult {
            model_id: provider.model_id.clone(),
            provider_id: provider.id.clone(),
        })
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
        for item in &config.providers {
            let provider = match item.provider_type.as_str() {
                "openai" => build_openai_provider(&item.params)?,
                other => return Err(ConfigError::UnknownProviderType(other.to_string())),
            };

            let entry = ProviderEntry {
                id: item.id.clone(),
                provider_type: item.provider_type.clone(),
                model_id: item.model_id.clone(),
                endpoints: item.endpoints.clone(),
                provider: Arc::new(provider),
            };
            providers.insert(item.id.clone(), entry);
        }

        Ok(Self::new(providers, strategy))
    }

    pub fn new(
        providers: HashMap<ProviderId, ProviderEntry>,
        strategy: Arc<dyn SelectionStrategy<M>>,
    ) -> Self {
        Self {
            providers,
            strategy,
        }
    }

    pub fn select(
        &self,
        endpoint: Endpoint,
        model_id: Option<&str>,
        metadata: &M,
    ) -> Result<&ProviderEntry, anyhow::Error> {
        let selected = self
            .strategy
            .select(endpoint, model_id, metadata)
            .map_err(|err| anyhow::anyhow!("selection failed: {err:?}"))?;
        self.providers
            .get(&selected.provider_id)
            .ok_or_else(|| anyhow::anyhow!("unsupported provider {}", selected.provider_id))
    }

    pub fn providers(&self) -> &HashMap<ProviderId, ProviderEntry> {
        &self.providers
    }
}
