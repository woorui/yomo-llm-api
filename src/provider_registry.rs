use std::collections::HashMap;
use std::sync::Arc;

use crate::config::Config;
use crate::config::ConfigError;
use crate::config::EndpointRouteSet;
use crate::endpoint::Endpoint;
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
}

#[derive(Clone, Debug)]
pub struct SelectedRoute {
    pub model_id: String,
    pub model: String,
    pub provider_id: String,
}

#[derive(Debug)]
pub enum RouteSelectionError {
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
    ) -> Result<SelectedRoute, RouteSelectionError>;
}

#[derive(Clone)]
pub struct ByModel {
    routes: HashMap<Endpoint, EndpointRouteSet>,
}

impl ByModel {
    pub fn new(routes: HashMap<Endpoint, EndpointRouteSet>) -> Self {
        Self { routes }
    }
}

impl<M> SelectionStrategy<M> for ByModel {
    fn select(
        &self,
        endpoint: Endpoint,
        model_id: Option<&str>,
        _metadata: &M,
    ) -> Result<SelectedRoute, RouteSelectionError> {
        let Some(routes) = self.routes.get(&endpoint) else {
            return Err(RouteSelectionError::EndpointNotConfigured);
        };

        if let Some(model_id) = model_id {
            if let Some(route) = routes
                .routes
                .iter()
                .find(|route| route.model_id.eq_ignore_ascii_case(model_id))
            {
                return Ok(SelectedRoute {
                    model_id: route.model_id.clone(),
                    model: route.model.clone(),
                    provider_id: route.provider_id.clone(),
                });
            }
            return Err(RouteSelectionError::ModelNotConfigured);
        }

        if routes.routes.len() == 1 {
            let route = &routes.routes[0];
            return Ok(SelectedRoute {
                model_id: route.model_id.clone(),
                model: route.model.clone(),
                provider_id: route.provider_id.clone(),
            });
        }

        if let Some(route) = routes.routes.iter().find(|route| route.default) {
            return Ok(SelectedRoute {
                model_id: route.model_id.clone(),
                model: route.model.clone(),
                provider_id: route.provider_id.clone(),
            });
        }

        Err(RouteSelectionError::ModelAmbiguous)
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
                model: item.model.clone(),
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
            .map_err(|err| anyhow::anyhow!("route selection failed: {err:?}"))?;
        self.providers
            .get(&selected.provider_id)
            .ok_or_else(|| anyhow::anyhow!("unsupported provider {}", selected.provider_id))
    }

    pub fn providers(&self) -> &HashMap<ProviderId, ProviderEntry> {
        &self.providers
    }
}
