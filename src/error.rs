use std::error::Error;
use std::fmt;

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

#[derive(Debug)]
pub enum ProviderError {
    Unsupported(String),
}

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProviderError::Unsupported(message) => write!(f, "unsupported: {message}"),
        }
    }
}

impl Error for ProviderError {}
