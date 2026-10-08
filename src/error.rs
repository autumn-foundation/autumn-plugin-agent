//! Error types for the agent plugin.
//!
//! Every failure the plugin can produce carries an [`ErrorKind`]. The kind
//! maps to an HTTP status code through [`ErrorKind::status_code`], and to an
//! [`AutumnError`](autumn_web::AutumnError) through
//! [`AgentError::into_autumn_error`] (a method, not a `From` impl, because
//! Autumn already provides a blanket `From<E: Error>` that would conflict).
//! Handlers and jobs translate agent failures into framework responses with
//! one call.

use http::StatusCode;
use thiserror::Error;

/// Machine-readable category for an [`AgentError`].
///
/// Variants stay coarse on purpose: callers match on these, humans read the
/// message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorKind {
    /// Plugin configuration is missing or invalid.
    Config,
    /// The provider rejected the credentials (`AGENT_API_KEY` is wrong or
    /// revoked).
    Authentication,
    /// The provider rate-limited the request.
    RateLimited,
    /// The provider answered with an error payload.
    Provider,
    /// The HTTP transport failed before a response arrived.
    Transport,
    /// A tool failed while executing.
    Tool,
    /// A budget (steps, tokens, or context window) overflowed.
    Budget,
    /// A provider response could not be decoded.
    Decode,
}

impl ErrorKind {
    /// Map the kind to the HTTP status an app should answer with.
    ///
    /// The mapping treats the provider as an upstream dependency: provider
    /// failures surface as 502/503/429, never as 4xx against the caller.
    #[must_use]
    pub const fn status_code(self) -> StatusCode {
        match self {
            Self::Config | Self::Tool | Self::Decode => StatusCode::INTERNAL_SERVER_ERROR,
            Self::Authentication | Self::Provider => StatusCode::BAD_GATEWAY,
            Self::RateLimited => StatusCode::TOO_MANY_REQUESTS,
            Self::Transport => StatusCode::SERVICE_UNAVAILABLE,
            Self::Budget => StatusCode::PAYLOAD_TOO_LARGE,
        }
    }
}

/// The plugin's error type.
///
/// Build one with [`AgentError::new`] or the `From` impls for the usual
/// failure sources. Inspect it with [`AgentError::kind`] and
/// [`AgentError::status_code`].
#[derive(Debug, Error)]
#[error("[{kind:?}] {message}")]
pub struct AgentError {
    kind: ErrorKind,
    message: String,
    #[source]
    source: Option<Box<dyn std::error::Error + Send + Sync>>,
}

impl AgentError {
    /// Build an error from a kind and a message.
    #[must_use]
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            source: None,
        }
    }

    /// Build an error that keeps the underlying cause for `Error::source`.
    pub fn with_source<E>(kind: ErrorKind, message: impl Into<String>, source: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self {
            kind,
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }

    /// Return the machine-readable category.
    #[must_use]
    pub const fn kind(&self) -> ErrorKind {
        self.kind
    }

    /// Return the HTTP status an app should answer with.
    #[must_use]
    pub const fn status_code(&self) -> StatusCode {
        self.kind.status_code()
    }

    /// Return the human-readable message.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Translate the agent failure into a framework error.
    ///
    /// Kinds that describe the app's own misconfiguration or tool code map
    /// to 500 helpers; upstream provider failures map to 502/503/429 through
    /// [`AutumnError::with_status`](autumn_web::AutumnError::with_status).
    /// This is a method rather than a `From` impl because Autumn ships a
    /// blanket `From<E: Error>` for `AutumnError` that would conflict.
    #[cfg(feature = "autumn")]
    #[must_use]
    pub fn into_autumn_error(self) -> autumn_web::AutumnError {
        let message = self.to_string();
        match self.kind() {
            ErrorKind::Config | ErrorKind::Tool | ErrorKind::Decode => {
                autumn_web::AutumnError::internal_server_error_msg(message)
            }
            ErrorKind::Authentication | ErrorKind::Provider => {
                autumn_web::AutumnError::internal_server_error_msg(message)
                    .with_status(StatusCode::BAD_GATEWAY)
            }
            ErrorKind::RateLimited => autumn_web::AutumnError::service_unavailable_msg(message)
                .with_status(StatusCode::TOO_MANY_REQUESTS),
            ErrorKind::Transport => autumn_web::AutumnError::service_unavailable_msg(message),
            ErrorKind::Budget => autumn_web::AutumnError::internal_server_error_msg(message)
                .with_status(StatusCode::PAYLOAD_TOO_LARGE),
        }
    }
}

impl From<serde_json::Error> for AgentError {
    fn from(err: serde_json::Error) -> Self {
        Self::with_source(ErrorKind::Decode, "failed to decode JSON", err)
    }
}

impl From<reqwest::Error> for AgentError {
    fn from(err: reqwest::Error) -> Self {
        let kind = if err.is_timeout() || err.is_connect() {
            ErrorKind::Transport
        } else if err.is_decode() {
            ErrorKind::Decode
        } else {
            ErrorKind::Transport
        };
        Self::with_source(kind, "LLM provider request failed", err)
    }
}

#[cfg(test)]
mod tests;
