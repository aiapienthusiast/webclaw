/// Cheaper Inference provider — OpenAI-compatible chat completions with Cheaper Inference defaults.
use async_trait::async_trait;

use crate::error::LlmError;
use crate::provider::{CompletionRequest, LlmProvider};

use super::openai::OpenAiProvider;

pub struct CheaperInferenceProvider {
    inner: OpenAiProvider,
}

impl CheaperInferenceProvider {
    /// Returns `None` if no Cheaper Inference API key is available (param or env).
    pub fn new(
        key_override: Option<String>,
        base_url: Option<String>,
        model: Option<String>,
    ) -> Option<Self> {
        let key = super::load_api_key(key_override, "CHEAPER_INFERENCE_API_KEY")?;
        let base_url = base_url
            .or_else(|| std::env::var("CHEAPER_INFERENCE_BASE_URL").ok())
            .unwrap_or_else(|| "https://api.cheaperinference.com/v1".into());
        let model = model
            .or_else(|| std::env::var("CHEAPER_INFERENCE_MODEL").ok())
            .unwrap_or_else(|| "gpt-5.4-mini".into());
        let inner = OpenAiProvider::new(Some(key), Some(base_url), Some(model))?;
        Some(Self { inner })
    }

    pub fn default_model(&self) -> &str {
        self.inner.default_model()
    }
}

#[async_trait]
impl LlmProvider for CheaperInferenceProvider {
    async fn complete(&self, request: &CompletionRequest) -> Result<String, LlmError> {
        self.inner.complete(request).await
    }

    async fn is_available(&self) -> bool {
        self.inner.is_available().await
    }

    fn name(&self) -> &str {
        "cheaperinference"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_key_returns_none() {
        assert!(CheaperInferenceProvider::new(Some(String::new()), None, None).is_none());
    }

    #[test]
    fn explicit_key_constructs_with_cheaperinference_defaults() {
        let provider = CheaperInferenceProvider::new(Some("test-key".into()), None, None)
            .expect("should construct");
        assert_eq!(provider.name(), "cheaperinference");
        assert_eq!(provider.default_model(), "gpt-5.4-mini");
    }

    #[test]
    fn explicit_model_override() {
        let provider = CheaperInferenceProvider::new(
            Some("test-key".into()),
            Some("https://proxy.example.com/v1".into()),
            Some("some-model".into()),
        )
        .expect("should construct");
        assert_eq!(provider.default_model(), "some-model");
    }
}
