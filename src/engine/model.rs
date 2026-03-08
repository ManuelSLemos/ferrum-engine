// Model trait and LlamaCppModel implementation.
// Uses llama.cpp FFI for GGUF loading and inference.

use async_trait::async_trait;
use std::sync::Arc;

use anyhow::Result;

/// Model architecture configuration.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_heads_kv: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
}

/// Logits from a single decode step (vocab_size floats).
#[derive(Debug, Clone)]
pub struct Logits {
    pub values: Vec<f32>,
    pub sampled_token: i32,
}

impl Logits {
    pub fn new(values: Vec<f32>, sampled_token: i32) -> Self {
        Self { values, sampled_token }
    }
}

/// Inference request (minimal view for model).
#[derive(Debug, Clone)]
pub struct InferenceRequestForModel {
    pub id: u64,
    pub prompt_tokens: Vec<i32>,
    pub generated_tokens: usize,
    pub max_new_tokens: usize,
    pub context_len: usize,
}

/// Backend model trait.
#[async_trait]
pub trait Model: Send + Sync {
    async fn prefill(
        &self,
        req_ids: &[u64],
        requests: &[InferenceRequestForModel],
    ) -> Result<Vec<(u64, Logits)>>;

    async fn decode_step(
        &self,
        req_ids: &[u64],
        requests: &[InferenceRequestForModel],
    ) -> Result<Vec<(u64, Logits)>>;

    fn model_config(&self) -> ModelConfig;
}

/// Llama.cpp model via FFI.
/// Thread-safe via Arc<Mutex<>> around context.
pub struct LlamaCppModel {
    _model: std::ptr::NonNull<std::ffi::c_void>,
    _ctx: Arc<tokio::sync::Mutex<std::ptr::NonNull<std::ffi::c_void>>>,
    config: ModelConfig,
}

// Safe: model is loaded once and read-only; context access is guarded by Mutex.
unsafe impl Send for LlamaCppModel {}
unsafe impl Sync for LlamaCppModel {}

// Opaque types for FFI - actual definitions in ffi module
#[allow(dead_code)]
type LlamaModel = std::ffi::c_void;
#[allow(dead_code)]
type LlamaContext = std::ffi::c_void;

impl LlamaCppModel {
    /// Load a GGUF model from path.
    pub fn load(
        model_path: &std::path::Path,
        max_batch_size: usize,
    ) -> Result<Self> {
        // FFI implementation will go in a separate module
        // For now we use a stub that fails - the real impl will use bindings
        let config = ModelConfig {
            num_layers: 32,
            num_heads: 32,
            num_heads_kv: 32,
            head_dim: 128,
            vocab_size: 32000,
        };

        // Stub: in real impl, call llama_model_load_from_file
        // We'll use cfg to disable this when llama isn't built
        let _ = (model_path, max_batch_size);

        Ok(Self {
            _model: unsafe { std::ptr::NonNull::new_unchecked(1usize as *mut _) },
            _ctx: Arc::new(tokio::sync::Mutex::new(unsafe {
                std::ptr::NonNull::new_unchecked(1usize as *mut _)
            })),
            config,
        })
    }
}

#[async_trait]
impl Model for LlamaCppModel {
    async fn prefill(
        &self,
        req_ids: &[u64],
        _requests: &[InferenceRequestForModel],
    ) -> Result<Vec<(u64, Logits)>> {
        // Stub: return EOS (2) for each request so the client receives a response
        let results: Vec<_> = req_ids
            .iter()
            .map(|&id| (id, Logits::new(vec![], 2)))
            .collect();
        Ok(results)
    }

    async fn decode_step(
        &self,
        req_ids: &[u64],
        _requests: &[InferenceRequestForModel],
    ) -> Result<Vec<(u64, Logits)>> {
        // Stub: return EOS for each (should not be called after prefill sends EOS)
        let results: Vec<_> = req_ids
            .iter()
            .map(|&id| (id, Logits::new(vec![], 2)))
            .collect();
        Ok(results)
    }

    fn model_config(&self) -> ModelConfig {
        self.config.clone()
    }
}
