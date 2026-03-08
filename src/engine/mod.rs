// Inference engine - main loop coordinating scheduler, model, and KV cache.

mod ffi;
pub mod model;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use tracing::debug;

use crate::kv_cache::KVCacheManager;
use crate::scheduler::{InferenceRequest, StopReason, Token};

use self::model::{InferenceRequestForModel, Logits, Model};

/// Main inference engine coordinating scheduler, model, and KV cache.
pub struct InferenceEngine {
    model: Arc<dyn Model>,
    scheduler: Arc<crate::scheduler::Scheduler>,
    kv_cache: Arc<KVCacheManager>,
    eos_token_id: i32,
    next_request_id: AtomicU64,
}

impl InferenceEngine {
    pub fn new(
        model: Arc<dyn Model>,
        scheduler: Arc<crate::scheduler::Scheduler>,
        kv_cache: Arc<KVCacheManager>,
        eos_token_id: i32,
    ) -> Self {
        Self {
            model,
            scheduler,
            kv_cache,
            eos_token_id,
            next_request_id: AtomicU64::new(0),
        }
    }

    /// Submit a request to the scheduler.
    pub fn submit_request(&self, req: InferenceRequest) {
        self.scheduler.submit(req);
    }

    /// Allocate a unique request ID.
    pub fn next_request_id(&self) -> u64 {
        self.next_request_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Main inference loop.
    pub async fn run_loop(self: Arc<Self>) -> Result<()> {
        let engine = self.clone();
        loop {
            let batch = engine.scheduler.schedule_step();

            if batch.is_empty() {
                tokio::time::sleep(Duration::from_micros(100)).await;
                continue;
            }

            // Run prefill and decode in parallel when both are non-empty
            let prefill_ids = batch.prefill.clone();
            let decode_ids = batch.decode.clone();

            if !prefill_ids.is_empty() {
                let prefill_results = engine.run_prefill(&prefill_ids).await?;
                engine.handle_logits(&prefill_results, true).await?;
            }

            if !decode_ids.is_empty() {
                let decode_results = engine.run_decode(&decode_ids).await?;
                engine.handle_logits(&decode_results, false).await?;
            }
        }
    }

    async fn run_prefill(&self, req_ids: &[u64]) -> Result<Vec<(u64, Logits)>> {
        let requests = self.scheduler.get_running(req_ids);
        let model_requests: Vec<InferenceRequestForModel> = requests
            .iter()
            .map(|r| InferenceRequestForModel {
                id: r.id,
                prompt_tokens: r.prompt_tokens.clone(),
                generated_tokens: r.generated_tokens,
                max_new_tokens: r.max_new_tokens,
                context_len: r.context_len(),
            })
            .collect();
        self.model.prefill(req_ids, &model_requests).await
    }

    async fn run_decode(&self, req_ids: &[u64]) -> Result<Vec<(u64, Logits)>> {
        let requests = self.scheduler.get_running(req_ids);
        let model_requests: Vec<InferenceRequestForModel> = requests
            .iter()
            .map(|r| InferenceRequestForModel {
                id: r.id,
                prompt_tokens: r.prompt_tokens.clone(),
                generated_tokens: r.generated_tokens,
                max_new_tokens: r.max_new_tokens,
                context_len: r.context_len(),
            })
            .collect();
        self.model.decode_step(req_ids, &model_requests).await
    }

    async fn handle_logits(&self, results: &[(u64, Logits)], from_prefill: bool) -> Result<()> {
        let req_ids: Vec<u64> = results.iter().map(|(id, _)| *id).collect();
        let running = self.scheduler.get_running(&req_ids);

        for (req_id, logits) in results {
            let req = running.iter().find(|r| r.id == *req_id);
            let Some(req) = req else {
                continue;
            };

            let token_id = logits.sampled_token;
            let is_eos = token_id == self.eos_token_id;
            let reached_max = req.generated_tokens + 1 >= req.max_new_tokens;

            // Send token to client
            let text = String::new(); // TODO: lookup in vocab if needed for streaming
            let _ = req.response_tx.send(Token {
                id: *req_id,
                token_id,
                text,
                is_eos,
            });

            debug!(request_id = req_id, token_id, "token generated");

            if is_eos || reached_max {
                let reason = if is_eos {
                    StopReason::Eos
                } else {
                    StopReason::Length
                };
                self.scheduler.mark_finished(*req_id, reason);
            } else {
                if from_prefill {
                    self.scheduler.mark_prefill_done(*req_id);
                }
                self.scheduler.increment_generated(*req_id);
            }
        }

        Ok(())
    }

    pub fn kv_cache_usage(&self) -> f32 {
        self.kv_cache.memory_usage()
    }

    pub fn queue_depth(&self) -> usize {
        self.scheduler.queue_depth()
    }

    pub fn active_requests(&self) -> usize {
        self.scheduler.active_requests()
    }
}
