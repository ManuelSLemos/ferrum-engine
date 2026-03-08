// Axum routes for OpenAI-compatible API.

use axum::{
    extract::State,
    response::{IntoResponse, sse::{Event, Sse}},
    routing::{get, post},
    Json, Router,
};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

use crate::engine::InferenceEngine;
use crate::scheduler::{InferenceRequest, Token};

use super::types::*;

pub fn router(engine: Arc<InferenceEngine>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/models", get(models))
        .route("/health", get(health))
        .with_state(engine)
}

async fn health(State(engine): State<Arc<InferenceEngine>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        kv_cache_usage: engine.kv_cache_usage(),
        queue_depth: engine.queue_depth(),
        active_requests: engine.active_requests(),
    })
}

async fn models(State(_engine): State<Arc<InferenceEngine>>) -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: "default".to_string(),
            object: "model".to_string(),
        }],
    })
}

async fn chat_completions(
    State(engine): State<Arc<InferenceEngine>>,
    Json(req): Json<ChatCompletionRequest>,
) -> axum::response::Response {
    let id = Uuid::new_v4().to_string();
    let req_id = engine.next_request_id();
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Build prompt from messages (simple concatenation; chat template would go here)
    let prompt = req
        .messages
        .iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n");

    // Tokenize - for MVP we use a simple split; real impl uses model tokenizer
    let prompt_tokens: Vec<i32> = {
        let words: Vec<i32> = prompt
            .split_whitespace()
            .enumerate()
            .map(|(i, _)| i as i32)
            .collect();
        if words.is_empty() {
            let fallback: Vec<i32> = prompt.bytes().map(|b| b as i32).take(100).collect();
            if fallback.is_empty() {
                vec![0]
            } else {
                fallback
            }
        } else {
            words
        }
    };

    let max_tokens = req.max_tokens.unwrap_or(256) as usize;
    let prompt_tokens_len = prompt_tokens.len();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Token>();

    let inference_req = InferenceRequest::new(req_id, prompt_tokens, max_tokens, tx);

    engine.submit_request(inference_req);

    if req.stream {
        let stream = async_stream::stream! {
            while let Some(token) = rx.recv().await {
                let content = token.text.clone();
                let chunk = ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: req.model.clone(),
                    choices: vec![ChatCompletionChunkChoice {
                        index: 0,
                        delta: ChatMessageDelta {
                            role: None,
                            content: Some(content),
                        },
                        finish_reason: if token.is_eos { Some("stop".to_string()) } else { None },
                    }],
                };
                yield Ok::<_, std::convert::Infallible>(Event::default().json_data(chunk).unwrap());
                if token.is_eos {
                    break;
                }
            }
        };

        Sse::new(stream).into_response()
    } else {
        // Collect full response
        let mut full_content = String::new();
        while let Some(token) = rx.recv().await {
            full_content.push_str(&token.text);
            if token.is_eos {
                break;
            }
        }

        let completion_tokens = full_content.split_whitespace().count();
        let response = ChatCompletionResponse {
            id: id.clone(),
            object: "chat.completion".to_string(),
            created,
            model: req.model.clone(),
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ChatMessageResponse {
                    role: "assistant".to_string(),
                    content: full_content,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(Usage {
                prompt_tokens: prompt_tokens_len as u32,
                completion_tokens: completion_tokens as u32,
                total_tokens: 0,
            }),
        };

        Json(response).into_response()
    }
}

async fn completions(
    State(engine): State<Arc<InferenceEngine>>,
    Json(req): Json<CompletionRequest>,
) -> axum::response::Response {
    // Convert to chat format and delegate
    let chat_req = ChatCompletionRequest {
        model: req.model,
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: req.prompt,
        }],
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: None,
        stream: req.stream,
    };

    chat_completions(State(engine), Json(chat_req)).await
}
