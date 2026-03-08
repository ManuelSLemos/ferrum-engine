# Ferrum Engine

High-performance LLM inference engine in Rust — an alternative to Ollama and vLLM.

## Features

- **GGUF support** via llama.cpp FFI
- **OpenAI-compatible API** (chat completions, completions, models, health)
- **Continuous batching** with LIFO preemption
- **KV-cache management** with block-based allocation

## Prerequisites

- Rust toolchain
- CMake 3.14+
- C++ compiler with C++17 support
- (Optional) CUDA toolkit for GPU inference
- (Optional) libclang for bindgen

## Build

```bash
# Clone llama.cpp (required)
git clone https://github.com/ggml-org/llama.cpp vendor/llama.cpp

# Build with CPU backend
cargo build --release

# Build with CUDA (set CUDA_PATH if needed)
cargo build --release --features cuda

# Build stub only (no llama.cpp, for testing)
FERRUM_SKIP_LLAMA=1 cargo build --release
```

## Usage

```bash
# Start server
ferrum-engine --model-path /path/to/model.gguf

# Or with env vars
FERRUM_MODEL_PATH=/path/to/model.gguf FERRUM_PORT=8080 ferrum-engine
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Chat completions (OpenAI compatible) |
| POST | `/v1/completions` | Text completions |
| GET | `/v1/models` | List loaded models |
| GET | `/health` | Health check with KV cache metrics |

## Configuration

| Option | Env | Default |
|--------|-----|---------|
| `--model-path` | `FERRUM_MODEL_PATH` | Required |
| `--gpu-memory-fraction` | `FERRUM_GPU_MEMORY_FRACTION` | 0.85 |
| `--max-batch-size` | `FERRUM_MAX_BATCH_SIZE` | 32 |
| `--block-size` | `FERRUM_BLOCK_SIZE` | 16 |
| `--host` | `FERRUM_HOST` | 0.0.0.0 |
| `--port` | `FERRUM_PORT` | 8080 |

## Project Structure

```
rabbit-engine/
├── src/
│   ├── main.rs          # Entry point
│   ├── config.rs        # CLI/env configuration
│   ├── api/             # REST API (OpenAI compatible)
│   ├── scheduler/       # Continuous batching
│   ├── kv_cache/        # KV-cache memory manager
│   └── engine/          # Inference engine + llama.cpp model
└── vendor/llama.cpp/    # Git submodule/clone
```
