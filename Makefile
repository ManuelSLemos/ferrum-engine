# Makefile for ferrum-engine
# Usage: make build && make run

PATH := $(HOME)/.cargo/bin:$(PATH)
export PATH

MODELS_DIR ?= models
MODEL_REPO ?= unsloth/Qwen3.5-0.8B-GGUF
MODEL_FILE ?= Qwen3.5-0.8B-Q4_K_M.gguf
MODEL_PATH ?= $(MODELS_DIR)/$(MODEL_FILE)
DOCKER_IMAGE ?= python:3.11-slim

.PHONY: download-model build run install-rust help

help:
	@echo "Available targets:"
	@echo "  make install-rust   - Install Rust toolchain (run this first if Rust is not installed)"
	@echo "  make build          - Build ferrum-engine (requires Rust, llama.cpp in vendor/)"
	@echo "  make run            - Run server with $(MODEL_PATH)"
	@echo "  make download-model - Download Qwen3.5 0.8B (Q4_K_M) to $(MODELS_DIR)/"
	@echo ""
	@echo "Variables:"
	@echo "  MODEL_PATH  - Path to GGUF model (default: $(MODEL_PATH))"
	@echo "  MODELS_DIR, MODEL_REPO, MODEL_FILE - For download-model"

install-rust:
	@command -v cargo >/dev/null 2>&1 && (echo "Rust already installed:"; cargo --version) || \
		(curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
		. $(HOME)/.cargo/env && cargo --version && echo "Rust installed. Run: source ~/.cargo/env && make build")

download-model:
	@mkdir -p $(MODELS_DIR)
	@echo "Downloading $(MODEL_FILE) from $(MODEL_REPO)..."
	docker run --rm \
		-e PIP_ROOT_USER_ACTION=ignore \
		-v "$(PWD)/$(MODELS_DIR):/data" \
		-w /data \
		$(DOCKER_IMAGE) \
		sh -c "pip install --quiet huggingface_hub && \
			python -c \"from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='$(MODEL_REPO)', filename='$(MODEL_FILE)', local_dir='.')\""
	@echo "Model saved to $(MODELS_DIR)/$(MODEL_FILE)"

build:
	@command -v cargo >/dev/null 2>&1 || (echo "Rust not found. Install from https://rustup.rs" && exit 1)
	cargo build --release

run: build
	./target/release/ferrum-engine --model-path $(MODEL_PATH)
