# llm-router

`llm-router` is a lightweight OpenAI-compatible HTTP router for local and self-hosted LLM backends.

It routes requests by model ID, exposes a merged `/v1/models` list, and can optionally pre-stage model files from Hugging Face before forwarding traffic.

## Features

- OpenAI-style proxy surface (`/v1/*` passthrough)
- Model-based backend routing via `model` in JSON request bodies
- Aggregated model listing at `/v1/models` and `/models`
- Optional per-model local file checks and on-demand Hugging Face downloads
- Optional PI `models.json` contextWindow synchronization (disabled by default)
- Simple health endpoint at `/healthz`

## Requirements

- Go 1.24+

## Quick Start

1. Copy the example config and edit it for your local backends:

```bash
cp router.example.yaml router.yaml
```
2. Start the router:

```bash
go run .
```

3. Call it like an OpenAI-compatible endpoint:

```bash
curl http://127.0.0.1:8090/v1/models
```

## Configuration

Configuration is loaded from `router.yaml` by default. The repository tracks `router.example.yaml` as the template, and `router.yaml` is local-only. Override path with:

- `LLM_ROUTER_CONFIG=/path/to/router.yaml`

You can also override the listen port with:

- `LLM_ROUTER_PORT=8090`

### Top-level keys

- `listen_addr`: Bind address (default `127.0.0.1:${LLM_ROUTER_PORT|8090}`)
- `enable_pi_model_sync`: Enables syncing route `context_window` values into PI's `models.json` on startup. Default: `false`
- `pi_models_json_path`: PI models file path used only when PI sync is enabled. If omitted while enabled, defaults to `~/.pi/agent/models.json`
- `routes`: List of model routes (must contain at least one)

### Route keys

- `id`: Model ID expected in request bodies
- `backend_url`: Upstream base URL for that model
- `context_window`: Value advertised in `/v1/models` and used for optional PI sync
- `model_file_path`: Optional local file path. If set and missing, the router can fetch it via `huggingface`
- `huggingface`: Optional object for download metadata:
  - `repo`
  - `filename`
  - `revision` (default `main`)
  - `token_env` (default `HF_TOKEN`)
  - `target_path` (defaults to `model_file_path`)

## Build

```bash
go build -o llm-router .
```

## Test

```bash
go test ./...
```
