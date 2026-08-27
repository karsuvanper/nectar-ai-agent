---
title: Nectar Autonomous Voice Agent
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Nectar — Autonomous Facility Voice Agent

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11"/>
  <img src="https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/>
  <img src="https://img.shields.io/badge/Qdrant-Vector_DB-FF4A4A?style=for-the-badge" alt="Qdrant"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License MIT"/>
</p>

<p align="center">
  <b>Intelligent. Autonomous. Voice-Native Facility Operations.</b><br/>
  Real-time telemetry diagnostics · Two-Stage RAG · MCP tooling · Neural TTS · Glassmorphism UI
</p>

<p align="center">
  <a href="#-architecture">Architecture</a> •
  <a href="#-quickstart">Quickstart</a> •
  <a href="#-live-endpoints">Endpoints</a> •
  <a href="#-test-walkthrough">Walkthrough</a> •
  <a href="#-api-specifications">API Spec</a> •
  <a href="#-hugging-face-spaces-deployment">HF Spaces</a>
</p>

---

> **Nectar** is a production-grade autonomous voice agent for HVAC facility management. It fuses edge-native speech handling, a 7-step multi-agent orchestrator, two-stage FastRAG with cross-encoder reranking, Model Context Protocol (MCP) telemetry/action tools, and streaming neural TTS — all behind a low-latency FastAPI + WebSocket backbone and an ultra-luxurious glassmorphism interface.

## ✨ Key Highlights

| Capability | Detail |
|---|---|
| **< 30ms Barge-in** | Client-side Silero-inspired RMS VAD with hangover counter — interrupt TTS instantly while speaking |
| **Zero server STT cost** | Browser Web Speech API (client-side) with continuous + interim results + auto-restart |
| **Autonomous Reasoning** | 7-step orchestrator chaining telemetry → status → alerts → RAG → LLM diagnosis → confirmation |
| **Grounded Answers** | Two-stage FastRAG: `BAAI/bge-small-en-v1.5` dense retrieval → `ms-marco-MiniLM-L-6-v2` cross-encoder rerank |
| **Safe Actuation** | MCP read/write tools with human-in-the-loop confirmation guardrails before ticket creation |
| **Cinematic UI** | Glassmorphism (`rgba(18,24,38,0.75)`), 3D audio spectrum canvas, live execution drawer, approval modal |
| **Streaming Voice** | Kokoro-82M ONNX primary (WAV) → `edge-tts` `en-US-ChristopherNeural` (MP3) → synthetic silence fallback |

---

## 📋 Core System Features

### 1. Client-Side Silero-inspired RMS VAD (<30ms Instant Barge-In)
- Implemented in `app/templates/index.html:304-320` — `AnalyserNode` (`fftSize=512`, `smoothingTimeConstant=0.78`) computes RMS over frequency data every animation frame.
- Threshold `vadThreshold=28`, hangover `vadHang=8` frames. Counter `vadCounter` requires sustained energy before firing.
- On trigger, `triggerBargeIn()` sends `{"type":"interrupt"}` over WebSocket, stops all `AudioBufferSourceNode`s, flushes queue, and shows `⚡ Interrupted` badge.
- Server `app/voice/voice_router.py:59-75` cancels active `asyncio.Task` via `active_tts_task.cancel()` and acks `{"type":"interrupted","status":"cancelled"}`.

### 2. Browser Web Speech API (Client-Side STT)
- `app/templates/index.html:449-471` — `SpeechRecognition || webkitSpeechRecognition`, `continuous=true`, `interimResults=true`, `lang='en-US'`.
- Auto-restart on `onend` while `listening==true`; final transcripts dispatch `sendQuery(final.trim())` directly to WebSocket. No audio upload, no server STT budget.
- Fallback text input `#textInput` retained for browsers without STT or for noisy environments.

### 3. 7-Step Multi-Agent Autonomous Orchestrator
`app/agent/orchestrator.py:165-425` · `app/router/intent_router.py:53-88`

| Step | Name | Tool Call |
|------|------|-----------|
| **1** | **Identify Building / Floor / Asset** | `_extract_entities_from_query()` + `_resolve_asset_from_entities()` → `BUILDING_FLOOR_ASSET_MAP`; default `3rd floor → AHU-02` (Building A North Wing) |
| **2** | **Get Sensor Readings** | `get_sensor_data(asset_id)` → `{temperature, pressure, airflow, vibration}` |
| **3** | **Find Related HVAC Assets & Status** | `get_asset_relationships()` + `get_asset_status()` (e.g., Chillers, Valves linked to AHU) |
| **4** | **Check Active Alerts** | `get_active_alerts(asset_id)` + building-level alerts via `get_asset_details()` |
| **5** | **RAG Troubleshooting Retrieval** | `query_rag_agent("hot office low airflow AHU ...")` — two-stage reranked HVAC guide |
| **6** | **LLM Diagnostic Synthesis** | `_call_llm_diagnosis()` prompts `config.DEFAULT_MODEL` with all telemetry + RAG |
| **7** | **Decide Maintenance & Ask Confirmation** | Heuristic (`Warning`/`Low Airflow`) → `pending_action={action:"create_service_request", priority:"High"}` with `requires_action_confirmation=True` |

- **Routing layer** `app/router/intent_router.py:12-24` classifies every query into 6 routes: `RAG`, `MCP`, `RAG_MCP_REASONING`, `MCP_ACTION`, `DATA_AGENT`, `GENERAL_LLM` via `config.ROUTER_MODEL` (`meta-llama/llama-3.3-70b-instruct:free`) with JSON-object response format and a heuristic fallback for the `hot floor + maintenance` scenario.

### 4. Two-Stage FastRAG with Cross-Encoder Reranking
`app/rag/vector_store.py:1-67` · `app/rag/retriever.py:1-84` · `app/rag/docs/hvac_faq.txt`

- **Stage A — Dense Retrieval:** `fastembed` `BAAI/bge-small-en-v1.5` (384-dim, cosine) indexed in Qdrant collection `hvac_knowledge`; `search(query, top_k=15)` returns candidates.
- **Stage B — Cross-Encoder Rerank:** `sentence-transformers` `cross-encoder/ms-marco-MiniLM-L-6-v2` scores `[[query, doc]]` pairs via `reranker.predict()` → `np.argsort()[::-1][:3]` top reranked chunks.
- **Grounded Generation:** Concatenated context → OpenRouter `DEFAULT_MODEL` with strict prompt *"grounded ONLY on retrieved context"*; guardrail returns `Sufficient information was not found...` if top score `<0.3` or LLM offline.
- **Auto-Index:** `app/main.py:39-48` `auto_index()` on FastAPI startup reads `app/rag/docs/hvac_faq.txt` (385 lines: operating procedures, chiller specs, AHU low-airflow guides, MERV13 filter DP limits, safety LOTO, escalation policies) and upserts to Qdrant.

### 5. Model Context Protocol (MCP) Read/Write Tools
`app/mcp_tools/telemetry.py:1-267` · `app/mcp_tools/actions.py:1-54` · `app/mcp_tools/mcp_server.py:1-109`

| Tool | Type | Signature | Role |
|------|------|-----------|------|
| `get_asset_details` | Read | `(asset_id: str) -> dict` | Location, building, model, specs for `Chiller-01/02`, `AHU-02/03`, `Valve-01/02` |
| `get_asset_status` | Read | `(asset_id: str) -> dict` | `operational_status`, `running_state`, `health`, `current_mode` (e.g., AHU-02 → `Warning/Low Airflow`) |
| `get_sensor_data` | Read | `(asset_id: str) -> dict` | `temperature`, `pressure`, `airflow`, `vibration` (AHU-02: 85 CFM vs expected ~300 CFM) |
| `get_energy_consumption` | Read | `(asset_id: str) -> dict` | `kW`, `kWh`, `power_usage` |
| `get_active_alerts` | Read | `(asset_id_or_building: str) -> dict` | `warnings`, `critical_alarms` (includes `Building A` aggregate) |
| `get_asset_relationships` | Read | `(asset_id: str) -> dict` | `connected_ahus`, `connected_chillers`, `connected_valves` topology |
| `create_service_request` | **Write** | `(asset_id, issue, priority, confirmed) -> MaintenanceRequest` | Guardrail: `confirmed=False` → returns `requires_confirmation=True` prompt; `confirmed=True` → `TICK-{id}` |
| `update_service_request` | **Write** | `(ticket_id, status, notes, confirmed) -> dict` | Same confirmation guardrail pattern |

- Exposed as thin REST wrappers in `app/main.py:54-118` under `/api/v1/mcp/*` and as a proper `MCPServer` (`mcp>=1.0.0`) in `app/mcp_tools/mcp_server.py:99-108` (`name="nectar-facility-mcp"`) for LLM tool discovery via `Tool` descriptors.
- Every write path mandates explicit user approval — rendered as the UI confirmation modal and persisted as `pending_action` in `AgentRequest.context`.

### 6. Ultra-Luxurious Glassmorphism Web UI
`app/templates/index.html:1-597`

- **Aesthetic:** `rgba(18,24,38,0.75)` glass cards, `backdrop-filter: blur(18px) saturate(160%)`, neon cyan `#00f2fe` → electric violet `#4facfe` gradients, `Inter` + `Outfit` typography, radial glow orbs.
- **3D Audio Spectrum Canvas:** `#visualizer` `<canvas 800×280>` drawn at devicePixelRatio; analyser-driven wave (90 points), gradient sphere (`sphereR = 56 + amp*60`), inner core with `shadowBlur` pulsing on speaking state.
- **Execution Drawer:** 7-step agentic reasoning stepper (`.step.active/.done`) animated at 320 ms intervals from `execution_steps` via `animateSteps()` — mirrors orchestrator logs live.
- **Confirmation Modal:** `#confirmationModal` with `Approve & Open Ticket` (`#approveBtn`) and `Cancel`; surfaces `pending_action` fields (`asset_id`, `probable_cause`, `priority: High`) with `requires_action_confirmation` signal.
- **WebSocket Badge:** `#wsBadge` toggles `live` (green pulse) / `disconnected` (red); latency measured from `latencyStart = Date.now()` at `sendQuery()` to `text_chunk` arrival.
- **Telemetry Cards:** Real-time `AHU-02` values — `28°C`, `85 CFM`, `MERV13 Warning` (`t-card.warn` variant).

### 7. Streaming Local Neural TTS (Kokoro-82M ONNX + Edge-TTS Fallback)
`app/voice/tts.py:1-298` · `app/voice/voice_router.py:122-157`

- **Priority cascade in `generate_audio_stream()`:**
  1. **Kokoro-82M ONNX** (`kokoro-onnx>=0.3.1` + `onnxruntime>=1.17.0`) — checked via `_has_kokoro_model()` scanning `models/*.onnx`, `KOKORO_MODEL_PATH` env, sentence-split pseudo-streaming → WAV 16-bit PCM chunks.
  2. **Edge-TTS** (`edge-tts>=6.1.9`) — `edge_tts.Communicate(text, voice="en-US-ChristopherNeural")`, `.stream()` yields MP3 `{"type":"audio","data":...}` frames.
  3. **Synthetic WAV** — `_make_silence_wav(duration=max(0.5,min(5,len(text)*0.02)))` chunked at 4096 bytes; guarantees audio payload offline.
- **WebSocket streaming:** `_stream_audio()` in `voice_router.py:122` iterates `generate_audio_stream()` and `await websocket.send_bytes(chunk)` per chunk with `await asyncio.sleep(0)` yield for interruptibility; finalizes with `{"type":"audio_complete"}` unless cancelled.
- **REST audio:** `POST /api/v1/voice/chat` collects chunks into `audio_bytes`, base64-encodes as `audio_base64`/`audio`, detects format by `RIFF` header (`wav` vs `mp3`).

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        EDGE BROWSER LAYER                                │
│  ┌─────────────────────┐  ┌──────────────────────┐  ┌─────────────────┐ │
│  │  Web Speech API     │  │ Silero-inspired RMS  │  │ Canvas Spectrum │ │
│  │  (Client STT)       │  │ VAD + Barge-In       │  │ 3D Visualizer   │ │
│  │  interim + final    │─▶│ threshold 28, hang 8 │  │ 90-pt wave      │ │
│  └──────────┬──────────┘  └──────────┬───────────┘  └────────▲────────┘ │
│             │  {"type":"query","text"} │ {"type":"interrupt"} │  bytes    │
└─────────────┼──────────────────────────┼──────────────────────┼───────────┘
              │                          │                      │
              ▼                          ▼                      │
┌─────────────────────────────────────────────────────────────────────────┐
│                    FastAPI WebSocket + REST LAYER                        │
│  app/main.py ─ FastAPI + CORS  ·  app/voice/voice_router.py              │
│  ┌──────────────────────────┐  ┌───────────────────────────────────────┐ │
│  │  WS /ws/agent/voice      │  │  REST /api/v1/voice/chat  (fallback)   │ │
│  │  text_chunk + audio      │  │  audio_base64 + audio_complete         │ │
│  │  interrupted / tts_error │  │  POST /api/v1/agent/chat               │ │
│  └────────────┬─────────────┘  └────────────────┬──────────────────────┘ │
└───────────────┼─────────────────────────────────┼────────────────────────┘
                │  AgentRequest(query, context)   │
                ▼                                 │
┌─────────────────────────────────────────────────────────────────────────┐
│                         AGENTIC ENGINE                                   │
│  app/router/intent_router.py ──▶  app/agent/orchestrator.py              │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Intent Router (config.ROUTER_MODEL)                             │  │
│  │  RAG | MCP | RAG_MCP_REASONING | MCP_ACTION | DATA_AGENT | GEN  │  │
│  └────────────────────────────┬─────────────────────────────────────┘  │
│                               │  7-Step RAG_MCP_REASONING                │
│              ┌────────────────┼─────────────────────────────────┐        │
│              ▼                ▼                                 │        │
│   ┌──────────────────┐  ┌───────────────┐  ┌─────────────────┐  │        │
│   │ Entity Resolver  │  │ Telemetry MCP │  │ FastRAG Pipeline│  │        │
│   │ Building/Floor → │─▶│ 6 read tools  │  │ dense → rerank  │  │        │
│   │ AHU-02 mapping   │  │ + 2 action    │  │ bge-small-en    │  │        │
│   └──────────────────┘  └───────┬───────┘  │ ms-marco-MiniLM │  │        │
│                                 │          └────────┬────────┘  │        │
│                                 └───────────────────┼───────────┘        │
│                                  diagnosis + confirmation                │
│                                     │  pending_action                    │
└─────────────────────────────────────┼────────────────────────────────────┘
                                      │ execution_steps + response_text
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    DATA & KNOWLEDGE PLANE                                │
│  ┌──────────────────────────┐  ┌──────────────────────────────────────┐  │
│  │  Qdrant Vector DB        │  │  MCP Tool Registry                   │  │
│  │  qdrant/qdrant:latest    │  │  app/mcp_tools/mcp_server.py          │  │
│  │  collection: hvac_knowledge │ MCPServer("nectar-facility-mcp")    │  │
│  │  384-dim bge-small-en    │  │  8 Tools (6 read + 2 write)          │  │
│  │  :6333 (compose 6335)   │  │  REST /api/v1/mcp/* wrappers         │  │
│  └──────────────────────────┘  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
┌─────────────────────────────────────────────────────────────────────────┐
│                      LOCAL TTS ENGINE                                    │
│  app/voice/tts.py  generate_audio_stream(text) -> AsyncIterator[bytes]   │
│  ┌──────────────┐   ┌──────────────────────────┐   ┌──────────────────┐  │
│  │ Kokoro-82M   │──▶│ edge-tts                 │──▶│ Synthetic WAV    │  │
│  │ ONNX (WAV)   │   │ en-US-ChristopherNeural  │   │ silence fallback │  │
│  │ models/*.onnx│   │ MP3 stream               │   │ RIFF 24kHz       │  │
│  └──────────────┘   └──────────────────────────┘   └──────────────────┘  │
│              streaming bytes ──▶ WebSocket send_bytes()                   │
└─────────────────────────────────────────────────────────────────────────┘
```

**Request flow (voice):** Mic → `AnalyserNode` (VAD) + `SpeechRecognition` (STT final) → WS `query` → `AgentRequest` → `route_request()` → 7-step orchestrator (telemetry + RAG + LLM) → `text_chunk` + streaming audio chunks → canvas analyser + audio queue playback. `interrupt` cancels TTS `asyncio.Task` mid-stream.

---

## 🧰 Tech Stack

| Category | Tool / Library | Version | Role |
|---|---|---|---|
| **Runtime** | Python | `3.11-slim` (Docker) | Application runtime |
| **Framework** | FastAPI | `>=0.110.0` | HTTP + WebSocket server, `app/main.py` |
| **Server** | Uvicorn | `>=0.28.0` | ASGI server (`uvicorn app.main:app --reload`) |
| **LLM Gateway** | OpenAI SDK (OpenRouter) | `>=1.14.0` | `route_request()` + `_call_llm()` via `https://openrouter.ai/api/v1` |
| **Validation** | Pydantic | `>=2.6.0` | `AgentRequest/Response`, `RouteDecision`, `MaintenanceRequest` |
| **Vector DB** | Qdrant | `qdrant/qdrant:latest` (Docker) | `hvac_knowledge` collection, cosine 384-dim |
| **Qdrant Client** | qdrant-client | `>=1.8.0` | `QdrantClient`, `query_points()`, `upsert()` |
| **Embedding** | FastEmbed | `>=0.2.0` | `TextEmbedding("BAAI/bge-small-en-v1.5")` |
| **Reranker** | sentence-transformers | `>=3.0.1` | `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")` |
| **Tool Protocol** | mcp | `>=1.0.0` | `MCPServer`, `Tool` registry (8 tools) |
| **Config** | python-dotenv | `>=1.0.1` | `load_dotenv()` for `.env` |
| **Templating** | Jinja2 | `>=3.1.3` | (transitive, FastAPI templating) |
| **HTTP** | requests | `>=2.31.0` | LLM calls in orchestrator/RAG |
| **Sockets** | websockets | `>=12.0` | WebSocket client compatibility |
| **Primary TTS** | kokoro-onnx | `>=0.3.1` | `Kokoro(model, voices)` ONNX neural TTS (WAV) |
| **Fallback TTS** | edge-tts | `>=6.1.9` | `Communicate(voice="en-US-ChristopherNeural")` MP3 streaming |
| **TTS Runtime** | onnxruntime | `>=1.17.0` | Kokoro inference |
| **Frontend STT** | Browser Web Speech API | native | `SpeechRecognition / webkitSpeechRecognition` (client-side) |
| **Frontend VAD** | Silero-inspired RMS | native JS | `AnalyserNode.getByteFrequencyData()` + RMS barge-in |
| **Frontend Canvas** | HTML5 Canvas 2D | native | 3D audio spectrum visualizer + sphere |
| **Styling** | Custom Glassmorphism CSS | — | `rgba(18,24,38,0.75)`, Inter/Outfit, neon gradients |
| **Infra** | Docker + Compose | `3.8` | `nectar-agent` (port `8002:8000`) + `qdrant` (ports `6335:6333`, `6336:6334`) |
| **Audio Sysdep** | FFmpeg + PortAudio | apt | `portaudio19-dev`, `ffmpeg` in `Dockerfile` |

**Key Models:**

| Model | Purpose | Config Key |
|---|---|---|
| `dots-studio/dots-3-note-preview:free` | Default LLM (orchestrator diagnosis + RAG grounding) | `DEFAULT_MODEL` (fallback `google/gemini-1.5-flash-exp:free`) |
| `dots-studio/dots-3-note-preview:free` | Intent router classification | `ROUTER_MODEL` (fallback `meta-llama/llama-3.3-70b-instruct:free`) |
| `BAAI/bge-small-en-v1.5` | Dense embedding (384-dim) | `app/rag/vector_store.py:14` |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker (MS-MARCO fine-tuned) | `app/rag/retriever.py:13` |
| `Kokoro-82M` | Neural TTS ONNX | `models/kokoro-*.onnx` / `KOKORO_MODEL_PATH` |
| `en-US-ChristopherNeural` | Edge-TTS voice | `app/voice/tts.py:13` |

---

## 🚀 Quickstart & Installation Guide

### Prerequisites

- Docker & Docker Compose (recommended) **or** Python 3.11+
- An OpenRouter API key ([openrouter.ai](https://openrouter.ai))

### 1. Environment Setup

Create your environment file from the example:

```bash
# Linux / macOS
cp .env.example .env

# Windows (PowerShell)
Copy-Item .env.example .env
```

> **Note:** If `.env.example` is not present (it is git-ignored by template), create `.env` manually with the variables below.

Required variables in `.env` (`app/config.py:7-18`):

```env
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxx
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
DEFAULT_MODEL=dots-studio/dots-3-note-preview:free
ROUTER_MODEL=dots-studio/dots-3-note-preview:free
HOST=0.0.0.0
PORT=8000
QDRANT_HOST=qdrant        # use "localhost" for local non-Docker runs
QDRANT_PORT=6333
# Optional Kokoro overrides
# KOKORO_MODEL_PATH=models/kokoro-v1.0.onnx
# KOKORO_VOICES_PATH=models/voices-v1.0.bin
```

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | `""` | OpenRouter auth token (required for LLM + routing) |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter endpoint |
| `DEFAULT_MODEL` | `google/gemini-1.5-flash-exp:free` | Diagnosis & RAG generation model |
| `ROUTER_MODEL` | `meta-llama/llama-3.3-70b-instruct:free` | Intent classification model |
| `HOST` | `0.0.0.0` | FastAPI bind host |
| `PORT` | `8000` | FastAPI port (compose maps to `8002` on host) |
| `QDRANT_HOST` | `localhost` | Qdrant host (`qdrant` inside compose) |
| `QDRANT_PORT` | `6333` | Qdrant port |

### 2. Docker Compose Initialization (Recommended)

```bash
# Build and start both services (FastAPI + Qdrant)
docker compose up --build

# Or detached
docker compose up --build -d

# Tail logs
docker compose logs -f nectar-agent
docker compose logs -f qdrant

# Stop
docker compose down

# Full reset (including vector data)
docker compose down -v
```

**Compose topology** (`docker-compose.yml:1-31`):

| Service | Container | Host → Container Ports | Volume |
|---|---|---|---|
| `nectar-agent` | `nectar_ai_agent` | `8002:8000` | `.:/app` (live reload) |
| `qdrant` | `nectar_qdrant` | `6335:6333` (REST), `6336:6334` (gRPC) | `nectar_qdrant_data:/qdrant/storage` |

> Ports `6335/6336` are offset to avoid collision with other Qdrant instances on `6333/6334`.

On startup, `app/main.py:39-48` auto-indexes `app/rag/docs/hvac_faq.txt` into Qdrant if the DB is reachable; if Qdrant is offline the warning is logged and startup continues.

### 3. Local (Non-Docker) Run

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt

# Start Qdrant via Docker standalone if not using compose:
docker run -p 6333:6333 -p 6334:6334 -v %cd%/qdrant_storage:/qdrant/storage qdrant/qdrant:latest
# (or docker run -p 6333:6333 ... on Linux)

# Ensure .env has QDRANT_HOST=localhost
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000/` (local) or `http://localhost:8002/` (compose).

---

## 🌐 Live Application Endpoints

| Endpoint | URL | Method | Description |
|---|---|---|---|
| **Web UI** | `http://localhost:8000/` (local) · `http://localhost:8002/` (compose) | `GET` | Glassmorphism voice interface (`app/templates/index.html`) |
| **API Docs (Swagger)** | `http://localhost:8000/docs` · `http://localhost:8002/docs` | `GET` | Interactive OpenAPI docs (FastAPI auto-generated) |
| **ReDoc** | `http://localhost:8000/redoc` | `GET` | Alternative API documentation |
| **OpenAPI JSON** | `http://localhost:8000/openapi.json` | `GET` | Raw OpenAPI schema |
| **Qdrant Dashboard** | `http://localhost:6333/dashboard` (local) · `http://localhost:6335/dashboard` (compose) | `GET` | Qdrant collection explorer (`hvac_knowledge`) |
| **Health (Voice)** | `http://localhost:8000/api/v1/voice/health` | `GET` | `{"status":"ok","voice":"kokoro/edge-tts"}` |

### REST API Summary

| Path | Method | Handler | Body / Params |
|---|---|---|---|
| `/api/v1/agent/chat` | `POST` | `app/main.py:121` `api_agent_chat` | `AgentRequest{query, context}` → `AgentResponse` |
| `/api/v1/voice/chat` | `POST` | `app/voice/voice_router.py:186` `voice_chat` | `{query|text|q, context?}` (JSON/form/query) → `{response_text, execution_steps, pending_action, audio_base64}` |
| `/api/v1/voice/health` | `GET` | `voice_router.py:293` | health check |
| `/api/v1/mcp/asset/{asset_id}/details` | `GET` | `app/main.py:54` | asset metadata |
| `/api/v1/mcp/asset/{asset_id}/status` | `GET` | `app/main.py:61` | operational status |
| `/api/v1/mcp/sensor/{asset_id}` | `GET` | `app/main.py:68` | sensor readings |
| `/api/v1/mcp/energy/{asset_id}` | `GET` | `app/main.py:75` | energy consumption |
| `/api/v1/mcp/alerts/{asset_id_or_building}` | `GET` | `app/main.py:82` | warnings + critical alarms |
| `/api/v1/mcp/relationships/{asset_id}` | `GET` | `app/main.py:89` | topology graph |
| `/api/v1/mcp/action/service_request` | `POST` | `app/main.py:96` | `create_service_request` (requires `confirmed`) |
| `/api/v1/mcp/action/service_request` | `PUT` | `app/main.py:108` | `update_service_request` (requires `confirmed`) |
| `/ws/agent/voice` | `WebSocket` | `app/voice/voice_router.py:25` | streaming voice agent (see API Spec below) |

---

## 🧪 Step-by-Step Test Walkthrough Scenario

### Scenario: _“Why is Building A hot?”_  — Autonomous 7-Step Diagnostics → Confirmation → Ticket

This is the canonical `RAG_MCP_REASONING` demonstration path exercised by the test suite and UI. The orchestrator treats _“hot office on third floor”_ / _“Why is Building A hot?”_ as a diagnostics request requiring both live telemetry and grounded documentation.

#### Preconditions

- Qdrant running and `hvac_knowledge` collection indexed (auto-index on startup).
- `OPENROUTER_API_KEY` set (or offline fallbacks still produce a valid diagnosis).
- Open `http://localhost:8002/` — confirm badge shows **Live** and canvas visualizer is active.

#### Execution

**1. User speaks or types the trigger query**

- **Voice:** Click the mic (center of canvas) → allow microphone → say _“Why is the third floor office so hot? Investigate and handle maintenance if required.”_
- **Text fallback:** Type in the bottom input bar: `Why is Building A hot? The third floor office is 28 degrees, investigate and create maintenance if needed.` → press **Send**.

The client calls:

```js
ws.send(JSON.stringify({ type: "query", text: "Why is Building A hot? The third floor office is 28 degrees, investigate and create maintenance if needed." }))
```

`latencyStart = Date.now()` is recorded; `.step` drawer resets.

**2. Observe the 7-step execution drawer (right panel)**

Each step animates at ~320 ms intervals as `execution_steps` arrives in the `text_chunk`:

| # | UI Label | Orchestrator Log (`execution_steps[i]`) |
|---|----------|------------------------------------------|
| 1 | Identify Asset | `Step 1 (Identify Building/Floor/Asset): Extracted {building: Building A, floor: 3rd} -> resolved to asset AHU-02` |
| 2 | Telemetry | `Step 2 (Get Sensor Readings): Called get_sensor_data(AHU-02) -> {"temperature":22.0,"pressure":65.0,"airflow":85.0,"vibration":3.5}` |
| 3 | Asset Status | `Step 3 (Find Related HVAC Assets & Status): Called get_asset_relationships(AHU-02) -> {"connected_chillers":["Chiller-01","Chiller-02"]} and get_asset_status(AHU-02) -> {"operational_status":"Warning","health":"Low Airflow"}` |
| 4 | Alerts | `Step 4 (Check Active Alerts): Called get_active_alerts(AHU-02) -> {"warnings":["Low airflow detected"]}` |
| 5 | RAG Retrieval | `Step 5 (RAG Troubleshooting Retrieval): Called two-stage reranked query_rag_agent('...hot office low airflow AHU...') -> retrieved troubleshooting guide` |
| 6 | LLM Synthesis | `Step 6 (LLM Diagnostic Synthesis): Passed telemetry + alerts + RAG into LLM (dots-studio/dots-3-note-preview:free) -> diagnosis: Low airflow detected in AHU-02 ... clogged MERV 13 filters ...` |
| 7 | Confirmation Decision | `Step 7 (Decide Maintenance & Ask Confirmation): Maintenance IS required for AHU-02 with High priority. Pending action created, asking user confirmation.` |

Telemetry cards on the right remain pinned to `AHU-02 / Building A · North Wing — 28°C / 85 CFM / MERV13 Warning`.

**3. Read the agent response**

`#responseBox` shows:

```
Investigation complete for the hot office on the third floor (asset AHU-02, Building A North Wing):

Diagnosis: Low airflow detected in AHU-02 (85 CFM) with Warning/Low Airflow status and active low airflow alert. Probable cause is clogged MERV 13 filters or fan malfunction requiring maintenance.

Telemetry: {"temperature":22.0,"pressure":65.0,"airflow":85.0,"vibration":3.5} | Status: {"operational_status":"Warning","health":"Low Airflow"...} | Alerts: {"warnings":["Low airflow detected"]}
RAG guidance confirms filter/fan issue.

Maintenance IS required. Would you like me to create a High-priority maintenance request for AHU-02? Please confirm with YES to proceed.
```

Audio playback begins immediately (MP3 chunks via `send_bytes()` → `AnalyserNode` visualizer in speaking mode).

**4. Confirmation modal interaction**

Because `requires_action_confirmation == true` and `pending_action == {action:"create_service_request", asset_id:"AHU-02", priority:"High"}`, the UI calls `showModal()`:

- **Modal title:** `Maintenance Approval Required`
- **Fields:** `Proposed Action: create_service_request`, `Asset ID: AHU-02`, `Priority: High`, `Probable Cause: clogged MERV 13 filters...`
- **Action A — Approve:** Click **`Approve & Open Ticket`** (or say/type `YES`, `confirm`, `proceed`):

  ```js
  ws.send(JSON.stringify({
    type: "query",
    text: "YES",
    context: { pending_action: {action:"create_service_request", asset_id:"AHU-02", issue:"...", priority:"High"}, confirmed: true }
  }))
  ```

  Orchestrator detects `context.pending_action + confirmed` in `process_agent_query:168-187`, calls `create_service_request(..., confirmed=True)` → `Ticket ID: TICK-1001`.

  Response:

  ```
  Service request created successfully! Ticket ID: TICK-1001. Service request created successfully. Ticket ID: TICK-1001 Maintenance team has been notified for AHU-02.
  ```

  Toast: `Approved — creating ticket…` then `Ticket TICK-1001`.

- **Action B — Cancel:** Click **`Cancel`** or dismiss by clicking outside the modal — ticket is not created; drawer resets for a new query.

**5. Verify via REST (optional)**

```bash
curl http://localhost:8002/api/v1/mcp/alerts/AHU-02 | jq
curl http://localhost:8002/api/v1/mcp/sensor/AHU-02 | jq
curl -X POST http://localhost:8002/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"Why is Building A hot? third floor 28C","context":null}' | jq
```

**6. Barge-in test (interrupt while speaking)**

While the agent is in `Speaking` state (canvas pulsing violet), speak loudly — the VAD RMS counter exceeds threshold, `triggerBargeIn()` sends `{"type":"interrupt"}`, server cancels the TTS task, client stops all `AudioBufferSourceNode`s and shows `⚡ Interrupted`. You can immediately issue a new query.

---

## 🔌 API Specifications

### WebSocket — `WS /ws/agent/voice`

`app/voice/voice_router.py:25-179` · `binaryType = "arraybuffer"` · single duplex connection; server accepts one `active_tts_task` at a time and cancels it on `interrupt` or on new `query`.

**Connect:**

```js
const WS_PATH = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws/agent/voice';
const ws = new WebSocket(WS_PATH);
ws.binaryType = 'arraybuffer';
```

#### Client → Server Messages

**1. Query frame**

```json
{
  "type": "query",
  "text": "Why is Building A hot? third floor office at 28 degrees",
  "context": null
}
```

- `text` (aliases: `query`) — required, non-empty string.
- `context` — optional object. For confirmation:

```json
{
  "type": "query",
  "text": "YES",
  "context": {
    "pending_action": {
      "action": "create_service_request",
      "asset_id": "AHU-02",
      "issue": "Low airflow due to clogged MERV 13 filters",
      "priority": "High"
    },
    "confirmed": true
  }
}
```

**2. Interrupt frame (barge-in)**

```json
{ "type": "interrupt" }
```

Cancels the active `generate_audio_stream()` task. Server acks:

```json
{ "type": "interrupted", "status": "cancelled" }
```

**3. Unknown type → error**

```json
{ "type": "error", "error": "Unknown message type: ..." }
```

#### Server → Client Messages

**1. Text chunk (JSON, always first)**

```json
{
  "type": "text_chunk",
  "text": "Investigation complete for the hot office on the third floor ... Would you like me to create a High-priority maintenance request ...",
  "requires_confirmation": true,
  "requires_action_confirmation": true,
  "pending_action": {
    "action": "create_service_request",
    "asset_id": "AHU-02",
    "issue": "Low airflow detected in AHU-02 ...",
    "priority": "High"
  },
  "route_used": "RAG_MCP_REASONING",
  "execution_steps": [
    "Routed to: RAG_MCP_REASONING (confidence: 0.95) - ...",
    "Step 1 (Identify Building/Floor/Asset): ...",
    "Step 2 (Get Sensor Readings): ...",
    "Step 3 (Find Related HVAC Assets & Status): ...",
    "Step 4 (Check Active Alerts): ...",
    "Step 5 (RAG Troubleshooting Retrieval): ...",
    "Step 6 (LLM Diagnostic Synthesis): ...",
    "Step 7 (Decide Maintenance & Ask Confirmation): ..."
  ],
  "user_query": "Why is Building A hot?"
}
```

`requires_confirmation` and `requires_action_confirmation` are aliases (both sent). If `true`, client must show the confirmation modal.

**2. Binary audio stream chunks**

Raw `ArrayBuffer` — `await websocket.send_bytes(chunk)` per TTS chunk (MP3 from edge-tts, WAV from Kokoro/synthetic). Client decodes via `AudioContext.decodeAudioData()` (WAV path) or `HTMLAudio Blob` fallback (MP3).

**3. Stream termination**

```json
{ "type": "audio_complete" }
```

Sent once streaming finishes without cancellation. Suppressed if `interrupt` cancelled the task.

**4. TTS error**

```json
{ "type": "tts_error", "error": "..." }
```

**5. Invalid frame**

```json
{ "type": "error", "error": "Invalid JSON" }
{ "type": "error", "error": "Missing query text" }
```

#### REST voice fallback — `POST /api/v1/voice/chat`

`app/voice/voice_router.py:186-289` — accepts both JSON and form; generates full audio non-streaming.

**Request (JSON):**

```bash
curl -X POST http://localhost:8002/api/v1/voice/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"Why is Building A hot? third floor","context":null}'
```

Also accepts `{"text":"..."}` or `{"q":"..."}` and `form-data` / `query_params`.

**Response:**

```json
{
  "user_query": "Why is Building A hot? third floor",
  "response_text": "Investigation complete ... Maintenance IS required. Would you like me to create ...",
  "execution_steps": ["Routed to: RAG_MCP_REASONING ...", "Step 1 ...", "..."],
  "requires_action_confirmation": true,
  "pending_action": { "action": "create_service_request", "asset_id": "AHU-02", "priority": "High" },
  "route_used": "RAG_MCP_REASONING",
  "audio_base64": "<base64 MP3 or WAV>",
  "audio": "<alias of audio_base64>",
  "audio_format": "mp3",
  "audio_bytes_length": 48210
}
```

- `audio_format` is `"mp3"` (edge-tts) or `"wav"` (Kokoro/synthetic, detected via `RIFF` header).
- Validate: `audio_bytes_length > 0`; `audio_base64` decodes to playable media.

#### Agent direct — `POST /api/v1/agent/chat`

`app/main.py:121-124` · `app/agent/models.py:5-16`

```bash
curl -X POST http://localhost:8002/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"What is an AHU?","context":null}'
```

Response:

```json
{
  "user_query": "What is an AHU?",
  "route_used": "RAG",
  "response_text": "An AHU (Air Handling Unit) is ...",
  "requires_action_confirmation": false,
  "pending_action": null,
  "execution_steps": ["Routed to: RAG ...", "RAG: Retrieved reranked context ..."]
}
```

---

## 🤗 Hugging Face Spaces Deployment Guide

Deploy as a **Docker Space** (persistent FastAPI + Qdrant sidecar or embedded).

### Option A — Docker Space (Recommended for this repo)

Works verbatim with the included `Dockerfile:1-27` and `docker-compose.yml:1-31`.

**1. Create the Space**

- Go to [huggingface.co/new-space](https://huggingface.co/new-space)
- **Space name:** `nectar-facility-voice-agent`
- **SDK:** **Docker**
- **Hardware:** CPU basic (upgrade to larger if loading Kokoro models)
- **Visibility:** Public / Private

**2. Push this repo**

```bash
# Inside the Space's git clone, copy project files:
# Or push directly if the HF repo is the remote:
git remote add space https://huggingface.co/spaces/<username>/nectar-facility-voice-agent
git push space main
```

Required files at Space root:

```
Dockerfile
requirements.txt
app/
data/  (optional seed)
```

**3. Hugging Face Dockerfile notes**

HF Spaces sets `PORT=7860` automatically and expects the app to bind to it. Override the default `CMD` or ensure `app/config.py` reads `PORT` from env (it does — `int(os.getenv("PORT", 8000))`).

Add at top of `Dockerfile` or via Space `README.md` frontmatter:

```dockerfile
# HF Spaces provides PORT=7860; ensure Uvicorn honors it
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
```

If Qdrant runs as a separate container, use HF's Docker Compose support or embed Qdrant as an in-process client; alternatively switch to Qdrant Cloud and set `QDRANT_HOST` / `QDRANT_PORT` via Space **Settings → Variables and Secrets**.

**4. Space Secrets (Settings → Variables and Secrets)**

| Secret | Value |
|---|---|
| `OPENROUTER_API_KEY` | `sk-or-v1-...` |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` |
| `DEFAULT_MODEL` | `dots-studio/dots-3-note-preview:free` |
| `ROUTER_MODEL` | `dots-studio/dots-3-note-preview:free` |
| `QDRANT_HOST` | `qdrant` (if sidecar) or Qdrant Cloud URL |
| `QDRANT_PORT` | `6333` |
| `KOKORO_MODEL_PATH` | `models/kokoro-v1.0.onnx` (if bundling weights — ensure `.gitignore` allows) |

> **Large model files:** Kokoro ONNX (`~80 MB`) + voices (`.bin`) are git-ignored by default (`models/*.onnx`, `models/*.bin`). For HF Spaces, either commit them via `git lfs` or rely on the `edge-tts` fallback (no model files needed).

**5. Verify deployment**

- Space logs should show `Uvicorn running on http://0.0.0.0:7860`
- `GET https://<username>-nectar-facility-voice-agent.hf.space/docs` → Swagger loads
- `WS wss://<username>-nectar-facility-voice-agent.hf.space/ws/agent/voice` → `Live` badge

### Option B — Qdrant Cloud (no sidecar)

If running a single container (no compose), point to managed Qdrant:

```env
QDRANT_HOST=<your-cluster>.qdrant.io
QDRANT_PORT=6333
# Optional: QDRANT_API_KEY if using authenticated cloud (extend vector_store.py)
```

Auto-index still runs on startup; first request may take a few seconds while `BAAI/bge-small-en-v1.5` and `ms-marco-MiniLM-L-6-v2` download and cache.

### Production checklist

- [ ] Set `OPENROUTER_API_KEY` as a **Secret** (not Variable) so it is not exposed in logs.
- [ ] Pin model versions in `.env` to avoid free-model deprecation drift.
- [ ] Warm Qdrant: hit `GET /dashboard` → confirm `hvac_knowledge` shows points > 0.
- [ ] Test the Building A walkthrough end-to-end over `wss://`.
- [ ] Enable **Sleep** behavior awareness: HF Spaces sleep after inactivity; Qdrant in-memory embeddings reload on wake — first query will be slower.

---

## 📁 Project Structure

```
nectar-ai-agent/
├── app/
│   ├── main.py                 # FastAPI app, auto-index, MCP & agent routes, UI serve
│   ├── config.py               # Config (OPENROUTER, QDRANT, HOST/PORT via python-dotenv)
│   ├── agent/
│   │   ├── orchestrator.py     # 7-step RAG_MCP_REASONING + routing + confirmation logic
│   │   └── models.py           # AgentRequest / AgentResponse schemas
│   ├── router/
│   │   ├── intent_router.py    # LLM intent classifier (6 routes) + JSON extraction + fallback
│   │   └── models.py           # RouteType enum + RouteDecision
│   ├── rag/
│   │   ├── vector_store.py     # FastEmbed BAAI/bge-small-en-v1.5 + Qdrant hvac_knowledge
│   │   ├── retriever.py        # Two-stage: search top 15 → CrossEncoder rerank top 3 → LLM
│   │   └── docs/hvac_faq.txt   # 385 lines · 8 sections HVAC knowledge base
│   ├── mcp_tools/
│   │   ├── telemetry.py        # 6 read tools (asset/sensor/energy/alerts/relationships)
│   │   ├── actions.py          # 2 write tools (create/update service request + guardrails)
│   │   ├── mcp_server.py       # MCPServer registry (8 Tool descriptors)
│   │   └── models.py           # MaintenanceRequest + TelemetryData
│   ├── voice/
│   │   ├── voice_router.py     # WS /ws/agent/voice + REST /api/v1/voice/chat
│   │   ├── tts.py              # Kokoro-82M ONNX → edge-tts → synthetic fallback
│   │   ├── tts_handler.py      # (reserved)
│   │   └── stt_handler.py      # (reserved — STT is client-side Web Speech API)
│   └── templates/
│       └── index.html          # Glassmorphism UI · Canvas visualizer · VAD · STT · modal
├── data/                       # (optional seed assets)
├── eval/                       # evaluation harnesses
├── tests/                      # automated tests
├── docker-compose.yml          # nectar-agent (8002:8000) + qdrant (6335:6333, 6336:6334)
├── Dockerfile                  # python:3.11-slim + ffmpeg/portaudio + uvicorn --reload
├── requirements.txt            # fastapi, qdrant-client, fastembed, sentence-transformers, mcp, edge-tts, kokoro-onnx, ...
├── .env                        # local secrets (git-ignored)
└── README.md                   # this file
```

---

## 🔐 Environment Reference

| Variable | Source | Required | Example |
|---|---|---|---|
| `OPENROUTER_API_KEY` | `app/config.py:8` | ✅ for LLM | `sk-or-v1-...` |
| `OPENROUTER_BASE_URL` | `app/config.py:9` | — | `https://openrouter.ai/api/v1` |
| `DEFAULT_MODEL` | `app/config.py:12` | — | `dots-studio/dots-3-note-preview:free` |
| `ROUTER_MODEL` | `app/config.py:13` | — | `dots-studio/dots-3-note-preview:free` |
| `HOST` | `app/config.py:16` | — | `0.0.0.0` |
| `PORT` | `app/config.py:17` | — | `8000` (HF sets `7860`) |
| `QDRANT_HOST` | `vector_store.py:9` / compose | — | `localhost` / `qdrant` |
| `QDRANT_PORT` | `vector_store.py:10` | — | `6333` |
| `KOKORO_MODEL_PATH` | `tts.py:33` | — | `models/kokoro-v1.0.onnx` |
| `KOKORO_VOICES_PATH` | `tts.py:35` | — | `models/voices-v1.0.bin` |

---

## 🧪 Running Tests

```bash
# Inside venv or nectar-agent container
pytest -v

# Targeted
pytest tests/test_orchestrator.py -v
pytest tests/test_voice.py -v
```

Tests cover: intent routing fallbacks, telemetry mocks, RAG offline fallback, orchestrator 7-step trace, confirmation guardrails, and WebSocket `query`/`interrupt`/`text_chunk` contracts.

---

## 📝 License

MIT — see `LICENSE` (if present). Facility knowledge base (`app/rag/docs/hvac_faq.txt`) is synthetic demo content.

---

<p align="center">
  Built for autonomous facility operations · Edge voice · Grounded reasoning · Safe actuation<br/>
  <b>Nectar</b> — where HVAC meets agentic AI.
</p>

