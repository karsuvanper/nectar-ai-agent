import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from app.config import config
from app.router.intent_router import route_request
from app.router.models import RouteDecision
from app.rag.retriever import query_rag_agent
from app.agent.models import AgentRequest, AgentResponse
from app.agent.orchestrator import process_agent_query
from app.mcp_tools.telemetry import (
    get_asset_details,
    get_asset_status,
    get_sensor_data,
    get_energy_consumption,
    get_active_alerts,
    get_asset_relationships,
)
from app.mcp_tools.actions import create_service_request, update_service_request
from app.voice.voice_router import router as voice_router

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup lifespan that blocks Uvicorn until warmup finishes.

    Ensures model loading and fastembed initializations happen inside lifespan
    so health checks don't trigger premature shutdown, and Qdrant fallback
    is handled gracefully.
    """
    logger.info("Lifespan startup: warming up models and Qdrant...")
    # 1. Warmup embedding model (fastembed) - blocks until download finishes
    try:
        from app.rag.vector_store import warmup_embedding
        warmup_embedding()
        logger.info("Lifespan: embedding model warmup complete")
    except Exception as e:
        logger.warning(f"Lifespan: embedding warmup skipped/failed: {e}")
    # 2. Warmup reranker model - blocks until download finishes
    try:
        from app.rag.retriever import warmup_reranker
        warmup_reranker()
        logger.info("Lifespan: reranker warmup complete")
    except Exception as e:
        logger.warning(f"Lifespan: reranker warmup skipped/failed: {e}")
    # 3. Auto-index HVAC docs (uses Qdrant in-memory fallback if external unavailable)
    try:
        from app.rag.vector_store import index_documents
        with open("app/rag/docs/hvac_faq.txt", "r", encoding="utf-8") as f:
            documents = [line.strip() for line in f.readlines() if line.strip()]
        index_documents(documents)
        logger.info(f"Lifespan: auto-indexed {len(documents)} documents")
    except Exception as e:
        logger.warning(f"Auto-index skipped (Qdrant offline or warmup incomplete): {e}")
    yield
    logger.info("Lifespan shutdown")

app = FastAPI(
    title="Nectar Intelligent Facility Operations AI Agent",
    description="Autonomous Voice & Multi-Agent Platform powered by OpenRouter, RAG, and MCP Tools",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Keep legacy startup event for backward compatibility (lifespan already handles warmup)
@app.on_event("startup")
def auto_index_legacy():
    try:
        from app.rag.vector_store import index_documents
        with open("app/rag/docs/hvac_faq.txt", "r", encoding="utf-8") as f:
            documents = [line.strip() for line in f.readlines() if line.strip()]
        index_documents(documents)
    except Exception as e:
        logging.getLogger(__name__).warning(f"Auto-index (legacy) skipped: {e}")

# Thin API wrappers that delegate to the centralized MCP tool functions.
# Business logic lives in app/mcp_tools/telemetry.py and app/mcp_tools/actions.py.
# These endpoints simply translate HTTP requests into function calls.

@app.get("/api/v1/mcp/asset/{asset_id}/details")
def api_asset_details(asset_id: str):
    result = get_asset_details(asset_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

@app.get("/api/v1/mcp/asset/{asset_id}/status")
def api_asset_status(asset_id: str):
    result = get_asset_status(asset_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

@app.get("/api/v1/mcp/sensor/{asset_id}")
def api_sensor_data(asset_id: str):
    result = get_sensor_data(asset_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

@app.get("/api/v1/mcp/energy/{asset_id}")
def api_energy_consumption(asset_id: str):
    result = get_energy_consumption(asset_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

@app.get("/api/v1/mcp/alerts/{asset_id_or_building}")
def api_active_alerts(asset_id_or_building: str):
    result = get_active_alerts(asset_id_or_building)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

@app.get("/api/v1/mcp/relationships/{asset_id}")
def api_asset_relationships(asset_id: str):
    result = get_asset_relationships(asset_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

@app.post("/api/v1/mcp/action/service_request")
def api_create_service_request(
    asset_id: str = Body(...),
    issue: str = Body(...),
    priority: str = Body("Medium"),
    confirmed: bool = Body(False),
):
    result = create_service_request(asset_id, issue, priority, confirmed)
    if not result.is_created:
        return result
    return {"ticket_id": result.ticket_id, "message": result.message}

@app.put("/api/v1/mcp/action/service_request")
def api_update_service_request(
    ticket_id: str = Body(...),
    status: str = Body(...),
    notes: str = Body(""),
    confirmed: bool = Body(False),
):
    result = update_service_request(ticket_id, status, notes, confirmed)
    if result.get("requires_confirmation", False):
        return result
    return result


@app.post("/api/v1/agent/chat", response_model=AgentResponse)
def api_agent_chat(request: AgentRequest):
    response = process_agent_query(request)
    return response

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve ultra-luxurious glassmorphism UI."""
    candidates = [
        Path(__file__).parent / "templates" / "index.html",
        Path("app/templates/index.html"),
        Path("app/static/index.html"),
    ]
    for p in candidates:
        if p.is_file():
            return HTMLResponse(content=p.read_text(encoding="utf-8"), status_code=200)
    return HTMLResponse(content="<h1>Nectar UI not found</h1>", status_code=404)


# Voice router: exposes /ws/agent/voice and /api/v1/voice/chat
app.include_router(voice_router)
