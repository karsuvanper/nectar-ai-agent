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

app = FastAPI(
    title="Nectar Intelligent Facility Operations AI Agent",
    description="Autonomous Voice & Multi-Agent Platform powered by OpenRouter, RAG, and MCP Tools",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.rag.vector_store import index_documents

@app.on_event("startup")
def auto_index():
    try:
        with open("app/rag/docs/hvac_faq.txt", "r") as f:
            documents = [line.strip() for line in f.readlines() if line.strip()]
        index_documents(documents)
    except Exception as e:
        # Qdrant may be offline during tests/voice verification; don't crash startup
        import logging
        logging.getLogger(__name__).warning(f"Auto-index skipped (Qdrant offline): {e}")

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