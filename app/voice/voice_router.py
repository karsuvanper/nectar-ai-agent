"""Voice Router: WebSocket /ws/agent/voice and REST POST /api/v1/voice/chat."""

import asyncio
import base64
import json
import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse

from app.agent.models import AgentRequest, AgentResponse
from app.agent.orchestrator import process_agent_query
from app.voice.tts import generate_audio_stream

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# WebSocket endpoint /ws/agent/voice
# ---------------------------------------------------------------------------

@router.websocket("/ws/agent/voice")
async def voice_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for streaming voice agent.

    Query Frame: {"type": "query", "text": "...", "context": ...}
    -> process via orchestrator, send text_chunk + streaming binary audio.
    Interruption: {"type": "interrupt"} -> cancel active TTS task.
    """
    await websocket.accept()
    active_tts_task: Optional[asyncio.Task] = None
    # Queue to track if we should suppress further audio after interrupt
    try:
        while True:
            try:
                # Receive text or bytes; spec says JSON messages
                message = await websocket.receive_text()
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected")
                break
            except Exception as e:
                logger.warning(f"WebSocket receive error: {e}")
                break

            # Parse JSON
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "error": "Invalid JSON"})
                continue

            msg_type = data.get("type")

            # --- Interruption handling ---
            if msg_type == "interrupt":
                if active_tts_task and not active_tts_task.done():
                    logger.info("Interruption signal: cancelling TTS task")
                    active_tts_task.cancel()
                    try:
                        await active_tts_task
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        logger.debug(f"TTS task cancellation error: {e}")
                    active_tts_task = None
                # Flush / ack
                try:
                    await websocket.send_json({"type": "interrupted", "status": "cancelled"})
                except Exception:
                    pass
                continue

            # --- Query handling ---
            if msg_type == "query":
                text = data.get("text") or data.get("query") or ""
                context = data.get("context")
                if not text or not text.strip():
                    await websocket.send_json({"type": "error", "error": "Missing query text"})
                    continue

                # Cancel any prior streaming
                if active_tts_task and not active_tts_task.done():
                    active_tts_task.cancel()
                    try:
                        await active_tts_task
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        pass

                # Process orchestrator (sync function) in thread to avoid blocking
                try:
                    agent_req = AgentRequest(query=text, context=context)
                    # Use to_thread for non-blocking
                    response: AgentResponse = await asyncio.to_thread(process_agent_query, agent_req)
                except Exception as e:
                    logger.exception(f"Orchestrator error: {e}")
                    await websocket.send_json({"type": "error", "error": str(e)})
                    continue

                # Send text metadata frame
                try:
                    await websocket.send_json({
                        "type": "text_chunk",
                        "text": response.response_text,
                        "requires_confirmation": response.requires_action_confirmation,
                        "requires_action_confirmation": response.requires_action_confirmation,
                        "pending_action": response.pending_action,
                        "route_used": response.route_used,
                        "execution_steps": response.execution_steps,
                        "user_query": response.user_query,
                    })
                except Exception as e:
                    logger.warning(f"Failed to send text_chunk: {e}")
                    continue

                # Stream audio in background task
                async def _stream_audio(resp_text: str):
                    cancelled = False
                    try:
                        async for chunk in generate_audio_stream(resp_text):
                            # Check cancellation
                            await websocket.send_bytes(chunk)
                            # Small yield to allow interruption handling
                            await asyncio.sleep(0)
                    except asyncio.CancelledError:
                        logger.info("TTS streaming cancelled by interrupt")
                        cancelled = True
                        # Do not propagate; suppress audio_complete
                        return
                    except Exception as e:
                        logger.warning(f"TTS streaming error: {e}")
                        try:
                            await websocket.send_json({"type": "tts_error", "error": str(e)})
                        except Exception:
                            pass
                    finally:
                        if not cancelled:
                            try:
                                await websocket.send_json({"type": "audio_complete"})
                            except Exception:
                                pass

                active_tts_task = asyncio.create_task(_stream_audio(response.response_text))
                # Do NOT await here; loop continues to handle interrupts.
                # Optionally add done callback to clear ref
                def _done_callback(t: asyncio.Task):
                    if t.cancelled():
                        logger.debug("TTS task done (cancelled)")
                    elif t.exception():
                        logger.debug(f"TTS task done with exception: {t.exception()}")

                active_tts_task.add_done_callback(_done_callback)
                continue

            # Unknown type
            await websocket.send_json({"type": "error", "error": f"Unknown message type: {msg_type}"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.exception(f"WebSocket unexpected error: {e}")
        try:
            await websocket.close()
        except Exception:
            pass
    finally:
        if active_tts_task and not active_tts_task.done():
            active_tts_task.cancel()
            try:
                await active_tts_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass


# ---------------------------------------------------------------------------
# REST fallback POST /api/v1/voice/chat
# ---------------------------------------------------------------------------

@router.post("/api/v1/voice/chat")
async def voice_chat(request: Request):
    """
    REST fallback for voice chat.

    Accepts:
      - JSON AgentRequest: {"query": "...", "context": {...}}
        or {"text": "..."} or form data with query text.

    Processes query via orchestrator, generates full TTS audio,
    returns JSON with user_query, response_text, execution_steps,
    requires_action_confirmation, pending_action, audio_base64.
    """
    query_text: Optional[str] = None
    context: Optional[dict] = None

    content_type = request.headers.get("content-type", "")

    # Try JSON first if content-type indicates json or if we can parse json
    if "application/json" in content_type:
        try:
            body = await request.json()
            query_text = body.get("query") or body.get("text") or body.get("q")
            context = body.get("context")
        except Exception as e:
            logger.debug(f"JSON parse failed: {e}")
    elif "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        form = await request.form()
        query_text = form.get("query") or form.get("text") or form.get("q")
        ctx_raw = form.get("context")
        if ctx_raw:
            try:
                context = json.loads(ctx_raw) if isinstance(ctx_raw, str) else ctx_raw
            except Exception:
                context = None
    else:
        # Attempt JSON, fallback to form
        try:
            body = await request.json()
            if isinstance(body, dict):
                query_text = body.get("query") or body.get("text") or body.get("q")
                context = body.get("context")
        except Exception:
            pass
        if not query_text:
            try:
                form = await request.form()
                query_text = form.get("query") or form.get("text") or form.get("q")
                ctx_raw = form.get("context")
                if ctx_raw:
                    try:
                        context = json.loads(ctx_raw) if isinstance(ctx_raw, str) else ctx_raw
                    except Exception:
                        pass
            except Exception:
                pass
        # Also try query params
        if not query_text:
            query_text = request.query_params.get("query") or request.query_params.get("text")

    if not query_text or not query_text.strip():
        return JSONResponse(status_code=422, content={"detail": "Missing query/text field"})

    # Process via orchestrator (sync). Run directly; it's fast enough.
    try:
        agent_req = AgentRequest(query=query_text, context=context)
        response: AgentResponse = process_agent_query(agent_req)
    except Exception as e:
        logger.exception(f"Orchestrator error in REST: {e}")
        return JSONResponse(status_code=500, content={"detail": str(e)})

    # Generate complete TTS audio
    audio_bytes = b""
    try:
        async for chunk in generate_audio_stream(response.response_text):
            audio_bytes += chunk
    except Exception as e:
        logger.warning(f"TTS generation failed in REST: {e}")
        audio_bytes = b""

    # If still empty but we have response text, ensure synthetic at least
    if not audio_bytes and response.response_text:
        # Use synthetic fallback directly
        from app.voice.tts import _make_silence_wav
        audio_bytes = _make_silence_wav(duration=0.5)

    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8") if audio_bytes else ""
    # Determine format: edge-tts is mp3, synthetic is wav; detect by header
    audio_format = "mp3"
    if audio_bytes.startswith(b"RIFF"):
        audio_format = "wav"

    return {
        "user_query": response.user_query,
        "response_text": response.response_text,
        "execution_steps": response.execution_steps,
        "requires_action_confirmation": response.requires_action_confirmation,
        "pending_action": response.pending_action,
        "route_used": response.route_used,
        "audio_base64": audio_base64,
        "audio": audio_base64,  # alias for compatibility
        "audio_format": audio_format,
        "audio_bytes_length": len(audio_bytes),
    }


# Optional health endpoint
@router.get("/api/v1/voice/health")
async def voice_health():
    return {"status": "ok", "voice": "kokoro/edge-tts"}
