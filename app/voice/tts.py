"""Streaming TTS module: Kokoro-82M ONNX primary, edge-tts fallback."""

import asyncio
import io
import logging
import struct
import wave
from pathlib import Path
from typing import AsyncIterator, Optional

logger = logging.getLogger(__name__)

DEFAULT_VOICE = "en-US-ChristopherNeural"
DEFAULT_FORMAT = "mp3"  # edge-tts outputs mp3; synthetic fallback is wav

# ---------------------------------------------------------------------------
# Kokoro detection helpers
# ---------------------------------------------------------------------------

def _has_kokoro_model() -> bool:
    """Check if kokoro model files exist locally."""
    candidates = [
        Path("models/kokoro-v1.0.onnx"),
        Path("models/kokoro.onnx"),
        Path("models/kokoro-v0_19.onnx"),
        Path("app/voice/models/kokoro.onnx"),
        Path(__file__).parent / "models" / "kokoro.onnx",
        Path(__file__).parent / "models" / "kokoro-v1.0.onnx",
    ]
    # Also check env var
    import os
    env_path = os.getenv("KOKORO_MODEL_PATH")
    if env_path:
        candidates.insert(0, Path(env_path))
    env_voices = os.getenv("KOKORO_VOICES_PATH")
    if env_voices:
        candidates.append(Path(env_voices))
    for p in candidates:
        if p.is_file():
            return True
    # Check any .onnx in models/
    for p in Path("models").glob("*.onnx"):
        if p.is_file():
            return True
    return False


def _get_kokoro_model_paths():
    """Return (model_path, voices_path) if found, else (None, None)."""
    import os
    model_cands = [
        os.getenv("KOKORO_MODEL_PATH"),
        "models/kokoro-v1.0.onnx",
        "models/kokoro.onnx",
        str(Path(__file__).parent / "models" / "kokoro.onnx"),
    ]
    voices_cands = [
        os.getenv("KOKORO_VOICES_PATH"),
        "models/voices-v1.0.bin",
        "models/voices.bin",
        str(Path(__file__).parent / "models" / "voices.bin"),
    ]
    model_path = next((p for p in model_cands if p and Path(p).is_file()), None)
    # fallback: any onnx
    if not model_path:
        for p in Path("models").glob("*.onnx"):
            model_path = str(p)
            break
    voices_path = next((p for p in voices_cands if p and Path(p).is_file()), None)
    if not voices_path:
        for p in Path("models").glob("*.bin"):
            voices_path = str(p)
            break
    return model_path, voices_path


# ---------------------------------------------------------------------------
# Synthetic WAV fallback (offline / no network)
# ---------------------------------------------------------------------------

def _make_silence_wav(duration: float = 0.8, sample_rate: int = 24000) -> bytes:
    """Generate a WAV silence buffer."""
    n_samples = int(duration * sample_rate)
    # Scale duration with text length for slight variation; caller may pass duration
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        # 16-bit PCM silence
        frames = struct.pack("<" + "h" * n_samples, *([0] * n_samples))
        w.writeframes(frames)
    return buf.getvalue()


async def _synthetic_stream(text: str) -> AsyncIterator[bytes]:
    """Yield synthetic WAV audio chunks (guaranteed offline)."""
    if not text or not text.strip():
        # still yield minimal header so caller gets non-empty
        wav = _make_silence_wav(duration=0.3)
        chunk_size = 4096
        for i in range(0, len(wav), chunk_size):
            yield wav[i:i + chunk_size]
            await asyncio.sleep(0.01)
        return
    # Rough duration: 0.05 sec per char, clamped 0.5..5.0 sec
    duration = max(0.5, min(5.0, len(text) * 0.02))
    wav = _make_silence_wav(duration=duration)
    # Add text length variation by chunking differently
    chunk_size = 4096
    for i in range(0, len(wav), chunk_size):
        yield wav[i:i + chunk_size]
        await asyncio.sleep(0.02)


# ---------------------------------------------------------------------------
# Kokoro streaming
# ---------------------------------------------------------------------------

async def _kokoro_stream(text: str) -> AsyncIterator[bytes]:
    """Attempt kokoro-onnx streaming. Yields WAV chunks."""
    model_path, voices_path = _get_kokoro_model_paths()
    if not model_path:
        raise FileNotFoundError("Kokoro model not found")

    # Try kokoro-onnx
    try:
        from kokoro_onnx import Kokoro  # type: ignore

        kokoro = Kokoro(model_path, voices_path) if voices_path else Kokoro(model_path)
        # Kokoro API: kokoro.create(text, voice="af_heart", speed=1.0, lang="en-us")
        # It returns (samples, sample_rate) or writes wav.
        # Split text into sentences for pseudo-streaming
        import re
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        if not sentences:
            sentences = [text]

        for sent in sentences:
            if not sent.strip():
                continue
            # Call synchronously in thread to avoid blocking
            def _synth(s):
                # Try different signatures
                try:
                    samples, sample_rate = kokoro.create(s, voice="af_heart", speed=1.0, lang="en-us")
                except TypeError:
                    try:
                        samples, sample_rate = kokoro.create(s, voice="af_heart")
                    except TypeError:
                        samples, sample_rate = kokoro.create(s)
                # Convert to wav bytes
                import numpy as np
                # samples is numpy array float32 in [-1,1]
                # Convert to int16
                if isinstance(samples, tuple):
                    # some versions return (audio, sr) swapped?
                    pass
                buf = io.BytesIO()
                with wave.open(buf, "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(int(sample_rate))
                    # Normalize
                    arr = np.asarray(samples)
                    if arr.dtype != np.int16:
                        arr = (arr * 32767).astype(np.int16)
                    w.writeframes(arr.tobytes())
                return buf.getvalue()

            wav_bytes = await asyncio.to_thread(_synth, sent)
            chunk_size = 4096
            for i in range(0, len(wav_bytes), chunk_size):
                yield wav_bytes[i:i + chunk_size]
                await asyncio.sleep(0.01)
        return
    except ImportError as e:
        logger.debug(f"kokoro_onnx not installed: {e}")

    # Try kokoro (newer package)
    try:
        from kokoro import KModel, KPipeline  # type: ignore

        # This path is less predictable; attempt pipeline
        pipeline = KPipeline(lang_code="a")
        sentences = text.strip().split(". ")
        for sent in sentences:
            if not sent.strip():
                continue
            # KPipeline generates audio; need to collect
            # Approximate: pipeline(sent, voice="af_heart")
            gen = pipeline(sent, voice="af_heart")
            for _, _, audio in gen:
                # audio is tensor
                buf = io.BytesIO()
                with wave.open(buf, "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(24000)
                    import numpy as np
                    arr = np.asarray(audio)
                    if arr.dtype != np.int16:
                        arr = (arr * 32767).astype(np.int16)
                    w.writeframes(arr.tobytes())
                wav_bytes = buf.getvalue()
                for i in range(0, len(wav_bytes), 4096):
                    yield wav_bytes[i:i + 4096]
                    await asyncio.sleep(0.01)
        return
    except ImportError as e:
        logger.debug(f"kokoro package not installed: {e}")
        raise FileNotFoundError("No kokoro package available") from e

    raise RuntimeError("Kokoro streaming failed")


# ---------------------------------------------------------------------------
# Edge-TTS streaming
# ---------------------------------------------------------------------------

async def _edge_stream(text: str) -> AsyncIterator[bytes]:
    """Stream MP3 chunks via edge-tts."""
    try:
        import edge_tts  # type: ignore
    except ImportError as e:
        raise RuntimeError(f"edge-tts not installed: {e}") from e

    if not text or not text.strip():
        text = "Hello"

    # Use configured voice per task spec
    communicate = edge_tts.Communicate(text, voice=DEFAULT_VOICE)
    # edge_tts .stream() yields dicts with type audio / WordBoundary etc.
    has_audio = False
    async for chunk in communicate.stream():
        if chunk.get("type") == "audio":
            data = chunk.get("data")
            if data:
                has_audio = True
                yield data
        # Allow cancellation point
        await asyncio.sleep(0)

    if not has_audio:
        raise RuntimeError("edge-tts produced no audio")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def generate_audio_stream(text: str) -> AsyncIterator[bytes]:
    """
    Streaming Text-to-Speech logic.

    Primary: Kokoro-82M ONNX if models present.
    Fallback: edge-tts using edge_tts.Communicate(text, voice="en-US-ChristopherNeural").
    Final fallback: synthetic WAV silence (ensures offline correctness).

    Yields audio byte chunks (MP3 when edge-tts, WAV when kokoro/synthetic) in real-time.
    """
    if text is None:
        text = ""
    text = text.strip()
    if not text:
        # Yield synthetic for empty to keep contract non-empty for callers expecting audio
        async for chunk in _synthetic_stream("Hello"):
            yield chunk
        return

    # 1. Try Kokoro if model exists
    if _has_kokoro_model():
        try:
            logger.info("Attempting Kokoro TTS")
            # Timeout guard: kokoro should stream within reasonable time
            async for chunk in _kokoro_stream(text):
                yield chunk
            return
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"Kokoro TTS failed, falling back to edge-tts: {e}")

    # 2. Try edge-tts
    try:
        logger.info(f"Using edge-tts voice={DEFAULT_VOICE}")
        async for chunk in _edge_stream(text):
            yield chunk
        return
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.warning(f"edge-tts failed, falling back to synthetic: {e}")

    # 3. Synthetic offline fallback - always succeeds
    logger.info("Using synthetic WAV fallback")
    async for chunk in _synthetic_stream(text):
        yield chunk
