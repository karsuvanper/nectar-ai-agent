import os
import logging
from typing import List, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

logger = logging.getLogger(__name__)

# Force default QDRANT_HOST fallback to "memory" if host is "localhost" or empty inside docker/fallback logic
qdrant_host = os.getenv("QDRANT_HOST", "memory")
if qdrant_host.strip() in ["localhost", "127.0.0.1", ""]:
    qdrant_host = "memory"
QDRANT_HOST = qdrant_host.strip()
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333")) if os.getenv("QDRANT_PORT") else 6333


def _create_qdrant_client():
    """Create Qdrant client with graceful in-memory fallback.

    If QDRANT_HOST is not set, or is set to 'localhost', 'memory', or
    ':memory:', initialize directly as QdrantClient(":memory:").
    Catch connection errors gracefully so startup never crashes.
    """
    host_lower = (QDRANT_HOST or "").strip().lower()
    # In-memory cases: empty, localhost, 127.0.0.1, memory, :memory:
    if not host_lower or host_lower in ("localhost", "127.0.0.1", "memory", ":memory:"):
        try:
            logger.info("QDRANT_HOST='%s' -> using in-memory Qdrant (QdrantClient(':memory:'))", QDRANT_HOST)
            return QdrantClient(":memory:")
        except Exception as e:
            logger.warning(f"Failed to init in-memory Qdrant: {e}")
            raise

    # Try external Qdrant
    try:
        logger.info(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, check_compatibility=False)
    except Exception as e:
        logger.warning(f"Qdrant connection to {QDRANT_HOST}:{QDRANT_PORT} failed ({e}), falling back to in-memory")
        try:
            return QdrantClient(":memory:")
        except Exception as e2:
            logger.error(f"In-memory Qdrant fallback also failed: {e2}")
            raise


# Global client with graceful fallback - never crash on import
try:
    client = _create_qdrant_client()
except Exception as e:
    logger.warning(f"Qdrant initialization ultimately failed, trying in-memory one more time: {e}")
    try:
        client = QdrantClient(":memory:")
    except Exception as e2:
        logger.error(f"Failed to create any Qdrant client: {e2}")
        client = None  # will be handled gracefully in operations; startup won't crash

# Lazy embedding model - wrapped for lifespan warmup
_embedding_model = None


def get_embedding_model():
    """Lazy loader for TextEmbedding. Called inside lifespan to block until download finishes."""
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    try:
        from fastembed import TextEmbedding
        logger.info("Loading embedding model BAAI/bge-small-en-v1.5 ...")
        _embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        logger.info("Embedding model loaded")
    except Exception as e:
        logger.warning(f"Failed to load embedding model: {e}")
        raise
    return _embedding_model


def warmup_embedding():
    """Explicit warmup hook for lifespan - blocks until model download completes."""
    return get_embedding_model()


# Backward-compatible alias: access via property so old imports don't trigger eager load
# but still allow `from app.rag.vector_store import embedding_model` to work lazily.
class _LazyEmbeddingProxy:
    def __getattr__(self, name):
        return getattr(get_embedding_model(), name)

    def embed(self, *args, **kwargs):
        return get_embedding_model().embed(*args, **kwargs)


# Keep name for compatibility; actual model is loaded lazily
embedding_model = _LazyEmbeddingProxy()


def _ensure_collection():
    if client is None:
        logger.warning("Qdrant client is None, skipping _ensure_collection")
        return
    try:
        client.get_collection(collection_name="hvac_knowledge")
        return
    except Exception:
        pass
    try:
        client.create_collection(
            collection_name="hvac_knowledge",
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE,
            ),
        )
    except Exception as e:
        # Collection may already exist concurrently
        logger.warning(f"create_collection failed (may already exist): {e}")


def index_documents(documents: List[str], ids: List[int] = None) -> None:
    _ensure_collection()
    if client is None:
        logger.warning("Qdrant client is None, skipping index_documents")
        return
    if ids is None:
        ids = list(range(len(documents)))

    model = get_embedding_model()
    embeddings = list(model.embed(documents))

    points = []
    for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
        point = PointStruct(
            id=ids[i],
            vector=embedding.tolist(),
            payload={"text": doc},
        )
        points.append(point)

    client.upsert(collection_name="hvac_knowledge", points=points)


def search(query: str, top_k: int = 2) -> List[Tuple[str, float]]:
    _ensure_collection()
    if client is None:
        logger.warning("Qdrant client is None, returning empty search")
        return []
    model = get_embedding_model()
    query_embedding = list(model.embed([query]))[0]

    results = client.query_points(
        collection_name="hvac_knowledge",
        query=query_embedding.tolist(),
        limit=top_k,
    )

    results_texts = []
    for result in results.points:
        text = result.payload.get("text", "")
        score = result.score
        results_texts.append((text, score))

    return results_texts
