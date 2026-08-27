import os
from typing import List, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from fastembed import TextEmbedding


QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, check_compatibility=False)

embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")


def _ensure_collection():
    try:
        client.get_collection(collection_name="hvac_knowledge")
        return
    except Exception:
        pass
    client.create_collection(
        collection_name="hvac_knowledge",
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE,
        ),
    )


def index_documents(documents: List[str], ids: List[int] = None) -> None:
    _ensure_collection()
    if ids is None:
        ids = list(range(len(documents)))

    embeddings = list(embedding_model.embed(documents))

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
    query_embedding = list(embedding_model.embed([query]))[0]

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