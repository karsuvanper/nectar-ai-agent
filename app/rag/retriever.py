import os
import logging
from typing import List
import numpy as np

from app.rag.vector_store import search
from app.config import config

logger = logging.getLogger(__name__)

# Lazy CrossEncoder reranker - wrapped for lifespan warmup so Uvicorn blocks until download finishes
_reranker = None


def get_reranker():
    """Lazy loader for CrossEncoder. Blocks until model download completes when called from lifespan."""
    global _reranker
    if _reranker is not None:
        return _reranker
    try:
        from sentence_transformers import CrossEncoder
        logger.info("Loading reranker model cross-encoder/ms-marco-MiniLM-L-6-v2 ...")
        _reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        logger.info("Reranker model loaded")
    except Exception as e:
        logger.warning(f"Failed to load reranker model: {e}")
        raise
    return _reranker


def warmup_reranker():
    """Explicit warmup hook for FastAPI lifespan."""
    return get_reranker()


# Backward-compatible proxy so `from app.rag.retriever import reranker` still works but lazily
class _LazyRerankerProxy:
    def __getattr__(self, name):
        return getattr(get_reranker(), name)

    def predict(self, *args, **kwargs):
        return get_reranker().predict(*args, **kwargs)


reranker = _LazyRerankerProxy()


def query_rag_agent(user_query: str, top_k: int = 3) -> str:
    # Step A: Retrieve top_k=15 candidate chunks from Qdrant via vector similarity search
    candidates = search(user_query, top_k=15)

    if not candidates:
        return "Sufficient information was not found in the facility knowledge base to answer your request."

    # Step B: Construct query-document pairs for CrossEncoder
    # candidates is List[Tuple[str, float]] from vector similarity search
    # Extract just the text, ensuring they're plain strings
    chunk_texts = [str(chunk_text) for _, chunk_text in candidates]

    # Step C: Use CrossEncoder.predict() to compute relevance scores for query-document pairs
    # pairs = [[query, doc1], [query, doc2], ...]
    pair_list = [[user_query, doc_text] for doc_text in chunk_texts]

    # CrossEncoder predicts relevance scores for each pair
    model = get_reranker()
    rerank_scores = model.predict(pair_list)

    # rerank_scores is a numpy array of shape (n_candidates,)
    # Step D: Sort candidates by CrossEncoder score in descending order
    # Get indices that would sort the scores in descending order
    top_k_indices = np.argsort(rerank_scores)[::-1][:top_k]

    # Step E: Format the final context using top reranked candidates
    context_parts = []
    for idx in top_k_indices:
        if idx < len(candidates):
            context_parts.append(candidates[idx][0])  # Get the chunk_text (first element of tuple)

    context = "\n\n".join(context_parts)

    # Check guardrail: if top reranker score is very low, treat as no relevant info
    if len(rerank_scores) > 0 and float(rerank_scores[0]) < 0.3:
        return "Sufficient information was not found in the facility knowledge base to answer your request."

    # Prompt LLM to answer grounded ONLY on retrieved context
    prompt = f"""Answer the user's question strictly grounded ONLY on the following retrieved context from the facility knowledge base. Do not use any outside knowledge. If the answer cannot be derived from the context, state that sufficient information was not found.

Context:
{context}

User Question: {user_query}

Answer:"""

    from app.main import app
    import requests

    response = requests.post(
        url=f"{config.OPENROUTER_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": config.DEFAULT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        },
        timeout=30,
    )

    if response.status_code != 200:
        return "Sufficient information was not found in the facility knowledge base to answer your request."

    data = response.json()
    answer = data["choices"][0]["message"]["content"]
    return answer
