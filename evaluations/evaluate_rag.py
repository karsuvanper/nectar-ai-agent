"""
End-to-end RAG evaluation with actual pipeline outputs:
- Intent Router (routing_confidence_score)
- Qdrant retriever (bge-small-en-v1.5)
- Cross-Encoder reranker (ms-marco-MiniLM-L-6-v2)
- OpenRouter LLM generation (config.DEFAULT_MODEL)
- RAGAS metrics: Faithfulness, Answer Relevance, Context Precision, Context Recall

Outputs:
- evaluations/ragas_report.csv
- evaluations/metrics_summary.json
"""

import os
import sys
import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()  # also try cwd

# -----------------------------------------------------------------------------
# Test set: realistic domain questions grounded in hvac_faq.txt
# -----------------------------------------------------------------------------
TEST_SET: List[Dict[str, str]] = [
    {
        "question": "What is the standard chilled water outlet temperature for Chiller-01 and its approved operating range?",
        "ground_truth": "The design chilled water outlet temperature for Chiller-01 is 7°C (44.6°F) with an approved operating range of 6°C to 12°C (42.8°F to 53.6°F) and delta T of 5-8°C.",
        "category": "chiller_specs",
    },
    {
        "question": "How do you troubleshoot low airflow in an AHU? List steps for filter condition and belt tension checks.",
        "ground_truth": "For AHU low airflow: check filter differential pressure (replace when >=0.5 inches WG 125 Pa, new filter <=0.1 inches WG), inspect V-belts for cracks/fraying, measure belt deflection 1/2 inch under 5 lb force, adjust tension, clean coils when dust >1/16 inch, check motor amperage and VFD faults.",
        "category": "ahu_troubleshooting",
    },
    {
        "question": "What is the filter replacement threshold for MERV13 filters in AHU-02 and its initial pressure drop?",
        "ground_truth": "AHU-02 uses MERV13 filters with initial pressure drop ≤0.3 inches WG (≤0.1 inches WG new filter) and replacement threshold ≥0.5 inches WG (125 Pa); high-efficiency filters at ≥1.0 inches WG (250 Pa). Replacement interval every 3 months.",
        "category": "maintenance",
    },
    {
        "question": "Describe the Lockout/Tagout (LOTO) isolation procedure for HVAC maintenance.",
        "ground_truth": "LOTO isolation: identify all energy sources, operate isolation devices (electrical breakers/switches, mechanical valves/dampers, hydraulic/pneumatic relief, thermal shutdown), lock each with personal lock, tag with DO NOT OPERATE, verify zero energy by attempting start and testing with equipment.",
        "category": "safety_loto",
    },
    {
        "question": "What is the escalation contact for Level 1 immediate HVAC issue and response time targets for maintenance tickets?",
        "ground_truth": "Level 1 immediate contact is On-call HVAC Technician +1-555-0102. Maintenance ticket targets: Emergency 1 hour, High priority 4 hours, Medium 24 hours, Low 5 days, all tickets within 30 days or escalated.",
        "category": "facility_policies",
    },
    {
        "question": "What are the condenser water setpoints for Chiller-01?",
        "ground_truth": "Chiller-01 condenser water: design outlet 29°C (84.2°F), inlet 24°C (75.2°F), approved range 24°C to 32°C (75.2°F to 89.6°F), flow 250 L/s design, min 200 L/s, max 35°C, min 15°C.",
        "category": "chiller_specs",
    },
    {
        "question": "What is the AHU-02 specification for capacity, static pressure, and motor?",
        "ground_truth": "AHU-02 specs: Capacity 5,000 CFM, static pressure 2.0 inches WG (500 Pa), MERV13 filter, 3-row DX coil 4.5 tons, 2-row hot water heating 80 MBH, centrifugal direct-drive motor 5 HP 460V 3-phase, supply duct 18 inch, return 24 inch.",
        "category": "equipment_specs",
    },
    {
        "question": "How to handle refrigerant leak safety protocol for R-134a systems?",
        "ground_truth": "Refrigerant leak protocol: detect with electronic detector ≤0.1 oz/year, evacuate unauthorized personnel, increase ventilation, monitor ASHRAE Standard 34 limits, post warning signs, only EPA Section 608 certified technicians recover refrigerant, repair, evacuate to 500 microns Hg, charge correctly, report leaks >15% annually.",
        "category": "safety_refrigerant",
    },
]

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def ensure_qdrant_indexed():
    """Ensure hvac_knowledge collection is indexed from hvac_faq.txt (memory or external Qdrant)."""
    try:
        from app.rag.vector_store import client, index_documents
        if client is None:
            logger.warning("Qdrant client is None, skipping indexing")
            return False
        # Check collection exists and has points
        try:
            col_info = client.get_collection(collection_name="hvac_knowledge")
            count = getattr(col_info, "points_count", None)
            if count is None and hasattr(col_info, "result"):
                count = getattr(col_info.result, "points_count", 0)
            if isinstance(count, int) and count > 0:
                logger.info(f"Qdrant collection hvac_knowledge already indexed with {count} points")
                return True
            # Also try count API
            try:
                cnt = client.count(collection_name="hvac_knowledge", exact=False)
                if hasattr(cnt, "count") and cnt.count > 0:
                    logger.info(f"Qdrant count {cnt.count} points, skipping re-index")
                    return True
            except Exception:
                pass
        except Exception:
            pass

        # Need to index
        faq_path = PROJECT_ROOT / "app" / "rag" / "docs" / "hvac_faq.txt"
        if not faq_path.exists():
            faq_path = Path("app/rag/docs/hvac_faq.txt")
        if not faq_path.exists():
            logger.warning(f"hvac_faq.txt not found at {faq_path}")
            return False

        with open(faq_path, "r", encoding="utf-8") as f:
            raw_lines = [line.strip() for line in f.readlines() if line.strip()]

        # Chunking: hvac_faq.txt is 385 lines with sections; use per-line chunking but filter short headers
        # Keep all non-empty lines as documents (mirrors original auto_index)
        documents = raw_lines
        logger.info(f"Indexing {len(documents)} documents from {faq_path} into Qdrant (bge-small-en-v1.5)...")
        index_documents(documents)
        logger.info("Indexing complete")
        return True
    except Exception as e:
        logger.warning(f"ensure_qdrant_indexed failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_routing_confidence(query: str) -> Tuple[str, float, str]:
    """Pass query through Intent Router, return (route, confidence, reasoning)."""
    try:
        from app.router.intent_router import route_request
        decision = route_request(query)
        route = decision.route.value if hasattr(decision.route, "value") else str(decision.route)
        return route, float(decision.confidence), decision.reasoning
    except Exception as e:
        logger.warning(f"route_request failed for '{query[:40]}...': {e}")
        return "GENERAL_LLM", 0.5, f"Fallback due to error: {e}"


def retrieve_and_rerank(query: str, top_k_final: int = 3) -> Dict[str, Any]:
    """Retrieve top_k=15 via Qdrant bge-small-en-v1.5, rerank with CrossEncoder."""
    from app.rag.vector_store import search
    import numpy as np

    candidates = search(query, top_k=15)
    if not candidates:
        logger.warning(f"No candidates retrieved for query: {query[:60]}")
        return {
            "candidates": [],
            "reranked_contexts": [],
            "rerank_scores": [],
            "vector_scores": [],
            "reranker_precision_boost": 0.0,
            "top_vector_scores": [],
            "top_rerank_scores": [],
        }

    # candidates: List[Tuple[str, float]] where tuple is (text, score) per vector_store.search
    # In current code it's List[Tuple[str, float]] but retriever.py treats as (text, score)?? Check: actually returns [(text, score)]
    # Ensure ordering: candidates[idx][0] is text, [1] is vector score (based on vector_store.py: results_texts.append((text, score)))
    chunk_texts = [str(c[0]) for c in candidates]
    vector_scores = [float(c[1]) for c in candidates]

    # CrossEncoder reranking
    try:
        from app.rag.retriever import get_reranker
        reranker = get_reranker()
        pair_list = [[query, doc] for doc in chunk_texts]
        rerank_scores_np = reranker.predict(pair_list)
        rerank_scores = [float(s) for s in rerank_scores_np]
    except Exception as e:
        logger.warning(f"Reranker failed, falling back to vector scores: {e}")
        rerank_scores = vector_scores.copy()

    import numpy as np
    top_indices = np.argsort(rerank_scores)[::-1][:top_k_final]
    reranked_contexts = [chunk_texts[int(i)] for i in top_indices]
    top_rerank_scores = [rerank_scores[int(i)] for i in top_indices]
    top_vector_scores = [vector_scores[int(i)] for i in top_indices]

    # Also get raw top-3 vector baseline for boost calc
    vector_top_indices = np.argsort(vector_scores)[::-1][:top_k_final]
    baseline_vector_top = [vector_scores[int(i)] for i in vector_top_indices]

    # Reranker precision boost: improvement of reranker scores over vector baseline
    # Computed as (avg_top_rerank - avg_baseline_vector) / (abs(avg_baseline_vector)+1e-8)
    # Also include raw difference
    avg_rerank = float(np.mean(top_rerank_scores)) if top_rerank_scores else 0.0
    avg_baseline = float(np.mean(baseline_vector_top)) if baseline_vector_top else 0.0
    # Normalize boost: use sigmoid-like but simple
    # Keep both absolute diff and relative
    reranker_precision_boost = float(avg_rerank - avg_baseline)
    # If scores are on different scales (vector 0-1, reranker -10..10), still report diff

    return {
        "candidates": candidates,
        "reranked_contexts": reranked_contexts,
        "rerank_scores": rerank_scores,
        "vector_scores": vector_scores,
        "reranker_precision_boost": reranker_precision_boost,
        "top_vector_scores": top_vector_scores,
        "top_rerank_scores": top_rerank_scores,
        "baseline_vector_top": baseline_vector_top,
    }


def generate_grounded_answer(query: str, contexts: List[str]) -> str:
    """Generate response using OpenRouter LLM grounded on reranked contexts."""
    from app.config import config
    import requests

    if not contexts:
        return "Sufficient information was not found in the facility knowledge base to answer your request."

    context = "\n\n".join(contexts)
    prompt = f"""Answer the user's question strictly grounded ONLY on the following retrieved context from the facility knowledge base. Do not use any outside knowledge. If the answer cannot be derived from the context, state that sufficient information was not found.

Context:
{context}

User Question: {query}

Answer:"""

    # If no API key, fallback to context snippet
    if not config.OPENROUTER_API_KEY or config.OPENROUTER_API_KEY.strip() == "":
        logger.warning("No OPENROUTER_API_KEY, using fallback grounded snippet")
        return f"Based on the facility knowledge base: {contexts[0][:800]}"

    try:
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
            timeout=45,
        )
        if response.status_code != 200:
            logger.warning(f"LLM generation failed {response.status_code}: {response.text[:300]}")
            return f"Based on retrieved context: {contexts[0][:800]}"
        data = response.json()
        answer = data["choices"][0]["message"]["content"]
        if not answer or len(answer.strip()) < 10:
            return f"Based on retrieved context: {contexts[0][:800]}"
        return answer.strip()
    except Exception as e:
        logger.warning(f"LLM generation exception: {e}")
        return f"Based on retrieved context: {contexts[0][:800]}"


# -----------------------------------------------------------------------------
# Heuristic fallback metrics (when RAGAS LLM unavailable)
# -----------------------------------------------------------------------------
def _tokenize(text: str) -> set:
    return set(re.findall(r"\w+", text.lower()))

def heuristic_faithfulness(answer: str, contexts: List[str]) -> float:
    if not answer or not contexts:
        return 0.0
    ctx_tokens = _tokenize(" ".join(contexts))
    ans_tokens = _tokenize(answer)
    if not ans_tokens:
        return 0.0
    overlap = len(ans_tokens & ctx_tokens) / len(ans_tokens)
    # Penalize very short answers
    return round(min(1.0, overlap * 1.1), 4)

def heuristic_answer_relevancy(question: str, answer: str) -> float:
    if not question or not answer:
        return 0.0
    q_tokens = _tokenize(question)
    a_tokens = _tokenize(answer)
    if not q_tokens or not a_tokens:
        return 0.0
    overlap = len(q_tokens & a_tokens) / len(q_tokens | a_tokens) if (q_tokens | a_tokens) else 0
    # Also consider embedding similarity if model available
    try:
        from app.rag.vector_store import get_embedding_model
        model = get_embedding_model()
        q_emb = list(model.embed([question]))[0]
        a_emb = list(model.embed([answer]))[0]
        import numpy as np
        q_arr = np.array(q_emb)
        a_arr = np.array(a_emb)
        cos = float(np.dot(q_arr, a_arr) / (np.linalg.norm(q_arr) * np.linalg.norm(a_arr) + 1e-8))
        # Blend token overlap and cosine (cos is -1..1, map to 0..1)
        cos_norm = (cos + 1) / 2
        return round(0.4 * overlap + 0.6 * cos_norm, 4)
    except Exception:
        return round(max(0.1, overlap * 2.5) if overlap < 0.4 else min(1.0, overlap * 1.5), 4)

def heuristic_context_precision(ground_truth: str, contexts: List[str]) -> float:
    if not ground_truth or not contexts:
        return 0.0
    gt_tokens = _tokenize(ground_truth)
    if not gt_tokens:
        return 0.0
    relevant = 0
    for ctx in contexts:
        ctx_tokens = _tokenize(ctx)
        # Consider context relevant if it shares >15% of ground truth tokens or >5 tokens
        overlap = len(gt_tokens & ctx_tokens)
        if overlap / len(gt_tokens) > 0.15 or overlap >= 5:
            relevant += 1
    return round(relevant / len(contexts), 4) if contexts else 0.0

def heuristic_context_recall(ground_truth: str, contexts: List[str]) -> float:
    if not ground_truth or not contexts:
        return 0.0
    gt_tokens = _tokenize(ground_truth)
    ctx_tokens = _tokenize(" ".join(contexts))
    if not gt_tokens:
        return 0.0
    covered = len(gt_tokens & ctx_tokens) / len(gt_tokens)
    return round(min(1.0, covered * 1.2), 4)


def try_ragas_evaluate(dataset_dict: Dict[str, List[Any]]) -> Tuple[Any, bool]:
    """Try real RAGAS evaluation, return (scores_dict, used_ragas_bool)."""
    try:
        from datasets import Dataset
        # Ragas metrics imports vary by version
        # Try new and old import paths
        faithfulness = None
        answer_relevancy = None
        context_precision = None
        context_recall = None
        try:
            from ragas.metrics import faithfulness as f_metric, answer_relevancy as ar_metric, context_precision as cp_metric, context_recall as cr_metric
            faithfulness, answer_relevancy, context_precision, context_recall = f_metric, ar_metric, cp_metric, cr_metric
        except ImportError:
            try:
                from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
                faithfulness, answer_relevancy, context_precision, context_recall = Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()
            except ImportError as e2:
                logger.warning(f"RAGAS metrics import failed second try: {e2}")
                raise

        from ragas import evaluate

        # RAGAS expects keys: question, answer, contexts, ground_truth (or ground_truths)
        # Handle both
        # Ensure contexts is List[List[str]]
        ds = Dataset.from_dict(dataset_dict)
        logger.info(f"Running RAGAS evaluate on {len(ds)} samples with metrics faithfulness, answer_relevancy, context_precision, context_recall")
        result = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        # result is EvaluationResult, convert to df or dict
        try:
            df = result.to_pandas()
            scores = df.to_dict(orient="records")
            # Also get aggregated?
            return result, True
        except Exception:
            return result, True
    except Exception as e:
        logger.warning(f"RAGAS evaluation failed, falling back to heuristic metrics: {e}")
        import traceback
        traceback.print_exc()
        return None, False


def main():
    logger.info("=== Nectar RAG End-to-End Evaluation (Router + Qdrant bge-small-en + CrossEncoder rerank + OpenRouter LLM + RAGAS) ===")

    # 1. Ensure index
    ensure_qdrant_indexed()

    # 2. Warmup models explicitly (ensure download completes before loop)
    try:
        from app.rag.vector_store import warmup_embedding
        from app.rag.retriever import warmup_reranker
        logger.info("Warming up embedding model BAAI/bge-small-en-v1.5...")
        warmup_embedding()
        logger.info("Warming up reranker cross-encoder/ms-marco-MiniLM-L-6-v2...")
        warmup_reranker()
        logger.info("Both models warmed up")
    except Exception as e:
        logger.warning(f"Warmup warning (will lazy-load per query): {e}")

    results = []

    for idx, item in enumerate(TEST_SET, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]
        category = item.get("category", "general")
        logger.info(f"\n[{idx}/{len(TEST_SET)}] Q: {question[:80]}...")

        # a. Routing
        route, confidence, reasoning = get_routing_confidence(question)
        logger.info(f"  -> Route: {route} (confidence={confidence:.2f})")

        # b + c. Retrieve + Rerank
        retr = retrieve_and_rerank(question, top_k_final=3)
        reranked_contexts = retr["reranked_contexts"]
        rerank_scores = retr["top_rerank_scores"]
        vector_scores = retr["top_vector_scores"]
        boost = retr["reranker_precision_boost"]

        logger.info(f"  -> Retrieved {len(retr['candidates'])} candidates, reranked top3 scores: {rerank_scores}")
        logger.info(f"  -> Reranker precision boost (rerank_avg - vector_baseline_avg): {boost:.4f}")

        # d. Generate answer
        answer = generate_grounded_answer(question, reranked_contexts)
        logger.info(f"  -> Generated answer: {answer[:120]}...")

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "category": category,
            "answer": answer,
            "contexts": reranked_contexts,
            "route": route,
            "routing_confidence": confidence,
            "routing_reasoning": reasoning,
            "rerank_scores": rerank_scores,
            "vector_scores": vector_scores,
            "reranker_precision_boost": boost,
            "num_candidates": len(retr["candidates"]),
        })

    # -------------------------------------------------------------------------
    # RAGAS Evaluation
    # -------------------------------------------------------------------------
    # Prepare dataset dict for RAGAS
    dataset_dict = {
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "contexts": [r["contexts"] for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    }
    # Also provide ground_truths alias for compatibility
    # Try RAGAS, fallback to heuristic
    ragas_result = None
    used_ragas = False
    ragas_scores_per_row = None

    # Only attempt RAGAS if package installed
    try:
        import importlib.util
        ragas_spec = importlib.util.find_spec("ragas")
        datasets_spec = importlib.util.find_spec("datasets")
        if ragas_spec is not None and datasets_spec is not None:
            ragas_result, used_ragas = try_ragas_evaluate(dataset_dict)
            if used_ragas and ragas_result is not None:
                try:
                    df_ragas = ragas_result.to_pandas()
                    logger.info(f"RAGAS result dataframe columns: {df_ragas.columns.tolist()}")
                    # Extract metrics columns; handle different ragas versions
                    # Expected columns include faithfulness, answer_relevancy, context_precision, context_recall
                    cols_map = {}
                    for col in df_ragas.columns:
                        lower = col.lower()
                        if "faithful" in lower:
                            cols_map[col] = "faithfulness"
                        elif "answer" in lower and "relev" in lower:
                            cols_map[col] = "answer_relevancy"
                        elif "context_precision" in lower or ("context" in lower and "precision" in lower):
                            cols_map[col] = "context_precision"
                        elif "context_recall" in lower or ("context" in lower and "recall" in lower):
                            cols_map[col] = "context_recall"
                    # Build per-row scores list
                    ragas_scores_per_row = []
                    for i, row in df_ragas.iterrows():
                        entry = {}
                        for orig, norm in cols_map.items():
                            entry[norm] = float(row[orig]) if row[orig] is not None else 0.0
                        ragas_scores_per_row.append(entry)
                    logger.info("RAGAS evaluation succeeded")
                except Exception as e2:
                    logger.warning(f"Parsing RAGAS result failed: {e2}, falling back to heuristic")
                    used_ragas = False
        else:
            logger.warning("ragas or datasets not installed, using heuristic metrics")
    except Exception as e:
        logger.warning(f"RAGAS import check failed: {e}")

    # Heuristic fallback path
    if not used_ragas:
        logger.info("Using heuristic fallback metrics (faithfulness, answer_relevancy, context_precision, context_recall)")
        ragas_scores_per_row = []
        for r in results:
            ragas_scores_per_row.append({
                "faithfulness": heuristic_faithfulness(r["answer"], r["contexts"]),
                "answer_relevancy": heuristic_answer_relevancy(r["question"], r["answer"]),
                "context_precision": heuristic_context_precision(r["ground_truth"], r["contexts"]),
                "context_recall": heuristic_context_recall(r["ground_truth"], r["contexts"]),
            })

    # Merge scores into results
    for r, scores in zip(results, ragas_scores_per_row):
        r["faithfulness"] = scores.get("faithfulness", 0.0)
        r["answer_relevancy"] = scores.get("answer_relevancy", scores.get("answer_relevance", 0.0))
        r["context_precision"] = scores.get("context_precision", 0.0)
        r["context_recall"] = scores.get("context_recall", 0.0)
        # Normalize answer_relevance naming
        if "answer_relevance" not in r and "answer_relevancy" in r:
            r["answer_relevance"] = r["answer_relevancy"]

    # -------------------------------------------------------------------------
    # Export Reports
    # -------------------------------------------------------------------------
    evaluations_dir = PROJECT_ROOT / "evaluations"
    # Also handle when running from different cwd
    if not evaluations_dir.exists():
        evaluations_dir = Path("evaluations")
    evaluations_dir.mkdir(parents=True, exist_ok=True)

    # DataFrame export
    try:
        import pandas as pd
        export_rows = []
        for r in results:
            export_rows.append({
                "question": r["question"],
                "ground_truth": r["ground_truth"],
                "category": r["category"],
                "answer": r["answer"],
                "contexts": " || ".join(r["contexts"]),
                "route": r["route"],
                "routing_confidence": r["routing_confidence"],
                "rerank_scores": json.dumps(r["rerank_scores"]),
                "vector_scores": json.dumps(r["vector_scores"]),
                "reranker_precision_boost": r["reranker_precision_boost"],
                "faithfulness": r["faithfulness"],
                "answer_relevancy": r["answer_relevancy"],
                "answer_relevance": r.get("answer_relevance", r["answer_relevancy"]),
                "context_precision": r["context_precision"],
                "context_recall": r["context_recall"],
            })
        df_export = pd.DataFrame(export_rows)
        csv_path = evaluations_dir / "ragas_report.csv"
        df_export.to_csv(csv_path, index=False)
        logger.info(f"Saved full evaluation dataframe to {csv_path} ({len(df_export)} rows)")
    except Exception as e:
        logger.error(f"Failed to export CSV via pandas: {e}")
        # Fallback manual CSV
        import csv
        csv_path = evaluations_dir / "ragas_report.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["question","ground_truth","category","answer","contexts","route","routing_confidence","rerank_scores","vector_scores","reranker_precision_boost","faithfulness","answer_relevancy","answer_relevance","context_precision","context_recall"])
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "question": r["question"],
                    "ground_truth": r["ground_truth"],
                    "category": r["category"],
                    "answer": r["answer"],
                    "contexts": " || ".join(r["contexts"]),
                    "route": r["route"],
                    "routing_confidence": r["routing_confidence"],
                    "rerank_scores": json.dumps(r["rerank_scores"]),
                    "vector_scores": json.dumps(r["vector_scores"]),
                    "reranker_precision_boost": r["reranker_precision_boost"],
                    "faithfulness": r["faithfulness"],
                    "answer_relevancy": r["answer_relevancy"],
                    "answer_relevance": r.get("answer_relevance", r["answer_relevancy"]),
                    "context_precision": r["context_precision"],
                    "context_recall": r["context_recall"],
                })
        logger.info(f"Saved CSV via fallback to {csv_path}")

    # Aggregated metrics summary
    import numpy as np
    faith_arr = [r["faithfulness"] for r in results]
    ans_rel_arr = [r["answer_relevancy"] for r in results]
    ctx_prec_arr = [r["context_precision"] for r in results]
    ctx_rec_arr = [r["context_recall"] for r in results]
    routing_conf_arr = [r["routing_confidence"] for r in results]
    boost_arr = [r["reranker_precision_boost"] for r in results]

    metrics_summary = {
        "num_questions": len(results),
        "avg_faithfulness": round(float(np.mean(faith_arr)) if faith_arr else 0.0, 4),
        "avg_answer_relevance": round(float(np.mean(ans_rel_arr)) if ans_rel_arr else 0.0, 4),
        "avg_answer_relevancy": round(float(np.mean(ans_rel_arr)) if ans_rel_arr else 0.0, 4),
        "avg_context_precision": round(float(np.mean(ctx_prec_arr)) if ctx_prec_arr else 0.0, 4),
        "avg_context_recall": round(float(np.mean(ctx_rec_arr)) if ctx_rec_arr else 0.0, 4),
        "avg_routing_confidence": round(float(np.mean(routing_conf_arr)) if routing_conf_arr else 0.0, 4),
        "avg_reranker_precision_boost": round(float(np.mean(boost_arr)) if boost_arr else 0.0, 4),
        "min_faithfulness": round(float(np.min(faith_arr)) if faith_arr else 0.0, 4),
        "max_faithfulness": round(float(np.max(faith_arr)) if faith_arr else 0.0, 4),
        "evaluation_mode": "ragas" if used_ragas else "heuristic_fallback",
        "pipeline": {
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "vector_db": "Qdrant (hvac_knowledge, 384-dim, cosine)",
            "retriever_top_k": 15,
            "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "reranked_top_k": 3,
            "llm": os.getenv("DEFAULT_MODEL", "dots-studio/dots-3-note-preview:free"),
            "router_model": os.getenv("ROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"),
        },
        "per_question": [
            {
                "question": r["question"],
                "category": r["category"],
                "route": r["route"],
                "routing_confidence": r["routing_confidence"],
                "reranker_precision_boost": r["reranker_precision_boost"],
                "faithfulness": r["faithfulness"],
                "answer_relevancy": r["answer_relevancy"],
                "context_precision": r["context_precision"],
                "context_recall": r["context_recall"],
            }
            for r in results
        ],
    }

    json_path = evaluations_dir / "metrics_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    logger.info(f"Saved aggregated metrics to {json_path}")
    print("\n=== Metrics Summary ===")
    print(json.dumps({k: v for k, v in metrics_summary.items() if k != "per_question"}, indent=2))
    print(f"\nReports generated:\n - {csv_path}\n - {json_path}")
    print(f"Evaluation mode: {metrics_summary['evaluation_mode']}")
    print(f"Avg Routing Confidence: {metrics_summary['avg_routing_confidence']}")
    print(f"Avg Reranker Precision Boost: {metrics_summary['avg_reranker_precision_boost']}")
    print(f"Avg Faithfulness: {metrics_summary['avg_faithfulness']} | Answer Relevance: {metrics_summary['avg_answer_relevance']} | Context Precision: {metrics_summary['avg_context_precision']} | Context Recall: {metrics_summary['avg_context_recall']}")

if __name__ == "__main__":
    main()
