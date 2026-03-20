"""ICT RAG pipeline utilities (Pinecone REST + query transforms + reranking)."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Dict, List, Tuple

import requests


def resolve_index(index_name: str, pinecone_api_key: str) -> Tuple[str, int | None, str]:
    """Return (host, dimension, error)."""
    if not pinecone_api_key:
        return "", None, "PINECONE_API_KEY is not configured."

    try:
        response = requests.get(
            f"https://api.pinecone.io/indexes/{index_name}",
            headers={"Api-Key": pinecone_api_key, "Accept": "application/json"},
            timeout=15,
        )
        if response.status_code == 404:
            return "", None, f"Pinecone index '{index_name}' does not exist."
        response.raise_for_status()
        payload = response.json()
        host = payload.get("host") or payload.get("status", {}).get("host") or ""
        dim = payload.get("dimension")
        if dim is None:
            dim = payload.get("spec", {}).get("dimension")
        if not host:
            return "", None, "Pinecone index host could not be resolved."
        return host, dim, ""
    except requests.RequestException as exc:
        return "", None, f"Index lookup failed: {exc}"


def build_sparse_vector(text: str, modulus: int = 100_000) -> Dict[str, List[float]]:
    """Create a lightweight sparse representation for Pinecone sparse search."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    counts: Dict[int, float] = {}
    for token in tokens:
        idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % modulus
        counts[idx] = counts.get(idx, 0.0) + 1.0
    if not counts:
        return {"indices": [], "values": []}
    indices = sorted(counts)
    values = [counts[i] for i in indices]
    return {"indices": indices, "values": values}


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def heuristic_overlap_score(query: str, text: str) -> float:
    q = set(re.findall(r"[a-z0-9]+", query.lower()))
    t = set(re.findall(r"[a-z0-9]+", text.lower()))
    if not q or not t:
        return 0.0
    return len(q & t) / max(len(q), 1)


def transform_query(client: Any, query: str, strategy: str, model: str) -> List[str]:
    """Return list of queries for retrieval."""
    if strategy == "HyDE":
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Generate a concise hypothetical answer for retrieval grounding."},
                    {"role": "user", "content": query},
                ],
                temperature=0.2,
                max_tokens=180,
            )
            hypo = resp.choices[0].message.content or ""
            return [query, hypo] if hypo else [query]
        except Exception:
            return [query]

    if strategy in {"Query Expansion", "Query Decomposition"}:
        instruction = (
            "Produce 3 search queries as a JSON array."
            if strategy == "Query Expansion"
            else "Break the question into up to 3 sub-questions as a JSON array."
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": query},
                ],
                temperature=0.1,
                max_tokens=200,
            )
            text = resp.choices[0].message.content or "[]"
            candidates = re.findall(r'"([^"]+)"', text)
            outputs = [query] + [c.strip() for c in candidates if c.strip()]
            return list(dict.fromkeys(outputs))[:4]
        except Exception:
            return [query]

    return [query]


def run_pinecone_query(
    host: str,
    api_key: str,
    top_k: int,
    dense_vector: List[float] | None,
    sparse_vector: Dict[str, List[float]] | None,
) -> Tuple[List[Dict[str, Any]], str]:
    body: Dict[str, Any] = {"topK": top_k, "includeMetadata": True}
    if dense_vector is not None:
        body["vector"] = dense_vector
    if sparse_vector is not None:
        body["sparseVector"] = sparse_vector

    try:
        response = requests.post(
            f"https://{host}/query",
            headers={"Api-Key": api_key, "Content-Type": "application/json", "Accept": "application/json"},
            json=body,
            timeout=20,
        )
        if response.status_code >= 400:
            return [], f"Pinecone query failed HTTP {response.status_code}: {response.text[:800]}"
        matches = response.json().get("matches", [])
        return matches, ""
    except requests.RequestException as exc:
        return [], f"Pinecone query request failed: {exc}"


def rerank_documents(
    client: Any,
    query: str,
    docs: List[Dict[str, Any]],
    reranker: str,
    cohere_api_key: str,
    embedding_model: str,
) -> List[Dict[str, Any]]:
    if not docs:
        return docs

    if reranker == "Cohere Rerank" and cohere_api_key:
        try:
            response = requests.post(
                "https://api.cohere.ai/v1/rerank",
                headers={"Authorization": f"Bearer {cohere_api_key}", "Content-Type": "application/json"},
                json={
                    "model": "rerank-v3.5",
                    "query": query,
                    "documents": [d["text"] for d in docs],
                    "top_n": len(docs),
                },
                timeout=20,
            )
            response.raise_for_status()
            ranked = response.json().get("results", [])
            out = []
            for item in ranked:
                idx = item.get("index", 0)
                if 0 <= idx < len(docs):
                    enriched = dict(docs[idx])
                    enriched["rerank_score"] = item.get("relevance_score", 0)
                    out.append(enriched)
            return out if out else docs
        except Exception:
            pass

    if reranker == "BGE reranker":
        try:
            qv = client.embeddings.create(model=embedding_model, input=query).data[0].embedding
            dv = client.embeddings.create(model=embedding_model, input=[d["text"] for d in docs]).data
            scored = []
            for d, emb in zip(docs, dv):
                score = cosine_similarity(qv, emb.embedding)
                entry = dict(d)
                entry["rerank_score"] = score
                scored.append(entry)
            scored.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            return scored
        except Exception:
            pass

    # ColBERT-style lightweight lexical interaction fallback
    def colbert_like_score(doc_text: str) -> float:
        q_terms = re.findall(r"[a-z0-9]+", query.lower())
        d_terms = re.findall(r"[a-z0-9]+", doc_text.lower())
        if not q_terms or not d_terms:
            return 0.0
        d_set = set(d_terms)
        return sum(1.0 for t in q_terms if t in d_set) / len(q_terms)

    scored = []
    for d in docs:
        entry = dict(d)
        if reranker == "ColBERT":
            entry["rerank_score"] = colbert_like_score(d["text"])
        else:
            entry["rerank_score"] = heuristic_overlap_score(query, d["text"])
        scored.append(entry)
    scored.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
    return scored
