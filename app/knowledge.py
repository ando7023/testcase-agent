import hashlib
import json
import math
import re
import sqlite3
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .llm import LLMError, OpenAICompatibleClient
from .models import KnowledgeDocument


TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_]+")
RRF_K = 60


def tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    for token in TOKEN_PATTERN.findall(text):
        token = token.lower()
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            tokens.extend(list(token))
            tokens.extend(token[index : index + 2] for index in range(len(token) - 1))
        else:
            tokens.append(token)
    return tokens


def normalize_vector(vector: List[float]) -> List[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if not norm:
        return vector
    return [value / norm for value in vector]


class EmbeddingProvider:
    name = "embedding"

    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


class HashingEmbeddingProvider(EmbeddingProvider):
    """Stable local fallback used when a remote embedding service is unavailable."""

    name = "hashing-v1"

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> List[float]:
        vector = [0.0] * self.dimensions
        counts = Counter(tokenize(text))
        for token, count in counts.items():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign * (1.0 + math.log(float(count)))
        return normalize_vector(vector)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, client: OpenAICompatibleClient) -> None:
        self.client = client
        self.name = "openai-compatible:" + client.embedding_model

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [normalize_vector(vector) for vector in self.client.create_embeddings(texts)]


class SQLiteVectorIndex:
    """Small persistent vector cache. JSON knowledge remains the source of truth."""

    def __init__(self, path: Optional[Path]) -> None:
        self.path = path
        self._lock = threading.RLock()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS knowledge_vectors (
                        document_id TEXT NOT NULL,
                        fingerprint TEXT NOT NULL,
                        provider TEXT NOT NULL,
                        vector_json TEXT NOT NULL,
                        updated_at REAL NOT NULL,
                        PRIMARY KEY (document_id, provider)
                    )
                    """
                )

    def _connect(self) -> sqlite3.Connection:
        if self.path is None:
            raise RuntimeError("Vector index path is not configured")
        return sqlite3.connect(str(self.path), timeout=10)

    def get(
        self, document_id: str, fingerprint: str, provider: str
    ) -> Optional[List[float]]:
        if self.path is None:
            return None
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT vector_json FROM knowledge_vectors
                WHERE document_id = ? AND fingerprint = ? AND provider = ?
                """,
                (document_id, fingerprint, provider),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put(
        self,
        document_id: str,
        fingerprint: str,
        provider: str,
        vector: List[float],
    ) -> None:
        if self.path is None:
            return
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO knowledge_vectors
                (document_id, fingerprint, provider, vector_json, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    fingerprint,
                    provider,
                    json.dumps(vector, separators=(",", ":")),
                    time.time(),
                ),
            )


@dataclass
class SearchResult:
    document: KnowledgeDocument
    score: float
    lexical_score: float = 0.0
    vector_score: float = 0.0
    fusion_score: float = 0.0
    rerank_score: float = 0.0
    matched_terms: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)


class KnowledgeBase:
    """Hybrid BM25 + vector retrieval with RRF fusion and explainable reranking."""

    def __init__(
        self,
        documents: List[KnowledgeDocument],
        embedding_provider: Optional[EmbeddingProvider] = None,
        index_path: Optional[Path] = None,
    ) -> None:
        self.documents = documents
        self.embedding_provider = embedding_provider or HashingEmbeddingProvider()
        self.vector_index = SQLiteVectorIndex(index_path)
        self.last_trace: Dict[str, Any] = {}

    def search(
        self, query: str, limit: int = 4
    ) -> List[Tuple[KnowledgeDocument, float]]:
        detailed = self.search_detailed(query, limit)
        return [(item.document, item.score) for item in detailed]

    def search_detailed(self, query: str, limit: int = 12) -> List[SearchResult]:
        started = time.time()
        if not self.documents or not tokenize(query):
            self.last_trace = {
                "mode": "hybrid",
                "candidate_count": 0,
                "latency_ms": int((time.time() - started) * 1000),
            }
            return []

        candidate_limit = max(limit * 3, 24)
        lexical = self._bm25(query)[:candidate_limit]
        provider = self.embedding_provider
        fallback_reason = ""
        try:
            vector = self._vector_search(query, provider)[:candidate_limit]
        except (LLMError, ValueError, OSError, RuntimeError) as exc:
            fallback_reason = str(exc)
            provider = HashingEmbeddingProvider()
            vector = self._vector_search(query, provider)[:candidate_limit]

        lexical_scores = {document.id: score for document, score in lexical}
        vector_scores = {document.id: score for document, score in vector}
        lexical_ranks = {document.id: rank for rank, (document, _) in enumerate(lexical, 1)}
        vector_ranks = {document.id: rank for rank, (document, _) in enumerate(vector, 1)}
        documents = {document.id: document for document in self.documents}
        candidate_ids = set(lexical_scores) | set(vector_scores)

        lexical_normalized = self._normalize_scores(lexical_scores)
        vector_normalized = self._normalize_scores(vector_scores)
        fused: List[SearchResult] = []
        query_tokens = set(tokenize(query))
        intent_types = self._intent_doc_types(query)
        for document_id in candidate_ids:
            lexical_rank = lexical_ranks.get(document_id)
            vector_rank = vector_ranks.get(document_id)
            rrf = 0.0
            if lexical_rank:
                rrf += 1.0 / (RRF_K + lexical_rank)
            if vector_rank:
                rrf += 1.0 / (RRF_K + vector_rank)
            lexical_score = lexical_scores.get(document_id, 0.0)
            vector_score = vector_scores.get(document_id, 0.0)
            fusion_score = (
                0.55 * lexical_normalized.get(document_id, 0.0)
                + 0.35 * vector_normalized.get(document_id, 0.0)
                + 0.10 * min(1.0, rrf * (RRF_K + 1) / 2.0)
            )
            document = documents[document_id]
            document_tokens = set(
                tokenize(document.title + " " + document.content + " " + " ".join(document.tags))
            )
            matched_terms = sorted(query_tokens & document_tokens, key=len, reverse=True)[:12]
            coverage = len(matched_terms) / float(max(1, len(query_tokens)))
            type_bonus = 0.08 if document.doc_type in intent_types else 0.0
            title_bonus = 0.05 if query.lower() in document.title.lower() else 0.0
            source_bonus = 0.02 if document.page is not None and document.source else 0.0
            rerank_score = fusion_score + 0.12 * coverage + type_bonus + title_bonus + source_bonus
            reasons = []
            if lexical_rank:
                reasons.append("BM25 rank {}".format(lexical_rank))
            if vector_rank:
                reasons.append("vector rank {}".format(vector_rank))
            if type_bonus:
                reasons.append("doc_type intent match")
            if source_bonus:
                reasons.append("page-level evidence")
            fused.append(
                SearchResult(
                    document=document,
                    score=round(rerank_score, 6),
                    lexical_score=round(lexical_score, 6),
                    vector_score=round(vector_score, 6),
                    fusion_score=round(fusion_score, 6),
                    rerank_score=round(rerank_score, 6),
                    matched_terms=matched_terms,
                    reasons=reasons,
                )
            )

        ranked = sorted(
            fused,
            key=lambda item: (item.rerank_score, item.lexical_score, item.document.id),
            reverse=True,
        )[:limit]
        self.last_trace = {
            "mode": "bm25+vector+rrf+metadata_rerank",
            "embedding_provider": provider.name,
            "embedding_fallback_reason": fallback_reason,
            "document_count": len(self.documents),
            "lexical_candidates": len(lexical),
            "vector_candidates": len(vector),
            "candidate_count": len(candidate_ids),
            "selected_count": len(ranked),
            "latency_ms": int((time.time() - started) * 1000),
            "top_ids": [item.document.id for item in ranked],
        }
        return ranked

    def _bm25(self, query: str) -> List[Tuple[KnowledgeDocument, float]]:
        query_tokens = tokenize(query)
        tokenized_documents = [
            tokenize(document.content)
            + tokenize(document.title) * 3
            + tokenize(" ".join(document.tags)) * 2
            for document in self.documents
        ]
        document_count = len(tokenized_documents)
        average_length = sum(len(tokens) for tokens in tokenized_documents) / float(
            max(1, document_count)
        )
        document_frequency: Counter = Counter()
        for tokens in tokenized_documents:
            document_frequency.update(set(tokens))

        ranked = []
        k1 = 1.5
        b = 0.75
        for document, tokens in zip(self.documents, tokenized_documents):
            frequencies = Counter(tokens)
            length = len(tokens)
            score = 0.0
            for token in query_tokens:
                frequency = frequencies.get(token, 0)
                if not frequency:
                    continue
                df = document_frequency[token]
                inverse_frequency = math.log(
                    1.0 + (document_count - df + 0.5) / (df + 0.5)
                )
                denominator = frequency + k1 * (
                    1.0 - b + b * length / max(1.0, average_length)
                )
                score += inverse_frequency * frequency * (k1 + 1.0) / denominator
            if score > 0:
                ranked.append((document, score))
        return sorted(ranked, key=lambda item: (item[1], item[0].id), reverse=True)

    def _vector_search(
        self, query: str, provider: EmbeddingProvider
    ) -> List[Tuple[KnowledgeDocument, float]]:
        document_vectors: List[List[float]] = []
        missing_documents: List[KnowledgeDocument] = []
        missing_positions: List[int] = []
        for index, document in enumerate(self.documents):
            fingerprint = self._fingerprint(document)
            vector = self.vector_index.get(document.id, fingerprint, provider.name)
            if vector is None:
                document_vectors.append([])
                missing_documents.append(document)
                missing_positions.append(index)
            else:
                document_vectors.append(vector)

        if missing_documents:
            texts = [self._embedding_text(document) for document in missing_documents]
            embedded = provider.embed(texts)
            if len(embedded) != len(missing_documents):
                raise ValueError("Embedding provider returned an unexpected vector count")
            for position, document, vector in zip(
                missing_positions, missing_documents, embedded
            ):
                normalized = normalize_vector([float(value) for value in vector])
                document_vectors[position] = normalized
                self.vector_index.put(
                    document.id,
                    self._fingerprint(document),
                    provider.name,
                    normalized,
                )

        query_vectors = provider.embed([query])
        if not query_vectors:
            return []
        query_vector = normalize_vector(query_vectors[0])
        ranked = []
        for document, vector in zip(self.documents, document_vectors):
            if len(vector) != len(query_vector):
                continue
            score = sum(left * right for left, right in zip(query_vector, vector))
            ranked.append((document, score))
        return sorted(ranked, key=lambda item: (item[1], item[0].id), reverse=True)

    @staticmethod
    def _embedding_text(document: KnowledgeDocument) -> str:
        return "{}\n{}\n{}\n{}".format(
            document.title,
            " ".join(document.tags),
            document.section,
            document.content,
        )

    @staticmethod
    def _fingerprint(document: KnowledgeDocument) -> str:
        payload = "{}\n{}\n{}\n{}\n{}".format(
            document.title,
            document.content,
            " ".join(document.tags),
            document.version,
            document.status,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
        if not scores:
            return {}
        minimum = min(scores.values())
        maximum = max(scores.values())
        if math.isclose(minimum, maximum):
            return {key: 1.0 if maximum > 0 else 0.0 for key in scores}
        return {
            key: (value - minimum) / (maximum - minimum)
            for key, value in scores.items()
        }

    @staticmethod
    def _intent_doc_types(query: str) -> set:
        normalized = query.lower()
        intents = set()
        mapping = {
            "api_contract": ["接口", "api", "参数", "请求", "响应", "调用"],
            "event_contract": ["kafka", "事件", "消息", "topic", "通知"],
            "workflow": ["状态", "流程", "审核节点", "流转"],
            "release_rule": ["灰度", "发布", "兼容", "回滚"],
            "defect": ["缺陷", "badcase", "故障", "复盘"],
            "case_example": ["用例", "步骤", "预期"],
        }
        for doc_type, terms in mapping.items():
            if any(term in normalized for term in terms):
                intents.add(doc_type)
        return intents
