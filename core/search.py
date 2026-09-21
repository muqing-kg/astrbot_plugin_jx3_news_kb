"""Hybrid retrieval over SQLite FTS5 and stored embeddings."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from typing import Any

from .database import Database


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def quote_fts_term(term: str) -> str:
    return '"' + term.replace('"', '""') + '"'


class SearchService:
    def __init__(
        self,
        database: Database,
        embedding_provider: Any | None = None,
        reranker_provider: Any | None = None,
        fulltext_top_k: int = 24,
        vector_top_k: int = 24,
    ) -> None:
        self.db = database
        self._embedding_provider = embedding_provider
        self._reranker_provider = reranker_provider
        self.fulltext_top_k = max(1, int(fulltext_top_k))
        self.vector_top_k = max(1, int(vector_top_k))

    @property
    def embedding_provider(self) -> Any | None:
        return self._embedding_provider() if callable(self._embedding_provider) else self._embedding_provider

    @embedding_provider.setter
    def embedding_provider(self, provider: Any | None) -> None:
        self._embedding_provider = provider

    @property
    def reranker_provider(self) -> Any | None:
        return self._reranker_provider() if callable(self._reranker_provider) else self._reranker_provider

    @reranker_provider.setter
    def reranker_provider(self, provider: Any | None) -> None:
        self._reranker_provider = provider

    async def search(
        self,
        query: str,
        keywords: list[str] | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        announcement_type: str | None = None,
        top_k: int = 8,
        use_vector: bool = True,
        use_rerank: bool = True,
    ) -> list[dict[str, Any]]:
        terms = self._extract_terms(query, keywords)
        if not terms:
            return []

        fulltext = self._fulltext_search(
            terms, date_from, date_to, announcement_type, self.fulltext_top_k
        )
        vector: list[dict[str, Any]] = []
        if use_vector and self.embedding_provider is not None:
            try:
                vector = await self._vector_search(
                    query,
                    date_from,
                    date_to,
                    announcement_type,
                    self.vector_top_k,
                )
            except Exception:
                vector = []

        merged = self._merge(fulltext, vector)
        if use_rerank and self.reranker_provider is not None and merged:
            try:
                merged = await self._rerank(query, merged, top_k)
            except Exception:
                pass

        now = datetime.now()
        for item in merged:
            try:
                published = datetime.fromisoformat(item["published_at"])
                if published.tzinfo is not None:
                    published = published.astimezone().replace(tzinfo=None)
            except (TypeError, ValueError):
                days_old = 3650
            else:
                days_old = max(0.0, (now - published).total_seconds() / 86400)
            date_score = max(0.0, 1.0 - days_old / 365.0)
            item["date_score"] = date_score
            item["final_score"] = (
                0.58 * float(item.get("relevance_score", 0.0))
                + 0.22 * date_score
                + 0.20 * float(item.get("vector_score", 0.0))
            )

        merged.sort(key=lambda item: item["final_score"], reverse=True)
        return merged[: max(1, int(top_k))]

    def _extract_terms(self, query: str, keywords: list[str] | None) -> list[str]:
        terms = [query.strip()] if query.strip() else []
        for keyword in keywords or []:
            keyword = keyword.strip()
            if keyword and keyword not in terms:
                terms.append(keyword)
        cleaned: list[str] = []
        for term in terms:
            pieces = re.split(r"[\s,，、;；]+", term)
            for piece in pieces:
                if len(piece) >= 2 and piece not in cleaned:
                    cleaned.append(piece)
        expanded: list[str] = []
        for term in cleaned:
            expanded.append(term)
            if len(term) > 3 and re.search(r"[\u4e00-\u9fff]", term):
                for start in range(len(term) - 2):
                    trigram = term[start : start + 3]
                    if trigram not in expanded:
                        expanded.append(trigram)
        return expanded[:16]

    def _fulltext_search(
        self,
        terms: list[str],
        date_from: str | None,
        date_to: str | None,
        announcement_type: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        match_query = " OR ".join(quote_fts_term(term) for term in terms)
        clauses = ["chunks_fts MATCH ?"]
        params: list[Any] = [match_query]
        if date_from:
            clauses.append("c.announcement_date >= ?")
            params.append(date_from)
        if date_to:
            clauses.append("c.announcement_date <= ?")
            params.append(date_to)
        if announcement_type:
            clauses.append("c.announcement_type = ?")
            params.append(announcement_type)
        params.append(limit)
        rows = self._query(
            f"""
            SELECT c.*, bm25(chunks_fts) AS fts_score
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.chunk_id
            WHERE {' AND '.join(clauses)}
            ORDER BY bm25(chunks_fts)
            LIMIT ?
            """,
            params,
        )
        if rows:
            return [self._row_to_item(row, fulltext_score=1.0) for row in rows]

        # FTS5 trigram cannot match one- or two-character Chinese terms.
        term_clauses = []
        params = []
        for term in terms:
            term_clauses.append("(c.title LIKE ? OR c.content LIKE ?)")
            params.extend([f"%{term}%", f"%{term}%"])
        clauses = ["(" + " OR ".join(term_clauses) + ")"]
        if date_from:
            clauses.append("c.announcement_date >= ?")
            params.append(date_from)
        if date_to:
            clauses.append("c.announcement_date <= ?")
            params.append(date_to)
        if announcement_type:
            clauses.append("c.announcement_type = ?")
            params.append(announcement_type)
        params.append(limit)
        rows = self._query(
            f"""
            SELECT c.*, 1.0 AS fts_score
            FROM chunks c
            WHERE {' AND '.join(clauses)}
            ORDER BY c.announcement_date DESC
            LIMIT ?
            """,
            params,
        )
        return [self._row_to_item(row, fulltext_score=0.7) for row in rows]

    async def _vector_search(
        self,
        query: str,
        date_from: str | None,
        date_to: str | None,
        announcement_type: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        query_vector = await self.embedding_provider.get_embedding(query)
        clauses = ["embedding IS NOT NULL"]
        params: list[Any] = []
        if date_from:
            clauses.append("announcement_date >= ?")
            params.append(date_from)
        if date_to:
            clauses.append("announcement_date <= ?")
            params.append(date_to)
        if announcement_type:
            clauses.append("announcement_type = ?")
            params.append(announcement_type)
        rows = self._query(
            f"""
            SELECT id, announcement_id, revision_id, chunk_index, title,
                   announcement_date, announcement_type, url, content,
                   embedding, embedding_dim
            FROM chunks
            WHERE {' AND '.join(clauses)}
            """,
            params,
        )
        scored: list[tuple[float, Any]] = []
        for row in rows:
            try:
                vector = json.loads(bytes(row["embedding"]).decode("utf-8"))
            except (TypeError, ValueError, UnicodeDecodeError):
                continue
            similarity = cosine_similarity(query_vector, vector)
            scored.append((similarity, row))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            self._row_to_item(row, vector_score=max(0.0, (similarity + 1.0) / 2.0))
            for similarity, row in scored[:limit]
        ]

    async def _rerank(
        self,
        query: str,
        items: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        documents = [item["content"] for item in items]
        results = await self.reranker_provider.rerank(query, documents, top_n=len(items))
        scored: list[tuple[float, dict[str, Any]]] = []
        for result in results:
            index = int(getattr(result, "index", -1))
            if 0 <= index < len(items):
                score = float(getattr(result, "relevance_score", 0.0))
                scored.append((max(0.0, min(1.0, score)), items[index]))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        reranked: list[dict[str, Any]] = []
        for score, item in scored[:top_k]:
            copy = dict(item)
            copy["relevance_score"] = score
            reranked.append(copy)
        return reranked

    def _merge(
        self,
        fulltext: list[dict[str, Any]],
        vector: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: dict[int, dict[str, Any]] = {}
        for item in fulltext:
            copy = dict(item)
            copy["vector_score"] = 0.0
            merged[int(item["chunk_id"])] = copy
        for item in vector:
            chunk_id = int(item["chunk_id"])
            if chunk_id in merged:
                merged[chunk_id]["vector_score"] = float(item["vector_score"])
            else:
                copy = dict(item)
                copy["fulltext_score"] = 0.0
                copy["relevance_score"] = 0.0
                merged[chunk_id] = copy
        return list(merged.values())

    def _row_to_item(
        self,
        row: Any,
        fulltext_score: float = 0.0,
        vector_score: float = 0.0,
    ) -> dict[str, Any]:
        return {
            "chunk_id": int(row["id"]),
            "announcement_id": int(row["announcement_id"]),
            "revision_id": int(row["revision_id"]),
            "title": row["title"],
            "announcement_date": row["announcement_date"],
            "type": row["announcement_type"],
            "url": row["url"],
            "content": row["content"],
            "published_at": row["announcement_date"],
            "fulltext_score": fulltext_score,
            "vector_score": vector_score,
            "relevance_score": fulltext_score,
        }

    def _query(self, sql: str, params: list[Any]) -> list[Any]:
        with self.db.connect() as conn:
            return conn.execute(sql, params).fetchall()
