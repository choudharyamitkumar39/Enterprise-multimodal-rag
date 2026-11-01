"""
Hybrid Retrieval and Cross-Encoder Re-Ranking Module.
Combines:
1. Dense Vector Search (Pinecone Cosine Similarity)
2. Sparse Keyword Search (BM25 for exact codes, tokens, numbers)
3. Reciprocal Rank Fusion (RRF) for rank combination
4. Cross-Encoder Re-Ranking (Cross-attention matching)
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

logger = logging.getLogger("hybrid_retriever")


def tokenize_corpus(text: str) -> List[str]:
    """Simple alphanumeric lowercased tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


class HybridRerankRetriever:
    """
    Implements Hybrid Search (Dense + BM25) with Reciprocal Rank Fusion (RRF)
    and an optional Cross-Encoder re-ranker.
    """

    def __init__(
        self,
        vectorstore=None,
        cross_encoder_model: Optional[str] = None,
        rrf_k: int = 60,
    ):
        self.vectorstore = vectorstore
        self.rrf_k = rrf_k
        self.cross_encoder = None

        # Load cross-encoder model if requested
        if cross_encoder_model:
            try:
                from sentence_transformers import CrossEncoder
                self.cross_encoder = CrossEncoder(cross_encoder_model)
                logger.info(f"Loaded CrossEncoder model: {cross_encoder_model}")
            except Exception as e:
                logger.warning(f"Could not load CrossEncoder ({e}); falling back to RRF scoring.")
                self.cross_encoder = None

        # In-memory BM25 index per namespace/corpus
        self._bm25_indices: Dict[str, BM25Okapi] = {}
        self._bm25_docs: Dict[str, List[Document]] = {}

    def index_documents_bm25(self, namespace: str, docs: List[Document]):
        """Builds a BM25 index for the documents in a given namespace."""
        if not docs:
            return
        corpus = [tokenize_corpus(doc.page_content) for doc in docs]
        self._bm25_indices[namespace] = BM25Okapi(corpus)
        self._bm25_docs[namespace] = list(docs)
        logger.info(f"Built BM25 index for namespace '{namespace}' with {len(docs)} documents.")

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        namespace: Optional[str] = None,
        candidate_pool_size: int = 15,
    ) -> List[Document]:
        """
        Executes:
        1. Dense Vector Search via Pinecone -> candidate list
        2. Sparse BM25 Search on local index -> candidate list
        3. Reciprocal Rank Fusion (RRF) to merge ranks
        4. Cross-Encoder Re-Ranking over top candidates to pick final top-k
        """
        # 1. Dense retrieval from Pinecone
        dense_docs = []
        if self.vectorstore:
            try:
                retriever = self.vectorstore.as_retriever(
                    search_kwargs={"k": candidate_pool_size, "namespace": namespace}
                )
                dense_docs = retriever.invoke(query)
            except Exception as e:
                logger.warning(f"Dense retrieval error ({e}), proceeding with available candidates.")

        # 2. Sparse retrieval from BM25
        sparse_docs = []
        if namespace in self._bm25_indices:
            bm25 = self._bm25_indices[namespace]
            all_docs = self._bm25_docs[namespace]
            tokenized_query = tokenize_corpus(query)
            scores = bm25.get_scores(tokenized_query)
            # Pick top candidate_pool_size by score
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:candidate_pool_size]
            sparse_docs = [all_docs[i] for i in top_indices if scores[i] > 0]

        # If BM25 is not yet populated for this namespace, fall back directly to dense
        if not sparse_docs:
            merged_candidates = dense_docs[:candidate_pool_size]
        else:
            # 3. Reciprocal Rank Fusion (RRF)
            # Key = (page_content_prefix, page_number) to identify unique chunks
            doc_scores: Dict[str, float] = {}
            doc_lookup: Dict[str, Document] = {}

            # Score dense results
            for rank, doc in enumerate(dense_docs):
                key = f"{doc.metadata.get('page')}_{doc.page_content[:80]}"
                doc_scores[key] = doc_scores.get(key, 0.0) + (1.0 / (self.rrf_k + rank + 1))
                doc_lookup[key] = doc

            # Score sparse results
            for rank, doc in enumerate(sparse_docs):
                key = f"{doc.metadata.get('page')}_{doc.page_content[:80]}"
                doc_scores[key] = doc_scores.get(key, 0.0) + (1.0 / (self.rrf_k + rank + 1))
                doc_lookup[key] = doc

            sorted_keys = sorted(doc_scores.keys(), key=lambda k: doc_scores[k], reverse=True)
            merged_candidates = [doc_lookup[k] for k in sorted_keys[:candidate_pool_size]]

        if not merged_candidates:
            return []

        # 4. Cross-Encoder Re-Ranking (if loaded)
        if self.cross_encoder and len(merged_candidates) > 1:
            try:
                pairs = [[query, doc.page_content] for doc in merged_candidates]
                scores = self.cross_encoder.predict(pairs)
                # Sort merged candidates by cross-encoder score
                scored_candidates = sorted(zip(merged_candidates, scores), key=lambda x: x[1], reverse=True)
                # Annotate metadata with reranker score
                final_docs = []
                for doc, score in scored_candidates[:k]:
                    doc.metadata["rerank_score"] = float(score)
                    final_docs.append(doc)
                return final_docs
            except Exception as e:
                logger.warning(f"Cross-Encoder scoring failed ({e}); falling back to RRF ordering.")

        return merged_candidates[:k]
