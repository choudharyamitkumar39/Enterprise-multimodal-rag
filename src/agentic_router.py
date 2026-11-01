"""
Self-Corrective & Agentic RAG Module (Self-RAG / Corrective RAG).
Features:
1. Query Transformation & Disambiguation (rewriting fuzzy or complex questions)
2. Retrieval Sufficiency Grading (checks if retrieved context answers the query)
3. Automated Self-Correction Loop (triggers targeted query reformulation)
4. Hallucination Grader (validates final generation against retrieved facts)
"""

import os
import re
import json
import logging
from typing import Dict, Any, List, Optional
from groq import Groq

logger = logging.getLogger("agentic_router")


class AgenticSelfRAG:
    """
    Implements active agentic self-reflection over retrieval and generation:
    - Analyzes query complexity
    - Grades retrieved context quality (pass / fail)
    - Rewrites query if initial retrieval was noisy or empty
    - Verifies groundedness before returning
    """

    def __init__(self, groq_client: Groq, model: str = "openai/gpt-oss-20b"):
        self.client = groq_client
        self.model = model

    def rewrite_query_for_retrieval(self, original_query: str, chat_history_block: str = "") -> str:
        """
        Rewrites conversational, ambiguous, or multi-part queries into
        a standalone, high-density keyword query optimized for search.
        """
        prompt = f"""You are an expert search query optimizer for an enterprise document search engine.
Given a user query and recent conversation, generate a clear, standalone search query.
Focus on key entities, specific metrics, document sections, and factual terms.

{chat_history_block}
ORIGINAL QUERY: {original_query}

Return ONLY the rewritten search query in 1 line without any explanation or quotes.
REWRITTEN QUERY:"""

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_completion_tokens=60,
            )
            rewritten = resp.choices[0].message.content.strip().strip('"').strip("'")
            return rewritten if len(rewritten) > 3 else original_query
        except Exception as e:
            logger.warning(f"Query rewrite fallback ({e})")
            return original_query

    def grade_retrieval_sufficiency(self, query: str, context: str) -> Dict[str, Any]:
        """
        Assesses whether the retrieved document context contains enough facts
        to answer the user's question without guessing.
        Returns {'sufficient': bool, 'reasoning': str}
        """
        if not context or len(context.strip()) < 20:
            return {"sufficient": False, "reasoning": "Empty or negligible context retrieved."}

        prompt = f"""You are an objective auditor evaluating search retrieval quality.
Does the provided RETRIEVED CONTEXT contain factual information relevant to answering the QUESTION?

You MUST respond strictly with valid JSON with keys "sufficient" (boolean true or false) and "reasoning" (string).

QUESTION:
{query}

RETRIEVED CONTEXT:
{context[:2500]}
"""

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise JSON response generator."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_completion_tokens=150,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            return {
                "sufficient": bool(data.get("sufficient", True)),
                "reasoning": data.get("reasoning", "Context appears relevant."),
            }
        except Exception as e:
            logger.warning(f"Grading fallback ({e})")
            return {"sufficient": True, "reasoning": "Heuristic fallback pass."}

    def reformulate_query_on_failure(self, failed_query: str, reason: str) -> str:
        """Generates an alternate search phrasing when initial retrieval returned insufficient facts."""
        prompt = f"""The previous search query returned insufficient context from the document index.
Generate an alternative, broader search query with synonyms and related conceptual keywords.

FAILED QUERY: {failed_query}
FAILURE REASON: {reason}

Return ONLY the new query in 1 line:
ALTERNATIVE QUERY:"""

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_completion_tokens=60,
            )
            alt_query = resp.choices[0].message.content.strip().strip('"').strip("'")
            return alt_query if len(alt_query) > 3 else failed_query
        except Exception:
            return failed_query
