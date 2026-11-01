"""
Multi-Document Cross-Analysis & Comparative Synthesis Module.
Enables cross-document reasoning, parallel multi-namespace retrieval,
and side-by-side comparative matrix synthesis with multi-doc citations.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional
from groq import Groq

logger = logging.getLogger("multi_doc_analyzer")


class MultiDocComparator:
    """
    Coordinates comparative intelligence across multiple indexed documents.
    Features:
    - Parallel multi-namespace retrieval (ThreadPoolExecutor)
    - Comparative Query Decomposition
    - Structured Side-by-Side Matrix Synthesis
    - Distinct Multi-Doc Citations (e.g. [Doc 1: Page 4] vs [Doc 2: Page 2])
    """

    def __init__(self, pipeline, groq_client: Groq, model: str = "openai/gpt-oss-20b"):
        self.pipeline = pipeline
        self.client = groq_client
        self.model = model

    def decompose_query(self, user_query: str, doc_names: List[str]) -> Dict[str, str]:
        """
        Decomposes a comparative question into targeted sub-questions for each document.
        """
        prompt = f"""You are a multi-document query planner.
The user wants to compare information across {len(doc_names)} documents: {', '.join(doc_names)}.

USER QUESTION: {user_query}

Respond with a JSON object containing a "queries" dictionary mapping each document name to a targeted search query.

JSON output:"""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a JSON assistant. Respond ONLY with a valid JSON object."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_completion_tokens=250,
                response_format={"type": "json_object"},
            )
            import json
            data = json.loads(resp.choices[0].message.content)
            return data.get("queries", {doc: user_query for doc in doc_names})
        except Exception as e:
            logger.warning(f"Query decomposition fallback ({e})")
            return {doc: user_query for doc in doc_names}

    def parallel_retrieve(
        self,
        doc_targets: List[Dict[str, str]],  # [{"name": "...", "namespace": "..."}, ...]
        query: str,
        k: int = 4,
    ) -> Dict[str, Any]:
        """
        Retrieves context chunks from each document namespace in parallel.
        """
        results = {}

        def fetch_doc_context(target):
            name = target["name"]
            ns = target["namespace"]
            try:
                if hasattr(self.pipeline, "hybrid_retriever"):
                    docs = self.pipeline.hybrid_retriever.hybrid_search(
                        query=query,
                        k=k,
                        namespace=ns,
                    )
                else:
                    docs = self.pipeline.retriever.invoke(query)
                return name, docs
            except Exception as e:
                logger.error(f"Error retrieving for {name}: {e}")
                return name, []

        with ThreadPoolExecutor(max_workers=min(len(doc_targets), 4)) as executor:
            fetched = executor.map(fetch_doc_context, doc_targets)
            for name, docs in fetched:
                results[name] = docs

        return results

    def comparative_synthesis(
        self,
        question: str,
        doc_contexts: Dict[str, List[Any]],  # {doc_name: [Document, ...]}
    ) -> str:
        """
        Synthesizes a structured comparative report with:
        1. Executive Takeaways
        2. Side-by-Side Markdown Comparison Table
        3. Document-specific page citations
        """
        # Format multi-document context blocks
        formatted_blocks = []
        for doc_name, docs in doc_contexts.items():
            parts = []
            for d in docs:
                p = d.metadata.get("page", "?")
                mod = str(d.metadata.get("modality", "")).upper()
                parts.append(f"[{doc_name} | Page {p} | {mod}]\n{d.page_content}")
            formatted_blocks.append(f"=== EVIDENCE FROM {doc_name.upper()} ===\n" + "\n\n".join(parts))

        combined_context = "\n\n" + ("=" * 45) + "\n\n".join(formatted_blocks)

        prompt = f"""You are an enterprise research analyst providing an executive cross-document comparison.

Use ONLY the provided evidence from the respective documents.

INSTRUCTIONS:
1. Provide a concise 'Executive Summary' comparing the key findings across documents.
2. Provide a structured 'Comparison Matrix' (Markdown table) with columns for Key Dimension / Metric and one column per document.
3. Conclude with 'Key Differences & Takeaways'.
4. STRICT AUDITABILITY: Always cite the source document and page number, e.g. [{list(doc_contexts.keys())[0]}: Page X].
5. If a document does not mention a metric, state 'Not reported in document' in the table cell.

{combined_context}

USER QUESTION:
{question}

COMPARATIVE SYNTHESIS:"""

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_completion_tokens=1200,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"Comparative analysis encountered an issue: {e}"
