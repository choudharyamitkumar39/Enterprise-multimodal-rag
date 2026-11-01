"""
Component-Level Evaluation Module (Level 1)
Evaluates the Retriever and Generator in complete isolation.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from dotenv import load_dotenv

from src.rag_pipeline import MultimodalRAGPipeline, find_env_file

DATASETS_DIR = Path(__file__).resolve().parent / "datasets"


class ComponentEvaluator:
    """Evaluates individual components of the Multimodal RAG system in isolation."""

    def __init__(self, pipeline: MultimodalRAGPipeline = None):
        env_file = find_env_file()
        if env_file:
            load_dotenv(env_file, override=True)
        else:
            load_dotenv(override=True)

        self.pipeline = pipeline or MultimodalRAGPipeline()
        self.retriever_bench_file = DATASETS_DIR / "retrieval_bench.json"

    def evaluate_retriever(self, k_values: List[int] = [1, 3, 5]) -> Dict[str, Any]:
        """
        Evaluates Retriever in isolation against target documents:
        - Hit Rate @ k (Is at least 1 relevant page in top-k?)
        - Recall @ k (Fraction of target pages retrieved in top-k)
        - Mean Reciprocal Rank (MRR)
        - Context Precision @ k
        """
        with open(self.retriever_bench_file, "r", encoding="utf-8") as f:
            bench_queries = json.load(f)

        max_k = max(k_values)
        per_query_results = []
        reciprocal_ranks = []
        hit_counts = {k: 0 for k in k_values}
        recalls = {k: [] for k in k_values}
        precisions = {k: [] for k in k_values}

        print("\n" + "=" * 55)
        print(" [LEVEL 1] Running Component-Level Retriever Evaluation")
        print("=" * 55)

        for item in bench_queries:
            qid = item["query_id"]
            query = item["query"]
            target_pages = set(item["target_pages"])
            required_terms = item.get("required_terms", [])

            # Invoke retriever directly with max_k
            t0 = time.time()
            retriever = self.pipeline.vectorstore.as_retriever(search_kwargs={"k": max_k})
            docs = retriever.invoke(query)
            retrieval_time = (time.time() - t0) * 1000  # ms

            retrieved_pages = [int(doc.metadata.get("page", 0)) for doc in docs]
            retrieved_modalities = [str(doc.metadata.get("modality", "")) for doc in docs]

            # Calculate MRR (1 / rank of first relevant page)
            first_rank = 0
            for rank, page in enumerate(retrieved_pages, start=1):
                if page in target_pages:
                    first_rank = rank
                    break
            rr = 1.0 / first_rank if first_rank > 0 else 0.0
            reciprocal_ranks.append(rr)

            # Metrics per k
            row = {
                "Query ID": qid,
                "Query": query[:45] + "...",
                "Target Pages": list(target_pages),
                "Top Chunks": retrieved_pages,
                "Modalities": retrieved_modalities,
                "MRR": round(rr, 3),
                "Latency (ms)": round(retrieval_time, 1),
            }

            for k in k_values:
                top_k_pages = set(retrieved_pages[:k])
                # Hit Rate
                is_hit = bool(top_k_pages.intersection(target_pages))
                if is_hit:
                    hit_counts[k] += 1
                row[f"Hit@{k}"] = is_hit

                # Recall
                recall_val = len(top_k_pages.intersection(target_pages)) / len(target_pages)
                recalls[k].append(recall_val)

                # Precision
                hits_in_k = sum(1 for p in retrieved_pages[:k] if p in target_pages)
                precisions[k].append(hits_in_k / k)

            per_query_results.append(row)

        total_queries = len(bench_queries)
        hit_rates = {f"Hit Rate @ {k}": f"{(hit_counts[k] / total_queries) * 100:.1f}%" for k in k_values}
        mean_recalls = {f"Mean Recall @ {k}": f"{(sum(recalls[k]) / total_queries) * 100:.1f}%" for k in k_values}
        mean_precisions = {f"Precision @ {k}": f"{(sum(precisions[k]) / total_queries) * 100:.1f}%" for k in k_values}
        overall_mrr = sum(reciprocal_ranks) / total_queries

        summary = {
            "Total Retrieval Queries": total_queries,
            "Mean Reciprocal Rank (MRR)": round(overall_mrr, 3),
            **hit_rates,
            **mean_recalls,
            **mean_precisions,
        }

        df_results = pd.DataFrame(per_query_results)
        return {"summary": summary, "details": df_results}

    def evaluate_generator_in_isolation(self) -> Dict[str, Any]:
        """
        Evaluates the Generator in complete isolation by feeding clean,
        synthetic ground-truth contexts directly (eliminating retrieval noise).
        Tests both Text LLM and Vision LLM synthesis capabilities.
        """
        print("\n" + "=" * 55)
        print(" [LEVEL 1] Running Component-Level Generator Isolation Evaluation")
        print("=" * 55)

        test_cases = [
            {
                "id": "GEN_TEXT_CLEAN",
                "type": "text",
                "clean_context": "[Page 1 | TEXT]\nNovaCore Systems Ltd. was founded in Singapore. In FY2026, total revenue reached $132.0M with 1,480 employees across 42 countries.",
                "question": "What is NovaCore's revenue and where is it located?",
                "expected_facts": ["132.0", "Singapore"],
            },
            {
                "id": "GEN_TABLE_CLEAN",
                "type": "table",
                "clean_context": "[Page 4 | TABLE]\n| Region | 2026 revenue | YoY growth | CSAT |\n| Europe | $41.2M | 31% | 4.7/5 |\n| Asia Pacific | $34.9M | 24% | 4.8/5 |",
                "question": "Which region had the highest growth and what was its percentage?",
                "expected_facts": ["Europe", "31%"],
            },
            {
                "id": "GEN_VISION_CLEAN",
                "type": "visual",
                "clean_context": "[Page 7 | VISUAL]\nFigure 4: Average support resolution time dropped from 14.2 hours in January to 8.1 hours in August 2026.",
                "image_path": str(PROJECT_ROOT / "data" / "extracted_images" / "page_7_image_1.png"),
                "question": "What was the support resolution time in August?",
                "expected_facts": ["8.1"],
            },
        ]

        results = []
        for tc in test_cases:
            t0 = time.time()
            if tc["type"] == "visual" and Path(tc["image_path"]).exists():
                answer = self.pipeline.answer_with_vision(
                    question=tc["question"],
                    context=tc["clean_context"],
                    image_paths=[tc["image_path"]],
                )
            else:
                answer = self.pipeline.text_rag_chain.invoke(
                    {"context": tc["clean_context"], "question": tc["question"], "chat_history_block": ""}
                )
            lat = (time.time() - t0) * 1000

            # Verify factual preservation
            hits = sum(1 for f in tc["expected_facts"] if f.lower() in answer.lower())
            fact_score = hits / len(tc["expected_facts"])

            results.append({
                "Test ID": tc["id"],
                "Generator Modality": tc["type"],
                "Fact Retention": f"{fact_score * 100:.0f}%",
                "Latency (ms)": round(lat, 1),
                "Passed": fact_score == 1.0,
            })

        df = pd.DataFrame(results)
        summary = {
            "Total Generator Tests": len(test_cases),
            "Clean Context Fact Accuracy": f"{(df['Passed'].sum() / len(test_cases)) * 100:.1f}%",
            "Mean Generator Latency": f"{df['Latency (ms)'].mean():.1f} ms",
        }
        return {"summary": summary, "details": df}


if __name__ == "__main__":
    evaluator = ComponentEvaluator()
    ret_res = evaluator.evaluate_retriever()
    print("\n--- RETRIEVER SUMMARY ---")
    for k, v in ret_res["summary"].items():
        print(f"{k:<30}: {v}")

    gen_res = evaluator.evaluate_generator_in_isolation()
    print("\n--- GENERATOR SUMMARY ---")
    for k, v in gen_res["summary"].items():
        print(f"{k:<30}: {v}")
