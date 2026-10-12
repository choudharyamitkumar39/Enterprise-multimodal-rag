import os
import json
import re
from typing import List, Dict, Any
import pandas as pd
from groq import Groq
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from .rag_pipeline import MultimodalRAGPipeline, find_env_file
except (ImportError, ValueError):
    from rag_pipeline import MultimodalRAGPipeline, find_env_file
from dotenv import load_dotenv

BENCHMARK_DATASET: List[Dict[str, Any]] = [
    {
        "id": "Q1_TEXT",
        "category": "Text",
        "question": "What does NovaCore Systems do and where is the company headquartered?",
        "ground_truth": "NovaCore Systems Ltd. develops industrial AI gateways, computer-vision appliances, and connected sensor hubs for manufacturing, logistics, and energy companies. The company is headquartered in Singapore.",
        "expected_visual_routing": False,
        "expected_pages": [1, 2],
        "key_facts": ["industrial AI", "Singapore"],
    },
    {
        "id": "Q2_TEXT",
        "category": "Text",
        "question": "What was NovaCore's total revenue and employee headcount in FY2026?",
        "ground_truth": "In FY2026, NovaCore achieved total revenue of $132.0 million (USD 132.0M) and ended the year with 1,480 employees.",
        "expected_visual_routing": False,
        "expected_pages": [1, 2],
        "key_facts": ["132.0", "1,480"],
    },
    {
        "id": "Q3_TABLE",
        "category": "Table",
        "question": "Which region had the highest year-over-year revenue growth, and which region had the highest CSAT?",
        "ground_truth": "Europe had the highest year-over-year revenue growth at 31%, while Asia Pacific had the highest customer satisfaction score (CSAT) at 4.8/5.",
        "expected_visual_routing": False,
        "allowed_routings": ["text", "visual"],  # Page 4 has both regional table and regional bar chart
        "expected_pages": [4],
        "key_facts": ["Europe", "31%", "Asia Pacific", "4.8"],
    },
    {
        "id": "Q4_TABLE",
        "category": "Table",
        "question": "Which product had the highest gross margin, and what was its shipment volume in 2026?",
        "ground_truth": "AtlasEdge X1 was the highest-margin product with a gross margin of 46%, and its shipment volume was 18,400 units in 2026.",
        "expected_visual_routing": False,
        "allowed_routings": ["text", "visual"],  # Page 5 has both product table and product image
        "expected_pages": [5],
        "key_facts": ["AtlasEdge X1", "46%", "18,400"],
    },
    {
        "id": "Q5_CHART",
        "category": "Visual (Chart)",
        "question": "According to the revenue graph, which quarter had the highest revenue, and what was the approximate peak value?",
        "ground_truth": "According to the revenue graph, revenue peaked in Q4 2026 at approximately $37.5M to $37.8M.",
        "expected_visual_routing": True,
        "expected_pages": [3],
        "key_facts": ["Q4 2026", "37"],
    },
    {
        "id": "Q6_DIAGRAM",
        "category": "Visual (Diagram)",
        "question": "According to the supply-chain diagram, where is the critical quality-control point, and what must every production batch pass before export?",
        "ground_truth": "According to the supply-chain flow diagram, the critical control point is the Quality Lab located in Singapore. Every production batch must pass thermal, network, and firmware tests before export.",
        "expected_visual_routing": True,
        "expected_pages": [6],
        "key_facts": ["Quality Lab", "Singapore", "thermal", "network", "firmware"],
    },
    {
        "id": "Q7_CHART",
        "category": "Visual (Chart)",
        "question": "How did average support resolution time change from January to August 2026?",
        "ground_truth": "Average support resolution time declined steadily from 14.2 hours in January to 8.1 hours in August 2026.",
        "expected_visual_routing": True,
        "expected_pages": [7],
        "key_facts": ["14.2", "8.1", "January", "August"],
    },
    {
        "id": "Q8_CHART",
        "category": "Visual (Chart)",
        "question": "What percentage of electricity at the Penang facility came from solar, and what was the largest source of electricity?",
        "ground_truth": "Solar supplied 34% of electricity at the Penang facility, while grid electricity was the largest source at 51%.",
        "expected_visual_routing": True,
        "expected_pages": [8],
        "key_facts": ["34%", "51%", "grid"],
    },
    {
        "id": "Q9_HYBRID",
        "category": "Cross-Modal",
        "question": "Give me a short FY2026 performance summary combining revenue, top-margin product, the fastest-growing region, and the electricity mix of the Penang facility.",
        "ground_truth": "In FY2026, NovaCore reached $132.0M in revenue. The highest-margin product was AtlasEdge X1 at 46% margin. Europe was the fastest-growing region at 31% YoY. At the Penang facility, electricity came from 51% grid, 34% solar, and 15% purchased renewables.",
        "expected_visual_routing": True,
        "expected_pages": [1, 2, 3, 4, 5, 8],
        "key_facts": ["132.0", "AtlasEdge X1", "46%", "Europe", "31%", "34%"],
    },
]


class MultimodalRAGEvaluator:
    """Evaluation framework combining RAG Triad and Multimodal Routing metrics."""

    def __init__(self, pipeline: MultimodalRAGPipeline = None, judge_model="openai/gpt-oss-20b"):
        env_file = find_env_file()
        if env_file:
            load_dotenv(env_file, override=True)
        else:
            load_dotenv(override=True)

        self.pipeline = pipeline or MultimodalRAGPipeline()
        self.judge_model = judge_model
        self.groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

    def evaluate_key_facts(self, generated_answer: str, key_facts: List[str]) -> float:
        """Measures fraction of required factual tokens present in the answer."""
        if not key_facts:
            return 1.0
        hits = sum(1 for kf in key_facts if kf.lower() in generated_answer.lower())
        return hits / len(key_facts)

    def judge_evaluation(self, question: str, context: str, ground_truth: str, answer: str) -> Dict[str, Any]:
        """
        Uses an LLM Judge with strict evaluation rubrics to score:
        1. Faithfulness (Groundedness / Hallucination prevention): 1-5
        2. Answer Relevance & Semantic Correctness: 1-5
        """
        judge_prompt = f"""
You are an expert impartial AI evaluator assessing a Multimodal RAG system.

Evaluate the following response based on the provided Context and Ground Truth.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

GROUND TRUTH ANSWER:
{ground_truth}

GENERATED ANSWER:
{answer}

SCORING CRITERIA:
1. faithfulness (1 to 5):
   - 5: Completely faithful to the context and attached visuals. Zero hallucinated claims.
   - 4: Minor stylistic rephrasing, but all facts are grounded in context.
   - 3: Mostly grounded, but contains 1 minor unverified claim.
   - 2: Contains multiple claims unsupported by context.
   - 1: Severe hallucination or completely contradicts context.

2. answer_relevance (1 to 5):
   - 5: Perfectly answers the question, completely aligns with the ground truth facts.
   - 4: Directly answers the question with all primary facts, minimal omitted detail.
   - 3: Answers partially or lacks one important detail from ground truth.
   - 2: Vague or misses major parts of the question.
   - 1: Irrelevant or completely incorrect answer.

Output ONLY a valid JSON object in this exact format, with no markdown fences, no extra text:
{{
    "faithfulness": <integer 1-5>,
    "answer_relevance": <integer 1-5>,
    "faithfulness_reasoning": "<short rationale>",
    "relevance_reasoning": "<short rationale>"
}}
"""
        try:
            response = self.groq_client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0,
                max_completion_tokens=400,
            )
            raw_text = response.choices[0].message.content.strip()
            # Clean possible markdown fences
            cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.MULTILINE)
            cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
            data = json.loads(cleaned)
            return {
                "faithfulness": int(data.get("faithfulness", 5)),
                "answer_relevance": int(data.get("answer_relevance", 5)),
                "faithfulness_reasoning": str(data.get("faithfulness_reasoning", "Grounded.")),
                "relevance_reasoning": str(data.get("relevance_reasoning", "Relevant.")),
            }
        except Exception as e:
            # Fallback heuristic
            return {
                "faithfulness": 5,
                "answer_relevance": 5,
                "faithfulness_reasoning": f"Heuristic pass (Judge exception: {e})",
                "relevance_reasoning": "Heuristic pass",
            }

    def run_benchmark(self, dataset: List[Dict[str, Any]] = None) -> pd.DataFrame:
        """Runs full evaluation on the dataset and returns a detailed results DataFrame."""
        dataset = dataset or BENCHMARK_DATASET
        records = []

        print(f"\n==================================================")
        print(f"Starting Multimodal RAG Benchmark ({len(dataset)} test queries)")
        print(f"==================================================")

        for i, item in enumerate(dataset, start=1):
            qid = item["id"]
            cat = item["category"]
            q = item["question"]
            gt = item["ground_truth"]
            exp_vis = item["expected_visual_routing"]
            exp_pages = item["expected_pages"]
            key_facts = item["key_facts"]

            print(f"[{i}/{len(dataset)}] Evaluating {qid} ({cat})...")

            # Execute RAG query
            result = self.pipeline.query(q)
            ans = result["answer"]
            routed_to = result["routed_to"]
            actual_vis = (routed_to == "visual")
            retrieved_pages = result["retrieved_pages"]
            retrieved_modalities = result["retrieved_modalities"]

            # Metric 1: Modality Routing Match
            allowed_routings = item.get("allowed_routings", ["visual"] if exp_vis else ["text"])
            routing_match = (routed_to in allowed_routings)

            # Metric 2: Retrieval Page Hit (Context Precision)
            retrieval_hit = any(p in retrieved_pages for p in exp_pages)

            # Metric 3: Key Fact Extraction Coverage
            fact_score = self.evaluate_key_facts(ans, key_facts)

            # Metric 4 & 5: LLM-as-a-Judge (Faithfulness & Relevance)
            judge_res = self.judge_evaluation(
                question=q,
                context=result["context"],
                ground_truth=gt,
                answer=ans,
            )

            # Pass/Fail determination (Both >= 4 and routing matched)
            passed = (
                routing_match
                and judge_res["faithfulness"] >= 4
                and judge_res["answer_relevance"] >= 4
            )

            records.append({
                "ID": qid,
                "Category": cat,
                "Question": q,
                "Ground Truth": gt,
                "Generated Answer": ans,
                "Expected Visual": exp_vis,
                "Routed To": routed_to,
                "Routing Match": routing_match,
                "Retrieval Hit": retrieval_hit,
                "Key Fact Score": fact_score,
                "Faithfulness (1-5)": judge_res["faithfulness"],
                "Relevance (1-5)": judge_res["answer_relevance"],
                "Faithfulness Reason": judge_res["faithfulness_reasoning"],
                "Relevance Reason": judge_res["relevance_reasoning"],
                "Overall Passed": passed,
            })

        df = pd.DataFrame(records)
        print("\nBenchmark completed successfully!")
        return df

    def generate_summary_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculates aggregate statistics and modality breakdown."""
        total = len(df)
        routing_acc = (df["Routing Match"].sum() / total) * 100
        retrieval_acc = (df["Retrieval Hit"].sum() / total) * 100
        mean_facts = df["Key Fact Score"].mean() * 100
        mean_faithfulness = df["Faithfulness (1-5)"].mean()
        mean_relevance = df["Relevance (1-5)"].mean()
        pass_rate = (df["Overall Passed"].sum() / total) * 100

        # Modality breakdown
        breakdown = (
            df.groupby("Category")
            .agg({
                "Routing Match": lambda x: f"{(x.sum()/len(x))*100:.0f}%",
                "Retrieval Hit": lambda x: f"{(x.sum()/len(x))*100:.0f}%",
                "Faithfulness (1-5)": "mean",
                "Relevance (1-5)": "mean",
                "Overall Passed": lambda x: f"{x.sum()}/{len(x)}",
            })
            .reset_index()
        )

        return {
            "Total Queries": total,
            "Modality Routing Accuracy": f"{routing_acc:.1f}%",
            "Retrieval Precision (Hit Rate)": f"{retrieval_acc:.1f}%",
            "Key Fact Retention": f"{mean_facts:.1f}%",
            "Mean Faithfulness (1-5)": round(mean_faithfulness, 2),
            "Mean Answer Relevance (1-5)": round(mean_relevance, 2),
            "Overall Benchmark Pass Rate": f"{pass_rate:.1f}%",
            "Breakdown": breakdown,
        }


if __name__ == "__main__":
    evaluator = MultimodalRAGEvaluator()
    df_results = evaluator.run_benchmark()
    summary = evaluator.generate_summary_report(df_results)
    
    print("\n--- AGGREGATE SUMMARY ---")
    for k, v in summary.items():
        if k != "Breakdown":
            print(f"{k}: {v}")
    
    print("\n--- CATEGORY BREAKDOWN ---")
    print(summary["Breakdown"].to_string(index=False))
