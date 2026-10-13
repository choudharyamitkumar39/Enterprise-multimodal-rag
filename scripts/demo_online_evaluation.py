"""
Standalone Demo for LangSmith Online Evaluation & Live Observability
"""
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.rag_pipeline import MultimodalRAGPipeline


def main():
    print("=" * 65)
    print("  Multimodal RAG: LangSmith Online Evaluation & Observability  ")
    print("=" * 65)

    pipeline = MultimodalRAGPipeline()
    online_evaluator = pipeline.online_evaluator

    print(f"\n[Configuration]")
    print(f"Project Name        : {online_evaluator.project_name}")
    print(f"LangSmith Connected : {online_evaluator.is_active}")
    if not online_evaluator.is_active:
        print("💡 Note: Add your real LANGCHAIN_API_KEY to .env to push live traces")
        print("   and evaluator feedback scores to your hosted LangSmith dashboard.")

    test_queries = [
        "What does NovaCore Systems do and where are its headquarters?",
        "According to the support graph, how did average resolution time change from January to August 2026?",
    ]

    for i, q in enumerate(test_queries, start=1):
        print(f"\n{'='*25} Query {i} {'='*25}")
        print(f"Q: {q}")
        
        # Run query through multimodal RAG pipeline
        result = pipeline.query(q, online_eval=False)  # We will evaluate explicitly below
        print(f"\n[Generated Answer]")
        print(result["answer"])
        print(f"\nModality Routed : {result['routed_to']}")
        print(f"Retrieved Pages : {result['retrieved_pages']}")

        # Run online evaluation
        print(f"\n[Live Online Evaluator Feedback]")
        eval_res = online_evaluator.evaluate_run(
            run_id=result.get("run_id"),
            question=result["question"],
            answer=result["answer"],
            context=result["context"],
            routed_to=result["routed_to"],
            background=False,
        )

        print(f"Faithfulness Score : {eval_res['faithfulness']}/5 (Hallucination Detection)")
        print(f"Answer Relevance   : {eval_res['answer_relevance']}/5")
        print(f"Evaluator Rationale: {eval_res['reasoning']}")
        print(f"Synced to Cloud    : {eval_res['synced_to_langsmith']}")

    print("\n" + "=" * 65)
    print("Online evaluation demo completed successfully!")
    print("=" * 65)


if __name__ == "__main__":
    main()
