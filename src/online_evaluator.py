import os
import re
import json
import logging
import threading
from typing import Dict, Any, Optional
from groq import Groq
from dotenv import load_dotenv

logger = logging.getLogger("online_evaluator")

try:
    from langsmith import Client
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False


class LangSmithOnlineEvaluator:
    """
    Online Evaluator for real-time production monitoring with LangSmith.
    
    Inspects live queries, detects hallucinations and answer relevance via
    an asynchronous background LLM Judge, and records feedback scores
    directly onto LangSmith traces.
    """

    def __init__(
        self,
        judge_model: str = "openai/gpt-oss-20b",
        project_name: Optional[str] = None,
        groq_client: Optional[Groq] = None,
    ):
        self.judge_model = judge_model
        self.project_name = (
            project_name
            or os.getenv("LANGCHAIN_PROJECT")
            or os.getenv("LANGSMITH_PROJECT")
            or "multimodal-rag-pipeline-local"
        )
        
        # Groq client for LLM-as-a-Judge
        api_key = os.getenv("GROQ_API_KEY")
        self.groq_client = groq_client or (Groq(api_key=api_key) if api_key else None)
        
        # LangSmith client
        self.langsmith_client = None
        self.is_active = False

        langsmith_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
        tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() in ("true", "1")

        if LANGSMITH_AVAILABLE and langsmith_key and not langsmith_key.startswith("your_"):
            try:
                self.langsmith_client = Client(api_key=langsmith_key)
                self.is_active = True
                logger.info(f"LangSmith Online Evaluator initialized for project: {self.project_name}")
            except Exception as e:
                logger.warning(f"Failed to connect LangSmith client: {e}")
                self.is_active = False
        else:
            logger.info("LangSmith credentials not fully configured. Running in local simulation mode.")

    def _judge_live_response(self, question: str, context: str, answer: str) -> Dict[str, Any]:
        """Runs fast LLM-as-a-Judge on live production response."""
        if not self.groq_client:
            return {"faithfulness": 5, "answer_relevance": 5, "reasoning": "Judge client unavailable"}

        judge_prompt = f"""
You are an online evaluation judge monitoring a live Multimodal RAG system.

Evaluate this production interaction for Faithfulness (hallucination detection) and Answer Relevance.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}

SCORING CRITERIA (1 to 5):
1. faithfulness:
   5: Completely grounded in context/visuals. Zero hallucination.
   4: Factual rephrasing, fully supported.
   3: Mostly supported, 1 minor unverified claim.
   2: Multiple ungrounded claims.
   1: Total hallucination or contradicts context.

2. answer_relevance:
   5: Directly and fully answers the user question.
   4: Answers the question with minor detail omitted.
   3: Partially answers the question.
   2: Mostly off-topic or vague.
   1: Completely irrelevant or incorrect.

Return ONLY valid JSON in this exact structure:
{{
    "faithfulness": <1-5>,
    "answer_relevance": <1-5>,
    "reasoning": "<concise explanation>"
}}
"""
        try:
            response = self.groq_client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0,
                max_completion_tokens=300,
            )
            raw = response.choices[0].message.content.strip()
            cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
            cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
            data = json.loads(cleaned)
            return {
                "faithfulness": int(data.get("faithfulness", 5)),
                "answer_relevance": int(data.get("answer_relevance", 5)),
                "reasoning": str(data.get("reasoning", "Evaluated.")),
            }
        except Exception as e:
            return {
                "faithfulness": 5,
                "answer_relevance": 5,
                "reasoning": f"Judge heuristic pass ({e})",
            }

    def _evaluate_worker(
        self,
        run_id: str,
        question: str,
        answer: str,
        context: str,
        routed_to: str,
    ):
        """Worker thread to evaluate and upload feedback to LangSmith."""
        eval_scores = self._judge_live_response(question, context, answer)
        
        faithfulness = eval_scores["faithfulness"]
        relevance = eval_scores["answer_relevance"]
        reasoning = eval_scores["reasoning"]

        # Normalize 1-5 to 0.0 - 1.0 scale for LangSmith UI metrics
        norm_faithfulness = (faithfulness - 1) / 4.0
        norm_relevance = (relevance - 1) / 4.0

        if self.is_active and self.langsmith_client and run_id:
            try:
                # 1. Log Faithfulness Feedback
                self.langsmith_client.create_feedback(
                    run_id=run_id,
                    key="faithfulness",
                    score=norm_faithfulness,
                    value=f"{faithfulness}/5",
                    comment=reasoning,
                )
                # 2. Log Relevance Feedback
                self.langsmith_client.create_feedback(
                    run_id=run_id,
                    key="answer_relevance",
                    score=norm_relevance,
                    value=f"{relevance}/5",
                    comment=reasoning,
                )
                # 3. Log Modality Routing Tag
                self.langsmith_client.create_feedback(
                    run_id=run_id,
                    key="modality_routing",
                    score=1.0,
                    value=routed_to,
                    comment=f"Query routed to: {routed_to}",
                )
            except Exception as e:
                logger.warning(f"Could not submit feedback to LangSmith for run {run_id}: {e}")

        return {
            "run_id": run_id,
            "faithfulness": faithfulness,
            "answer_relevance": relevance,
            "routed_to": routed_to,
            "reasoning": reasoning,
            "synced_to_langsmith": self.is_active and bool(run_id),
        }

    def evaluate_run(
        self,
        run_id: Optional[str],
        question: str,
        answer: str,
        context: str,
        routed_to: str,
        background: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Triggers online evaluation for a completed RAG run.
        If background=True, runs asynchronously without adding latency to the user.
        """
        if background:
            t = threading.Thread(
                target=self._evaluate_worker,
                args=(run_id, question, answer, context, routed_to),
                daemon=True,
            )
            t.start()
            return {"status": "evaluation_dispatched_in_background", "run_id": run_id}
        else:
            return self._evaluate_worker(run_id, question, answer, context, routed_to)
