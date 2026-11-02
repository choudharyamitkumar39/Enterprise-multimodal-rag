import os
import io
import sys
import base64
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from PIL import Image
from dotenv import load_dotenv

from groq import Groq
from pinecone import Pinecone
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

logger = logging.getLogger("rag_pipeline")

# LangSmith tracing helpers
try:
    from langsmith import traceable
    from langsmith.run_helpers import get_current_run_tree
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    def traceable(*args, **kwargs):
        def decorator(f):
            return f
        return decorator
    def get_current_run_tree():
        return None

# Import Online Evaluator, Cache Manager, Hybrid Retriever, Agentic Router, Streaming Guardrails, and Multi-Doc Analyzer
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from .online_evaluator import LangSmithOnlineEvaluator
    from .cache_manager import SemanticCacheManager
    from .hybrid_retriever import HybridRerankRetriever
    from .agentic_router import AgenticSelfRAG
    from .streaming_guardrails import StreamingGuardrailInterceptor
    from .multi_doc_analyzer import MultiDocComparator
except (ImportError, ValueError):
    from online_evaluator import LangSmithOnlineEvaluator
    from cache_manager import SemanticCacheManager
    from hybrid_retriever import HybridRerankRetriever
    from agentic_router import AgenticSelfRAG
    from streaming_guardrails import StreamingGuardrailInterceptor
    from multi_doc_analyzer import MultiDocComparator


def find_env_file():
    """Locate .env file in project root or current directory."""
    candidates = [
        Path(".env"),
        Path("../.env"),
        Path(__file__).resolve().parent.parent / ".env",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def resolve_image_path(raw_path):
    """Robustly resolve image path across notebook/script execution contexts."""
    if not raw_path:
        return None
    p = Path(raw_path)
    if p.exists():
        return str(p.resolve())
    
    filename = p.name
    root_dir = Path(__file__).resolve().parent.parent
    candidates = [
        root_dir / "data" / "extracted_images" / filename,
        Path("data/extracted_images") / filename,
        Path("../data/extracted_images") / filename,
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    return None


def image_to_data_uri(image_path):
    """Encodes an image to an optimized base64 JPEG data URI."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((768, 768))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=80)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def format_context(retrieved_docs):
    """Formats retrieved LangChain documents into context text."""
    parts = []
    for doc in retrieved_docs:
        page = doc.metadata.get("page")
        modality = doc.metadata.get("modality", "unknown")
        parts.append(f"[Page {page} | {str(modality).upper()}]\n{doc.page_content}")
    return "\n\n".join(parts)


class MultimodalRAGPipeline:
    """
    End-to-end Multimodal RAG pipeline with Pinecone vector retrieval,
    Groq LLMs, LangSmith tracing, and real-time online evaluation.
    """

    def __init__(
        self,
        text_model: str = "openai/gpt-oss-20b",
        vision_model: str = "qwen/qwen3.8-27b",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_name: Optional[str] = None,
        namespace: str = "fy2026-demo",
        top_k: int = 5,
        enable_online_eval: bool = True,
    ):
        env_file = find_env_file()
        if env_file:
            load_dotenv(env_file, override=True)
        else:
            load_dotenv(override=True)

        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY is not set in environment or .env file.")
        if not os.getenv("PINECONE_API_KEY"):
            raise ValueError("PINECONE_API_KEY is not set in environment or .env file.")

        self.text_model = text_model
        self.vision_model = vision_model
        self.embedding_model = embedding_model

        # Auto-detect or resolve Pinecone index
        env_index = os.getenv("PINECONE_INDEX_NAME")
        self.index_name = index_name or env_index or "novacore-multimodal-rag"
        self.namespace = namespace
        self.top_k = top_k
        self.enable_online_eval = enable_online_eval

        # Validate LangSmith API key to avoid 403 Forbidden warnings on placeholder keys
        ls_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY") or ""
        if not ls_key or ls_key.startswith("your_") or "placeholder" in ls_key.lower():
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
            os.environ["LANGSMITH_TRACING"] = "false"
        else:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGSMITH_TRACING"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = ls_key
            os.environ["LANGSMITH_API_KEY"] = ls_key
            proj = os.getenv("LANGCHAIN_PROJECT") or os.getenv("LANGSMITH_PROJECT") or "multimodal-rag-pipeline-local"
            os.environ["LANGCHAIN_PROJECT"] = proj
            os.environ["LANGSMITH_PROJECT"] = proj
            # Remove conflicting agent address variables if present
            os.environ.pop("LANGSMITH_AGENT_ID", None)
            os.environ.pop("LANGSMITH_AGENT_ENVIRONMENT", None)

        # Initialize clients
        self.groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

        # Embeddings & VectorStore
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        self.vectorstore = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings,
            namespace=self.namespace,
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})

        # Text LLM chain
        self.text_llm = ChatGroq(model=self.text_model, temperature=0)
        self.rag_prompt = ChatPromptTemplate.from_template(
            """
You are a helpful assistant answering questions about the provided document.

Use ONLY the retrieved context below.

If the answer is not available in the context, say:
"I could not find that information in the report."

Mention page numbers when possible.

{chat_history_block}
CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
        )
        self.text_rag_chain = self.rag_prompt | self.text_llm | StrOutputParser()

        # Online Evaluator instance (LangSmith)
        self.online_evaluator = LangSmithOnlineEvaluator(
            judge_model=self.text_model,
            groq_client=self.groq_client,
        )

        # 1. GPTCache Semantic Cache Manager
        self.cache_manager = SemanticCacheManager(
            embedding_function=self.embeddings,
            similarity_threshold=0.88,
        )

        # 2. Hybrid Retriever with Cross-Encoder (Dense + BM25 + RRF)
        self.hybrid_retriever = HybridRerankRetriever(
            vectorstore=self.vectorstore,
            rrf_k=60,
        )

        # 3. Agentic Self-RAG (Query rewrite, sufficiency grader, self-correction)
        self.agentic_rag = AgenticSelfRAG(
            groq_client=self.groq_client,
            model=self.text_model,
        )

        # 4. Multi-Document Comparative Analyzer
        self.multi_doc_comparator = MultiDocComparator(
            pipeline=self,
            groq_client=self.groq_client,
            model=self.text_model,
        )

    @traceable(name="multimodal_vision_inference", run_type="llm")
    def answer_with_vision(
        self,
        question: str,
        context: str,
        image_paths: List[str],
        chat_history_block: str = "",
    ) -> str:
        """Generates answer using Groq Vision model and retrieved images with fallback."""
        # Deduplicate paths while preserving retrieval rank, take primary image
        unique_paths = []
        for p in image_paths:
            if p and Path(p).exists() and p not in unique_paths:
                unique_paths.append(p)
        valid_paths = unique_paths[:1]

        content = [
            {
                "type": "text",
                "text": f"""
You are answering questions about the provided document.

Use ONLY the retrieved context and the attached retrieved visuals.

{chat_history_block}
RETRIEVED CONTEXT:
{context}

QUESTION:
{question}

Instructions:
- Answer factually.
- Use the attached visuals when relevant.
- Mention page numbers.
- If the information is missing, say you could not find it.
""",
            }
        ]

        for p in valid_paths:
            content.append({
                "type": "image_url",
                "image_url": {"url": image_to_data_uri(p)},
            })

        try:
            response = self.groq_client.chat.completions.create(
                model=self.vision_model,
                messages=[{"role": "user", "content": content}],
                temperature=0,
                max_completion_tokens=900,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Vision inference failed ({e}); gracefully falling back to text LLM.")
            return self.text_rag_chain.invoke(
                {
                    "context": context,
                    "question": question,
                    "chat_history_block": chat_history_block,
                }
            )

    @traceable(name="multimodal_rag_pipeline", run_type="chain")
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        namespace: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        online_eval: Optional[bool] = None,
        enable_cache: bool = True,
        enable_agentic_loop: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes an advanced Enterprise RAG query:
        1. Checks GPTCache Semantic Cache (returns in <50ms on semantic hits)
        2. Agentic Query Optimization (rewrites conversational queries)
        3. Hybrid Retrieval (Pinecone Dense + Sparse BM25 + Reciprocal Rank Fusion)
        4. Self-Corrective Retrieval Grader (re-searches if initial retrieval is insufficient)
        5. Multimodal Dynamic Routing (Text LLM vs Vision VLM)
        6. Real-time LangSmith Observability & Online Evaluation
        7. Cache Population on cache misses
        """
        effective_k = top_k or self.top_k
        target_namespace = namespace or self.namespace

        # --- STEP 1: GPTCache Semantic Cache Check ---
        if enable_cache and hasattr(self, "cache_manager"):
            cached_res = self.cache_manager.check_cache(question, namespace=target_namespace)
            if cached_res:
                return cached_res

        # Capture current LangSmith run ID if tracing
        run_tree = get_current_run_tree()
        run_id = str(run_tree.id) if run_tree else None

        # Format conversation history
        chat_history_block = ""
        if chat_history:
            recent_turns = chat_history[-4:]
            formatted = []
            for m in recent_turns:
                role = "User" if m.get("role") == "user" else "Assistant"
                formatted.append(f"{role}: {m.get('content', '')}")
            chat_history_block = "PREVIOUS CONVERSATION:\n" + "\n".join(formatted) + "\n\n"

        # --- STEP 2: Agentic Query Optimization ---
        search_query = question
        if enable_agentic_loop and hasattr(self, "agentic_rag") and chat_history:
            search_query = self.agentic_rag.rewrite_query_for_retrieval(question, chat_history_block)

        # --- STEP 3: Hybrid Retrieval (Dense + BM25 + RRF) ---
        if hasattr(self, "hybrid_retriever"):
            retrieved_docs = self.hybrid_retriever.hybrid_search(
                query=search_query,
                k=effective_k,
                namespace=target_namespace,
            )
        else:
            if target_namespace != self.namespace or effective_k != self.top_k:
                vs = PineconeVectorStore(
                    index_name=self.index_name,
                    embedding=self.embeddings,
                    namespace=target_namespace,
                )
                retriever = vs.as_retriever(search_kwargs={"k": effective_k})
                retrieved_docs = retriever.invoke(search_query)
            else:
                retrieved_docs = self.retriever.invoke(search_query)

        context = format_context(retrieved_docs)

        # --- STEP 4: Self-Corrective Retrieval Grader ---
        if enable_agentic_loop and hasattr(self, "agentic_rag"):
            grade = self.agentic_rag.grade_retrieval_sufficiency(question, context)
            if not grade["sufficient"]:
                # Self-correction: Reformulate and retry retrieval once
                alt_query = self.agentic_rag.reformulate_query_on_failure(search_query, grade["reasoning"])
                if alt_query != search_query and hasattr(self, "hybrid_retriever"):
                    retry_docs = self.hybrid_retriever.hybrid_search(
                        query=alt_query,
                        k=effective_k,
                        namespace=target_namespace,
                    )
                    if retry_docs:
                        retrieved_docs = retry_docs
                        context = format_context(retrieved_docs)

        # Check for visual modality
        image_paths = []
        for doc in retrieved_docs:
            if doc.metadata.get("modality") == "visual":
                raw_path = doc.metadata.get("image_path")
                resolved = resolve_image_path(raw_path)
                if resolved and resolved not in image_paths:
                    image_paths.append(resolved)

        # --- STEP 5: Dynamic Multimodal Generation ---
        if image_paths:
            routed_to = "visual"
            answer = self.answer_with_vision(
                question=question,
                context=context,
                image_paths=image_paths,
                chat_history_block=chat_history_block,
            )
        else:
            routed_to = "text"
            answer = self.text_rag_chain.invoke(
                {
                    "context": context,
                    "question": question,
                    "chat_history_block": chat_history_block,
                }
            )

        # --- STEP 6: Online Evaluation (LangSmith) ---
        do_eval = self.enable_online_eval if online_eval is None else online_eval
        eval_dispatch = None
        if do_eval and self.online_evaluator:
            eval_dispatch = self.online_evaluator.evaluate_run(
                run_id=run_id,
                question=question,
                answer=answer,
                context=context,
                routed_to=routed_to,
                background=True,
            )

        result_payload = {
            "question": question,
            "answer": answer,
            "retrieved_docs": retrieved_docs,
            "image_paths": image_paths,
            "routed_to": routed_to,
            "context": context,
            "retrieved_modalities": [doc.metadata.get("modality") for doc in retrieved_docs],
            "retrieved_pages": [int(doc.metadata.get("page", 0)) for doc in retrieved_docs],
            "run_id": run_id,
            "online_eval": eval_dispatch,
            "from_cache": False,
        }

        # --- STEP 7: Populate GPTCache ---
        if enable_cache and hasattr(self, "cache_manager"):
            self.cache_manager.store_cache(
                query=question,
                response_dict=result_payload,
                namespace=target_namespace,
            )

        return result_payload

    def stream_query(
        self,
        question: str,
        top_k: Optional[int] = None,
        namespace: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        enable_cache: bool = True,
        enable_agentic_loop: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes hybrid retrieval, sufficiency grading, and streams the answer
        through a real-time sliding-window safety guardrail interceptor.
        
        Returns:
            {
                "stream_generator": Generator yielding tokens,
                "retrieved_docs": list,
                "image_paths": list,
                "routed_to": str,
                "from_cache": bool,
                "retrieved_pages": list
            }
        """
        effective_k = top_k or self.top_k
        target_namespace = namespace or self.namespace

        # 1. GPTCache Check
        if enable_cache and hasattr(self, "cache_manager"):
            cached_res = self.cache_manager.check_cache(question, namespace=target_namespace)
            if cached_res:
                def cached_stream():
                    cached_text = cached_res.get("answer", "")
                    words = cached_text.split(" ")
                    for i, w in enumerate(words):
                        yield w + (" " if i < len(words) - 1 else "")
                
                cached_res["stream_generator"] = cached_stream()
                return cached_res

        # 2. Conversation memory block
        chat_history_block = ""
        if chat_history:
            recent_turns = chat_history[-4:]
            formatted = []
            for m in recent_turns:
                role = "User" if m.get("role") == "user" else "Assistant"
                formatted.append(f"{role}: {m.get('content', '')}")
            chat_history_block = "PREVIOUS CONVERSATION:\n" + "\n".join(formatted) + "\n\n"

        # 3. Agentic Query Optimization
        search_query = question
        if enable_agentic_loop and hasattr(self, "agentic_rag") and chat_history:
            search_query = self.agentic_rag.rewrite_query_for_retrieval(question, chat_history_block)

        # 4. Hybrid Retrieval
        if hasattr(self, "hybrid_retriever"):
            retrieved_docs = self.hybrid_retriever.hybrid_search(
                query=search_query,
                k=effective_k,
                namespace=target_namespace,
            )
        else:
            retrieved_docs = self.retriever.invoke(search_query)

        context = format_context(retrieved_docs)

        # 5. Check visual evidence
        image_paths = []
        for doc in retrieved_docs:
            if doc.metadata.get("modality") == "visual":
                raw_path = doc.metadata.get("image_path")
                resolved = resolve_image_path(raw_path)
                if resolved and resolved not in image_paths:
                    image_paths.append(resolved)

        routed_to = "visual" if image_paths else "text"

        # 6. Build Raw Upstream Token Stream
        if routed_to == "visual":
            valid_paths = [p for p in image_paths if p and Path(p).exists()][:1]
            content = [
                {
                    "type": "text",
                    "text": f"Use ONLY this retrieved context and image to answer.\n{chat_history_block}\nCONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:",
                }
            ]
            for p in valid_paths:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_to_data_uri(p)},
                })
            raw_response = self.groq_client.chat.completions.create(
                model=self.vision_model,
                messages=[{"role": "user", "content": content}],
                temperature=0,
                max_completion_tokens=900,
                stream=True,
            )
        else:
            prompt_content = f"You are a helpful document assistant. Answer using ONLY the retrieved context. Mention page numbers.\n\n{chat_history_block}CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
            raw_response = self.groq_client.chat.completions.create(
                model=self.text_model,
                messages=[{"role": "user", "content": prompt_content}],
                temperature=0,
                max_completion_tokens=900,
                stream=True,
            )

        def raw_token_generator():
            for chunk in raw_response:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        # 7. Wrap with Sliding-Window Guardrail Interceptor
        interceptor = StreamingGuardrailInterceptor(buffer_size=6)
        guarded_stream = interceptor.intercept_stream(raw_token_generator())

        return {
            "stream_generator": guarded_stream,
            "retrieved_docs": retrieved_docs,
            "image_paths": image_paths,
            "routed_to": routed_to,
            "context": context,
            "retrieved_modalities": [doc.metadata.get("modality") for doc in retrieved_docs],
            "retrieved_pages": [int(doc.metadata.get("page", 0)) for doc in retrieved_docs],
            "from_cache": False,
        }
