import os
import io
import base64
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv

from groq import Groq
from pinecone import Pinecone
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore


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
        root_dir / "novacore_extracted_images" / filename,
        Path("data/extracted_images") / filename,
        Path("../data/extracted_images") / filename,
        Path("novacore_extracted_images") / filename,
        Path("../novacore_extracted_images") / filename,
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    return None


def image_to_data_uri(image_path):
    """Encodes an image to a base64 JPEG data URI."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((1600, 1600))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
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
    """End-to-end Multimodal RAG pipeline with Pinecone and Groq."""

    def __init__(
        self,
        text_model="openai/gpt-oss-20b",
        vision_model="qwen/qwen3.8-27b",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        index_name="novacore-multimodal-rag",
        namespace="fy2026-demo",
        top_k=5,
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
        self.index_name = index_name
        self.namespace = namespace
        self.top_k = top_k

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
You are a helpful assistant answering questions about the
NovaCore Systems FY2026 company report.

Use ONLY the retrieved context below.

If the answer is not available in the context, say:
"I could not find that information in the report."

Mention page numbers when possible.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
        )
        self.text_rag_chain = self.rag_prompt | self.text_llm | StrOutputParser()

    def answer_with_vision(self, question, context, image_paths):
        """Generates answer using Groq Vision model and retrieved images."""
        valid_paths = [p for p in image_paths if p and Path(p).exists()][:3]
        content = [
            {
                "type": "text",
                "text": f"""
You are answering questions about the NovaCore Systems FY2026 report.

Use ONLY the retrieved context and the attached retrieved visuals.

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

        response = self.groq_client.chat.completions.create(
            model=self.vision_model,
            messages=[{"role": "user", "content": content}],
            temperature=0,
            max_completion_tokens=900,
        )
        return response.choices[0].message.content.strip()

    def query(self, question, top_k=None):
        """Execute a full multimodal RAG query and return structured results."""
        effective_k = top_k or self.top_k
        if effective_k != self.top_k:
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": effective_k})
            retrieved_docs = retriever.invoke(question)
        else:
            retrieved_docs = self.retriever.invoke(question)

        context = format_context(retrieved_docs)

        # Check for visual modality
        image_paths = []
        for doc in retrieved_docs:
            if doc.metadata.get("modality") == "visual":
                raw_path = doc.metadata.get("image_path")
                resolved = resolve_image_path(raw_path)
                if resolved:
                    image_paths.append(resolved)

        # Dynamic multimodal routing
        if image_paths:
            routed_to = "visual"
            answer = self.answer_with_vision(
                question=question,
                context=context,
                image_paths=image_paths,
            )
        else:
            routed_to = "text"
            answer = self.text_rag_chain.invoke(
                {"context": context, "question": question}
            )

        return {
            "question": question,
            "answer": answer,
            "retrieved_docs": retrieved_docs,
            "image_paths": image_paths,
            "routed_to": routed_to,
            "context": context,
            "retrieved_modalities": [doc.metadata.get("modality") for doc in retrieved_docs],
            "retrieved_pages": [int(doc.metadata.get("page", 0)) for doc in retrieved_docs],
        }
