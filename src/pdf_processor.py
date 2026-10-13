import os
import io
import fitz
import base64
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional
from PIL import Image
from groq import Groq
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv


def image_to_data_uri(image_path: Path) -> str:
    """Encodes an image to a base64 JPEG data URI."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((1600, 1600))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def summarize_visual(image_path: Path, page_number: int, groq_client: Groq, vision_model: str) -> str:
    """Uses Groq Vision model to generate a rich factual summary of a visual."""
    image_data = image_to_data_uri(image_path)
    prompt = f"""
This visual was extracted from page {page_number} of the document.

Describe the useful factual information visible in the visual.

If it is a chart or graph:
- Mention important data values and labels
- Mention highest and lowest values
- Mention the main trend or percentages

If it is a diagram or flow:
- Identify important components, labels, and boxes
- Explain the flow, stages, or relationships

If it is a photograph or illustration:
- Describe the relevant factual business information

Keep the description concise, factual, and accurate.
"""
    response = groq_client.chat.completions.create(
        model=vision_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data}},
                ],
            }
        ],
        temperature=0,
        max_completion_tokens=500,
    )
    return response.choices[0].message.content.strip()


def process_and_index_pdf(
    pdf_path: Path,
    namespace: str,
    index_name: str = "novacore-multimodal-rag",
    vision_model: str = "qwen/qwen3.8-27b",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> int:
    """
    Parses any PDF into text, tables, and images, generates visual summaries,
    and indexes them into Pinecone under a dedicated chat namespace.
    
    Returns the total number of LangChain documents indexed.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at {pdf_path}")

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is not set.")
    groq_client = Groq(api_key=groq_api_key)

    # Output directory for extracted images
    project_root = Path(__file__).resolve().parent.parent
    extracted_images_dir = project_root / "data" / "extracted_images" / namespace
    extracted_images_dir.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback(0.1, "Opening and inspecting PDF document...")

    pdf = fitz.open(str(pdf_path))
    total_pages = len(pdf)
    documents: List[Document] = []
    extracted_xrefs = set()

    for page_idx in range(total_pages):
        page = pdf[page_idx]
        page_num = page_idx + 1
        page_progress = 0.1 + (0.6 * (page_idx / total_pages))

        if progress_callback:
            progress_callback(page_progress, f"Extracting page {page_num} of {total_pages} (text, tables, visuals)...")

        # 1. Text
        text = page.get_text("text").strip()
        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "page": page_num,
                        "modality": "text",
                        "source": pdf_path.name,
                    },
                )
            )

        # 2. Tables
        try:
            tabs = page.find_tables().tables
            for tab_num, table in enumerate(tabs, start=1):
                df = table.to_pandas()
                if not df.empty:
                    documents.append(
                        Document(
                            page_content=df.to_markdown(index=False),
                            metadata={
                                "page": page_num,
                                "modality": "table",
                                "table_number": tab_num,
                                "source": pdf_path.name,
                            },
                        )
                    )
        except Exception:
            pass

        # 3. Visuals (Images / Charts)
        images = page.get_images(full=True)
        for img_num, img_info in enumerate(images, start=1):
            xref = img_info[0]
            if xref in extracted_xrefs:
                continue
            extracted_xrefs.add(xref)

            try:
                img_data = pdf.extract_image(xref)
                img_bytes = img_data["image"]
                img_ext = img_data["ext"]

                # Filter out microscopic icons (< 5KB or tiny dims)
                if len(img_bytes) < 3000:
                    continue

                img_file = extracted_images_dir / f"page_{page_num}_img_{img_num}.{img_ext}"
                img_file.write_bytes(img_bytes)

                if progress_callback:
                    progress_callback(
                        page_progress + 0.05,
                        f"Generating VLM summary for visual on page {page_num}...",
                    )

                summary = summarize_visual(img_file, page_num, groq_client, vision_model)
                documents.append(
                    Document(
                        page_content=summary,
                        metadata={
                            "page": page_num,
                            "modality": "visual",
                            "image_path": str(img_file.resolve()),
                            "source": pdf_path.name,
                        },
                    )
                )
            except Exception:
                pass

    pdf.close()

    if not documents:
        raise ValueError("Could not extract any content from the provided PDF.")

    if progress_callback:
        progress_callback(0.8, f"Generating multimodal embeddings for {len(documents)} document chunks...")

    # Embeddings & Pinecone Upload
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace,
    )

    if progress_callback:
        progress_callback(0.9, f"Uploading vectors to Pinecone under namespace '{namespace}'...")

    vectorstore.add_documents(documents)

    if progress_callback:
        progress_callback(1.0, f"Ready! Successfully indexed {len(documents)} multimodal chunks.")

    return len(documents)
