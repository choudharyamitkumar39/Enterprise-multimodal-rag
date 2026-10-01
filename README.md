# Multimodal RAG with LangChain & Pinecone

This project demonstrates a complete **Multimodal Retrieval-Augmented Generation (RAG)** pipeline. It extracts text, tables, and images from enterprise PDF reports, creates multimodal embeddings, stores them in a Pinecone vector database, and uses a Vision-Language Model (VLM) to answer complex questions across all modalities.

## Project Structure

```
Multimodal RAG/
├── .env                        # API keys (Groq, Pinecone)
├── README.md                   # Project documentation
├── assets/                     # Architecture diagrams
├── data/
│   ├── extracted_images/       # Images extracted from PDF during ingestion
│   └── raw/                    # Source PDF documents
├── docs/                       # Presentation and reference materials
└── notebooks/
    └── Build_Multimodal_RAG_NovaCore.ipynb  # Main pipeline notebook
```

## Architecture

### 1. The Multimodal Problem
Traditional RAG pipelines flatten PDFs into pure text, completely losing the structured meaning of tables and the visual intelligence encoded in graphs and diagrams. This project solves that by preserving and routing different modalities.

![Overview](assets/overview.png)

### 2. Ingestion Pipeline
The ingestion pipeline processes the PDF using `PyMuPDF`. 
- **Text & Tables** are preserved as structured LangChain documents.
- **Images & Charts** are passed through a Vision Model to generate a rich factual summary.
- All modalities are then embedded using a single text embedding model and stored in **Pinecone**.

![Ingestion Pipeline](assets/ingestion_pipeline.png)

### 3. Generation & Routing
At query time, the system performs a semantic search in Pinecone. If the retrieved context includes a visual element, the system automatically loads the original raw image and routes the query to a Vision-Language Model (VLM). Otherwise, it uses a standard Text LLM.

![Routing](assets/generation_routing.png)

### 4. End-to-End Flow
![End-to-End](assets/end_to_end_architecture.png)

## Implementation Details
*   **PDF Extraction**: `PyMuPDF` (extracts text, tables as pandas/markdown, and raw images).
*   **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace).
*   **Vector Database**: `Pinecone` (Serverless index).
*   **Text LLM**: `openai/gpt-oss-20b` (Served via Groq).
*   **Vision LLM**: `qwen/qwen3.8-27b` (Served via Groq).
*   **Orchestration**: `LangChain`.

## Setup & Usage

1. Create a `.env` file in the root directory:
   ```
   GROQ_API_KEY=your_key_here
   PINECONE_API_KEY=your_key_here
   ```
2. Open `notebooks/Build_Multimodal_RAG_NovaCore.ipynb`.
3. Run all cells to ingest the document and test the multimodal queries!
