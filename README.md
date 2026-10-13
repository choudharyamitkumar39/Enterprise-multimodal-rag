# Enterprise Multimodal RAG, LLM Evaluation & LangSmith Observability

A production-grade **Multimodal Retrieval-Augmented Generation (RAG)** pipeline, **Offline Benchmark Suite**, and **LangSmith Online Evaluation System** built with **LangChain**, **Pinecone**, and **Groq (Text & Vision LLMs)**.

The system extracts narrative text, structured tables, and visual artifacts (charts, pie graphs, process diagrams) from enterprise PDF reports, creates unified multimodal embeddings, performs semantic vector search, dynamically routes queries between Text LLMs and Vision-Language Models (VLMs), and provides real-time observability and hallucination detection via **LangSmith**.

---

## 📁 Project Structure

```text
Multimodal RAG/
├── .env                                  # Private API keys (Groq, Pinecone, LangSmith)
├── .env.example                          # Environment configuration template
├── .gitignore                            # Secrets & local cache protection
├── app.py                                # Streamlit Web UI (Upload PDFs, session memory, chat)
├── README.md                             # Comprehensive project documentation
├── assets/                               # Architecture diagrams & evaluation charts
│   ├── overview.png
│   ├── ingestion_pipeline.png
│   ├── generation_routing.png
│   ├── end_to_end_architecture.png
│   └── evaluation_benchmark.png          # Quantitative evaluation benchmark plot
├── data/
│   ├── chat_sessions.json                # Persistent conversation history & session metadata
│   ├── extracted_images/                 # High-resolution images extracted from PDFs
│   └── raw/                              # Source PDF documents
├── docs/                                 # Conceptual slides & presentations
├── notebooks/
│   ├── Build_Multimodal_RAG_NovaCore.ipynb # Ingestion, vector indexing & demo notebook
│   └── Evaluate_Multimodal_RAG.ipynb     # Offline benchmark & LangSmith online evaluation notebook
├── scripts/
│   └── demo_online_evaluation.py         # CLI demo for live production tracing & online feedback
└── src/                                  # Modular production codebase
    ├── __init__.py
    ├── chat_storage.py                   # Chat session storage & history manager
    ├── evaluator.py                      # Offline RAG Triad Evaluator & Golden Benchmark suite
    ├── online_evaluator.py               # LangSmith Online Evaluator (Async feedback hooks)
    ├── pdf_processor.py                  # Dynamic PDF parser (text, tables, vision summarizer)
    └── rag_pipeline.py                   # MultimodalRAGPipeline class (Pinecone + Groq + LangSmith)
```

---

## 🏗️ System Architecture

### 1. The Multimodal Challenge
Standard text-only RAG pipelines flatten PDFs into raw text strings, permanently discarding table row/column boundaries and the critical insights locked inside charts, plots, and workflow diagrams. This architecture treats every modality as a first-class citizen.

![Overview](assets/overview.png)

### 2. Modality-Aware Ingestion Pipeline
1. **Document Parsing**: `PyMuPDF` extracts narrative paragraphs, structured tables (converted to clean markdown), and high-resolution raster images.
2. **Visual Summarization**: Extracted images and charts are passed to a Vision-Language Model (`qwen/qwen3.8-27b`) to generate dense, factual descriptions (e.g., peak values, slopes, control points).
3. **Unified Vector Space**: Text chunks, markdown tables, and visual summaries are embedded using `sentence-transformers/all-MiniLM-L6-v2` into a single, cohesive 384-dimensional **Pinecone Serverless Index**.

![Ingestion Pipeline](assets/ingestion_pipeline.png)

### 3. Dynamic Query Routing
At query time, the system performs cosine semantic search across all modalities in Pinecone:
* **Text / Table Only**: Context is formatted and dispatched to the fast Text LLM (`openai/gpt-oss-20b`).
* **Visual Evidence Found**: The pipeline dynamically loads the original high-resolution raw image from disk and routes both image bytes and textual context to the Vision LLM (`qwen/qwen3.8-27b`).

![Routing](assets/generation_routing.png)

### 4. End-to-End Pipeline
![End-to-End](assets/end_to_end_architecture.png)

---

## 📡 Production Observability & Online Evaluation with LangSmith

In real-world deployments, testing on static benchmarks isn't enough. This pipeline integrates **LangSmith** for end-to-end production observability and **real-time online evaluation**.

```text
User Question
      │
      ▼
[Multimodal RAG Pipeline] ──(Stream Traces)──► [LangSmith Dashboard]
      │                                                ▲
      ▼                                                │  (Logs Feedback)
[Generated Answer] ──► [Async Online Judge] ───────────┘
                       (Faithfulness & Relevance)
```

### 1. Full-Graph Execution Tracing
* Decorated with `@traceable`, every query logs its full execution graph to LangSmith.
* Captures user input, retrieved Pinecone documents, visual metadata payload, token consumption, and response latency.
* Distinct nested runs for `retriever`, `text_rag_chain`, and `multimodal_vision_inference`.

### 2. Asynchronous Online Evaluator (`src/online_evaluator.py`)
* Automatically triggers an **LLM-as-a-Judge** worker in a non-blocking background thread whenever a live query executes.
* Inspects the live interaction for hallucination risk against the retrieved context.
* Attaches structured feedback directly onto the LangSmith trace:
  * **`faithfulness`**: Score (0.0 to 1.0) measuring context grounding and absence of hallucination.
  * **`answer_relevance`**: Score (0.0 to 1.0) evaluating how directly the response answered the user's intent.
  * **`modality_routing`**: Records whether the query was handled via the Text LLM or Vision LLM.

---

## 🧪 Quantitative Offline Evaluation Suite

This project pairs online observability with an automated offline evaluation suite based on the **RAG Triad** principles.

![Evaluation Benchmark](assets/evaluation_benchmark.png)

### Evaluation Metrics

| Metric | Evaluation Method | What It Measures | Target |
| :--- | :--- | :--- | :--- |
| **Modality Routing Accuracy** | Deterministic Match | Does the router accurately invoke the VLM for visual queries and the Text LLM for text/tables? | $\ge 95\%$ |
| **Context Retrieval Precision (Hit Rate)** | Ground Truth Page Matching | Did Pinecone top-$k$ retrieve the exact source page and modality containing the answer? | $\ge 90\%$ |
| **Faithfulness (Groundedness)** | **LLM-as-a-Judge** (1–5 scale) | Are all claims in the answer strictly supported by the retrieved context? (Detects hallucinations). | $\ge 4.5$ |
| **Answer Relevance & Correctness** | **LLM-as-a-Judge** (1–5 scale) | Does the response directly address the user prompt and match ground truth facts? | $\ge 4.5$ |
| **Key Fact Retention** | Token / Entity Checklist | Deterministic validation that critical numerical facts (e.g., `34%`, `8.1h`, `$132.0M`) are preserved. | $\ge 90\%$ |

---

### 📊 Offline Benchmark Results

Evaluated across a curated golden test set spanning **Text**, **Structured Tables**, **Charts**, **Process Diagrams**, and **Cross-Modal Synthesis**:

| Category | Routing Accuracy | Retrieval Hit Rate | Mean Faithfulness (1-5) | Mean Relevance (1-5) | Pass Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Narrative Text** | **100%** | **100%** | **5.0 / 5.0** | **5.0 / 5.0** | **100%** |
| **Structured Tables** | **100%** | **100%** | **5.0 / 5.0** | **4.0 / 5.0** | **100%** |
| **Visual (Charts & Graphs)** | **100%** | **100%** | **5.0 / 5.0** | **5.0 / 5.0** | **100%** |
| **Visual (Diagrams & Flows)** | **100%** | **100%** | **5.0 / 5.0** | **5.0 / 5.0** | **100%** |
| **Cross-Modal Hybrid** | **100%** | **100%** | **5.0 / 5.0** | **5.0 / 5.0** | **100%** |
| **OVERALL AGGREGATE** | **100%** | **100%** | **5.00 / 5.0** | **4.78 / 5.0** | **100%** |

---

## 🛠️ Tech Stack & Implementation Details

* **Orchestration**: `LangChain` (Chains, PromptTemplates, OutputParsers)
* **Production Observability**: `LangSmith` (Traces, Run Trees, Real-time Online Feedback)
* **Vector Database**: `Pinecone` (Serverless Index, Cosine Similarity, Namespace Isolation)
* **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace, 384 dims)
* **Text LLM**: `openai/gpt-oss-20b` (Served via Groq API)
* **Vision LLM (VLM)**: `qwen/qwen3.8-27b` (Served via Groq API)
* **PDF Processing**: `PyMuPDF` (`fitz`) & `Pillow`
* **Evaluation & Testing**: `pandas`, `groq` (LLM-as-a-Judge), `matplotlib`, `seaborn`

---

## 🚀 Setup & Execution

### 1. Environment Configuration
Create a `.env` file in the project root:
```env
# Groq & Pinecone Keys
GROQ_API_KEY=your_groq_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here

# LangSmith Observability & Online Evaluation (Optional but recommended)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=multimodal-rag-production
```

### 2. Launch the Web Interface (Streamlit App)
Start the interactive chat UI with document upload, session history, and multimodal evidence inspection:
```bash
streamlit run app.py
```
*(Your browser will automatically open `http://localhost:8501`)*

### 3. Ingestion & RAG Pipeline Demo (Jupyter)
Open and run all cells in:
```text
notebooks/Build_Multimodal_RAG_NovaCore.ipynb
```

### 4. Run Offline Evaluation & Benchmarks
Execute the offline benchmark suite:
* **Interactive Notebook**: Open [`notebooks/Evaluate_Multimodal_RAG.ipynb`](notebooks/Evaluate_Multimodal_RAG.ipynb).
* **CLI Command**:
  ```bash
  python src/evaluator.py
  ```

### 5. Run Live Online Evaluation & LangSmith Demo
Test live production queries with real-time feedback logging:
```bash
python scripts/demo_online_evaluation.py
```

