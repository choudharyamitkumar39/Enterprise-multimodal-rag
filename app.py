import os
import sys
from pathlib import Path
from typing import Dict, Any, List

import streamlit as st
from dotenv import load_dotenv

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_pipeline import MultimodalRAGPipeline
from src.pdf_processor import process_and_index_pdf
from src.visualizer import highlight_page_passage
from src.chat_storage import (
    load_all_sessions,
    save_session,
    delete_session,
    create_new_session,
)

# Load environment
load_dotenv(PROJECT_ROOT / ".env", override=True)

# Page configuration
st.set_page_config(
    page_title="Multimodal RAG Assistant",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished ChatGPT-like appearance
st.markdown(
    """
    <style>
    .main-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #6b7280;
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
    }
    .doc-badge {
        display: inline-block;
        background-color: #e0f2fe;
        color: #0369a1;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .metric-badge {
        font-size: 0.8rem;
        color: #059669;
        font-weight: 600;
    }
    div[data-testid="stSidebarHeader"] {
        padding-bottom: 0rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_pipeline():
    """Initializes and caches the Multimodal RAG pipeline."""
    return MultimodalRAGPipeline()


def resolve_session_pdf_path(s_id: str, p_name: Optional[str]) -> Optional[Path]:
    """Resolves the physical PDF path on disk for visual highlighting."""
    raw_dir = PROJECT_ROOT / "data" / "raw"
    if not raw_dir.exists():
        return None
    # 1. Match files beginning with session_id (e.g. 94337b46_Viva-leisure-report.pdf)
    if s_id:
        session_matches = list(raw_dir.glob(f"{s_id}_*"))
        if session_matches:
            return session_matches[0]
    # 2. Match exact or prefixed with p_name
    if p_name:
        candidates = [
            raw_dir / f"{s_id}_{p_name}",
            raw_dir / p_name,
            raw_dir / "NovaCore_Multimodal_Company_Report_2026.pdf",
        ]
        for c in candidates:
            if c.exists():
                return c
    # 3. Fallback to NovaCore demo or any available PDF in data/raw
    default_demo = raw_dir / "NovaCore_Multimodal_Company_Report_2026.pdf"
    if default_demo.exists():
        return default_demo
    all_pdfs = list(raw_dir.glob("*.pdf"))
    return all_pdfs[0] if all_pdfs else None


pipeline = get_pipeline()

# Session State Initialization
if "active_session_id" not in st.session_state:
    sessions = load_all_sessions()
    if sessions:
        # Pick the most recently created session
        latest_id = list(sessions.keys())[-1]
        st.session_state.active_session_id = latest_id
    else:
        # Create initial default session
        new_id = create_new_session(
            title="Sample Report",
            pdf_name="Sample_Report.pdf",
            namespace="fy2026-demo",
        )
        st.session_state.active_session_id = new_id

# -------------------------------------------------------------
# SIDEBAR: Chat History & New Chat
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📑 Multimodal RAG")
    st.markdown("<p style='color: #6b7280; font-size: 0.85rem;'>Chat with Text, Tables, Charts & Diagrams</p>", unsafe_allow_html=True)

    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        new_id = create_new_session(title="New Conversation")
        st.session_state.active_session_id = new_id
        st.rerun()

    st.markdown("---")
    st.markdown("#### 💬 Chat History")

    all_sessions = load_all_sessions()
    
    if not all_sessions:
        st.caption("No conversations yet.")
    else:
        for s_id, s_data in reversed(list(all_sessions.items())):
            is_active = (s_id == st.session_state.active_session_id)
            title = s_data.get("title", "Untitled Chat")
            pdf_name = s_data.get("pdf_name")
            
            col_chat, col_del = st.columns([0.82, 0.18])
            with col_chat:
                btn_label = f"{'👉 ' if is_active else '📄 '}{title[:20]}"
                if st.button(
                    btn_label,
                    key=f"btn_chat_{s_id}",
                    use_container_width=True,
                    help=f"PDF: {pdf_name or 'None'}\nCreated: {s_data.get('created_at')}",
                ):
                    st.session_state.active_session_id = s_id
                    st.rerun()

            with col_del:
                if st.button("🗑️", key=f"del_{s_id}", help="Delete chat"):
                    delete_session(s_id)
                    remaining = load_all_sessions()
                    if remaining:
                        st.session_state.active_session_id = list(remaining.keys())[-1]
                    else:
                        st.session_state.active_session_id = create_new_session(title="New Conversation")
                    st.rerun()

    st.markdown("---")
    # System Status Indicator
    with st.expander("⚙️ System Status", expanded=False):
        st.caption("• **Vector DB**: Pinecone Serverless")
        st.caption("• **Hybrid Search**: Dense + BM25 + RRF")
        st.caption("• **Semantic Cache**: GPTCache (Active)")
        st.caption("• **Agentic Loop**: Self-RAG Active")
        st.caption("• **Text LLM**: OpenAI OSS 20B (Groq)")
        st.caption("• **Vision LLM**: Qwen 27B Vision (Groq)")
        ls_status = "Connected ✅" if pipeline.online_evaluator.is_active else "Local Mode ℹ️"
        st.caption(f"• **LangSmith**: {ls_status}")

    # Semantic Cache ROI Metrics
    if hasattr(pipeline, "cache_manager"):
        stats = pipeline.cache_manager.get_stats()
        with st.expander("⚡ Semantic Cache ROI", expanded=False):
            st.metric(label="Hit Rate", value=stats["Hit Rate"])
            st.caption(f"• **Total Inquiries**: {stats['Total Inquiries']}")
            st.caption(f"• **Tokens Saved**: {stats['Tokens Saved']}")
            st.caption(f"• **Latency Saved**: {stats['Latency Saved']}")
            st.caption(f"• **Cost Saved**: {stats['Estimated Cost Saved']}")

    st.markdown("---")
    # Multi-Document Cross-Analysis Selector
    st.markdown("#### 📑 Comparative Intelligence")
    multi_doc_mode = st.toggle("Cross-Document Compare Mode", value=False)
    selected_docs_for_compare = []
    if multi_doc_mode:
        all_indexed_docs = {}
        for s_id, s_info in load_all_sessions().items():
            if s_info.get("pdf_name"):
                all_indexed_docs[s_info["pdf_name"]] = s_info.get("namespace", f"chat-{s_id}")

        doc_options = list(all_indexed_docs.keys())
        if len(doc_options) >= 2:
            chosen = st.multiselect(
                "Select documents to compare:",
                options=doc_options,
                default=doc_options[:2],
            )
            selected_docs_for_compare = [{"name": name, "namespace": all_indexed_docs[name]} for name in chosen]
        else:
            st.info("Upload at least 2 PDFs across chats to unlock side-by-side comparative analysis.")


# -------------------------------------------------------------
# MAIN WINDOW
# -------------------------------------------------------------
current_sessions = load_all_sessions()
session_id = st.session_state.active_session_id

if session_id not in current_sessions:
    session_id = create_new_session(title="New Conversation")
    st.session_state.active_session_id = session_id
    current_sessions = load_all_sessions()

session = current_sessions[session_id]
pdf_name = session.get("pdf_name")
namespace = session.get("namespace", f"chat-{session_id}")
messages = session.get("messages", [])

st.markdown(f"<div class='main-title'>{session.get('title', 'New Chat')}</div>", unsafe_allow_html=True)

if pdf_name:
    st.markdown(f"<span class='doc-badge'>📄 Active Document: {pdf_name}</span>", unsafe_allow_html=True)
else:
    st.markdown("<p class='sub-title'>Upload a PDF document below to begin asking questions across its text, tables, and images.</p>", unsafe_allow_html=True)

# -------------------------------------------------------------
# UPLOAD SECTION (If no PDF attached or user wants to re-upload)
# -------------------------------------------------------------
if not pdf_name:
    st.info("📎 Please provide a document for this conversation:")
    
    col_upload, col_sample = st.columns([0.65, 0.35])
    
    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload any PDF Report, Paper, or Financial Deck",
            type=["pdf"],
            key=f"uploader_{session_id}",
        )
        if uploaded_file is not None:
            save_dir = PROJECT_ROOT / "data" / "raw"
            save_dir.mkdir(parents=True, exist_ok=True)
            saved_pdf_path = save_dir / f"{session_id}_{uploaded_file.name}"
            
            with open(saved_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            status_box = st.status(f"Ingesting '{uploaded_file.name}'...", expanded=True)
            
            def on_progress(pct: float, msg: str):
                status_box.write(f"• {msg}")

            try:
                doc_count = process_and_index_pdf(
                    pdf_path=saved_pdf_path,
                    namespace=namespace,
                    progress_callback=on_progress,
                )
                status_box.update(
                    label=f"Ingestion Complete! Indexed {doc_count} multimodal chunks.",
                    state="complete",
                    expanded=False,
                )
                session["pdf_name"] = uploaded_file.name
                clean_title = uploaded_file.name.replace(".pdf", "")[:28]
                session["title"] = clean_title
                save_session(session_id, session)
                st.success("Document ready for chat!")
                st.rerun()
            except Exception as e:
                status_box.update(label=f"Ingestion Failed: {e}", state="error")
                st.error(f"Error indexing PDF: {e}")

    with col_sample:
        st.markdown("**Or test instantly with a pre-indexed report:**")
        if st.button("⚡ Use Sample Demo Report", use_container_width=True):
            session["pdf_name"] = "Sample_Report.pdf"
            session["title"] = "Sample Report"
            session["namespace"] = "fy2026-demo"
            save_session(session_id, session)
            st.rerun()

# -------------------------------------------------------------
# CHAT DISPLAY & INPUT
# -------------------------------------------------------------
# Render conversation history
for msg_idx, msg in enumerate(messages):
    role = msg.get("role", "user")
    content = msg.get("content", "")
    with st.chat_message(role):
        st.markdown(content)
        
        # Display inline images if assistant retrieved visuals
        retrieved_images = msg.get("image_paths", [])
        if retrieved_images:
            st.markdown("**📸 Retrieved Evidence Visuals:**")
            cols = st.columns(min(len(retrieved_images), 3))
            for c_idx, img_p in enumerate(retrieved_images[:3]):
                with cols[c_idx]:
                    if Path(img_p).exists():
                        st.image(str(img_p), caption=f"Visual {c_idx+1}", use_container_width=True)

        # Display sources & Click-to-Highlight Grounding
        sources = msg.get("sources", [])
        if sources:
            with st.expander("📚 Retrieved Source Pages & Verified Grounding", expanded=False):
                active_pdf_path = resolve_session_pdf_path(session_id, pdf_name)
                for idx, s in enumerate(sources):
                    raw_p = s.get("page")
                    if raw_p is None:
                        continue
                    try:
                        p_num = int(float(raw_p))
                    except (ValueError, TypeError):
                        p_num = None

                    mod = str(s.get("modality", "")).upper()
                    doc_label = f"• **{s.get('doc_name')}** | " if s.get("doc_name") else ""
                    st.markdown(f"{doc_label}**Source #{idx+1}**: Page {p_num if p_num else raw_p} `[{mod}]`")
                    if s.get("snippet"):
                        st.caption(f"_{s.get('snippet')}..._")

                    if active_pdf_path and p_num:
                        with st.popover(
                            f"🔍 Audit Highlighted Page {p_num}",
                            key=f"hist_audit_{msg_idx}_{idx}_{p_num}_{session_id}",
                        ):
                            highlighted_bytes = highlight_page_passage(
                                pdf_path=active_pdf_path,
                                page_number=p_num,
                                search_snippet=s.get("snippet"),
                                bbox=s.get("bbox"),
                            )
                            if highlighted_bytes:
                                st.image(
                                    highlighted_bytes,
                                    caption=f"Verified Source: Page {p_num} (Highlight in Yellow)",
                                    use_container_width=True,
                                )
                            else:
                                st.info(f"Page {p_num} preview could not be generated.")

# User chat input
if pdf_name:
    user_input = st.chat_input("Ask anything about this document...")
    if user_input:
        # 1. Add and render user message
        messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # 2. Query Multimodal RAG pipeline (Streaming or Multi-Doc Mode)
        with st.chat_message("assistant"):
            # A. Multi-Document Comparative Mode
            if multi_doc_mode and len(selected_docs_for_compare) >= 2:
                with st.spinner("Executing parallel multi-namespace retrieval & cross-document synthesis..."):
                    doc_names = [d["name"] for d in selected_docs_for_compare]
                    doc_contexts = pipeline.multi_doc_comparator.parallel_retrieve(
                        doc_targets=selected_docs_for_compare,
                        query=user_input,
                        k=3,
                    )
                    answer_text = pipeline.multi_doc_comparator.comparative_synthesis(
                        question=user_input,
                        doc_contexts=doc_contexts,
                    )
                    st.markdown(answer_text)
                    routed_to = "multi_doc_comparative"
                    image_paths = []
                    sources_meta = []
                    for d_name, d_list in doc_contexts.items():
                        for doc in d_list:
                            sources_meta.append({
                                "doc_name": d_name,
                                "page": doc.metadata.get("page"),
                                "modality": doc.metadata.get("modality"),
                                "snippet": doc.page_content[:150],
                            })
                    with st.expander("📑 Cross-Document Retrieved Evidence Chunks", expanded=False):
                        for s in sources_meta:
                            st.caption(f"• **{s.get('doc_name')}** | Page {s.get('page')} | {str(s.get('modality', '')).upper()}")

            # B. Standard Single-Doc Mode with Real-Time Token Streaming & Safety Guardrails
            else:
                with st.spinner("Analyzing document across text, tables, and visuals..."):
                    stream_res = pipeline.stream_query(
                        question=user_input,
                        namespace=namespace,
                        chat_history=messages[:-1],
                        enable_cache=True,
                        enable_agentic_loop=True,
                    )

                # Render token-by-token stream live to UI with real-time guardrail interception
                stream_placeholder = st.empty()
                stream_generator = stream_res["stream_generator"]
                answer_text = stream_placeholder.write_stream(stream_generator)

                routed_to = stream_res.get("routed_to", "text")
                image_paths = stream_res.get("image_paths", [])
                retrieved_docs = stream_res.get("retrieved_docs", [])

                # Show retrieved images
                if image_paths:
                    st.markdown("**📸 Retrieved Evidence Visuals:**")
                    cols = st.columns(min(len(image_paths), 3))
                    for c_idx, img_p in enumerate(image_paths[:3]):
                        with cols[c_idx]:
                            if Path(img_p).exists():
                                st.image(str(img_p), caption=f"Visual {c_idx+1}", use_container_width=True)

                # Source citations & Click-to-Highlight Grounding
                sources_meta = []
                for d in retrieved_docs:
                    sources_meta.append({
                        "page": d.metadata.get("page"),
                        "modality": d.metadata.get("modality"),
                        "snippet": d.page_content[:150],
                        "bbox": d.metadata.get("bbox"),
                    })

                # Check if PDF path exists for visual grounding preview
                active_pdf_disk_path = resolve_session_pdf_path(session_id, pdf_name)

                with st.expander("📚 Retrieved Source Pages & Verified Grounding", expanded=False):
                    for idx, s in enumerate(sources_meta):
                        raw_p = s.get("page")
                        try:
                            p_num = int(float(raw_p)) if raw_p is not None else None
                        except (ValueError, TypeError):
                            p_num = None
                        mod = str(s.get("modality", "")).upper()
                        st.markdown(f"**Source #{idx+1}**: Page {p_num if p_num else raw_p} `[{mod}]`")
                        if s.get("snippet"):
                            st.caption(f"_{s.get('snippet', '')}..._")

                        # If source PDF is available on disk, allow 1-click highlighted inspection
                        if active_pdf_disk_path and p_num:
                            with st.popover(
                                f"🔍 Audit Highlighted Page {p_num}",
                                key=f"live_audit_{idx}_{p_num}_{session_id}",
                            ):
                                highlighted_bytes = highlight_page_passage(
                                    pdf_path=active_pdf_disk_path,
                                    page_number=p_num,
                                    search_snippet=s.get("snippet"),
                                    bbox=s.get("bbox"),
                                )
                                if highlighted_bytes:
                                    st.image(
                                        highlighted_bytes,
                                        caption=f"Verified Source: Page {p_num} (Highlight in Yellow)",
                                        use_container_width=True,
                                    )
                                else:
                                    st.info(f"Page {p_num} preview could not be generated.")
                    st.divider()

        # 3. Save assistant message and update storage
        messages.append({
            "role": "assistant",
            "content": answer_text,
            "routed_to": routed_to,
            "image_paths": image_paths,
            "sources": sources_meta,
        })
        session["messages"] = messages
        
        # Auto-update session title from first user question if it's the default name
        if session.get("title") in ("New Conversation", "New Chat"):
            session["title"] = user_input[:28] + ("..." if len(user_input) > 28 else "")

        save_session(session_id, session)
        st.rerun()
