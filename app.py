import os
import sys
import shutil
from pathlib import Path
from typing import Dict, Any, List

import streamlit as st
from PIL import Image
from dotenv import load_dotenv

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_pipeline import MultimodalRAGPipeline
from src.pdf_processor import process_and_index_pdf
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
            title="NovaCore FY2026 Report",
            pdf_name="NovaCore_Multimodal_Company_Report_2026.pdf",
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
        st.caption("• **Text LLM**: OpenAI OSS 20B (Groq)")
        st.caption("• **Vision LLM**: Qwen 27B Vision (Groq)")
        ls_status = "Connected ✅" if pipeline.online_evaluator.is_active else "Local Mode ℹ️"
        st.caption(f"• **LangSmith**: {ls_status}")


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
        if st.button("⚡ Use NovaCore FY2026 Sample Report", use_container_width=True):
            session["pdf_name"] = "NovaCore_Multimodal_Company_Report_2026.pdf"
            session["title"] = "NovaCore Report 2026"
            session["namespace"] = "fy2026-demo"
            save_session(session_id, session)
            st.rerun()

# -------------------------------------------------------------
# CHAT DISPLAY & INPUT
# -------------------------------------------------------------
# Render conversation history
for msg in messages:
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

        # Display sources
        sources = msg.get("sources", [])
        if sources:
            with st.expander("📚 Retrieved Source Pages", expanded=False):
                for s in sources:
                    st.caption(f"• Page {s.get('page')} | {str(s.get('modality', '')).upper()}")

# User chat input
if pdf_name:
    user_input = st.chat_input("Ask anything about this document...")
    if user_input:
        # 1. Add and render user message
        messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # 2. Query Multimodal RAG pipeline
        with st.chat_message("assistant"):
            with st.spinner("Analyzing document across text, tables, and visuals..."):
                query_result = pipeline.query(
                    question=user_input,
                    namespace=namespace,
                    chat_history=messages[:-1],  # pass conversation memory
                    online_eval=True,
                )

                answer_text = query_result["answer"]
                routed_to = query_result["routed_to"]
                image_paths = query_result.get("image_paths", [])
                retrieved_docs = query_result.get("retrieved_docs", [])

                st.markdown(answer_text)

                # Show retrieved images
                if image_paths:
                    st.markdown("**📸 Retrieved Evidence Visuals:**")
                    cols = st.columns(min(len(image_paths), 3))
                    for c_idx, img_p in enumerate(image_paths[:3]):
                        with cols[c_idx]:
                            if Path(img_p).exists():
                                st.image(str(img_p), caption=f"Visual {c_idx+1}", use_container_width=True)

                # Source citations
                sources_meta = []
                for d in retrieved_docs:
                    sources_meta.append({
                        "page": d.metadata.get("page"),
                        "modality": d.metadata.get("modality"),
                    })

                with st.expander("📚 Retrieved Source Pages", expanded=False):
                    for s in sources_meta:
                        st.caption(f"• Page {s.get('page')} | {str(s.get('modality', '')).upper()}")

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
