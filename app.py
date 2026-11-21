"""
DocuMind AI - An Intelligent Document Summarization and Learning Assistant
Main Streamlit Application Entrypoint.
"""

import os
import io
import streamlit as st

# Configure Streamlit page settings must be the first Streamlit command
# Browser-tab icon (our logo)
_FAVICON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "favicon.png")
if not os.path.exists(_FAVICON):
    _FAVICON = "🧠"

st.set_page_config(
    page_title="DocuMind AI - Document Learning Assistant",
    page_icon=_FAVICON,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get help": None, "Report a bug": None, "About": "DocuMind AI - your document learning assistant."}
)

# Internal module imports
from document_processor import (
    process_uploaded_file,
    extract_from_txt,
    ProcessedDocument,
    DocumentMetadata,
    DocumentPage
)
from rag_engine import RAGEngine
from ai_engine import GeminiService, NOT_CONFIGURED_MESSAGE, is_admin_mode
from quiz_generator import render_interactive_quiz
from flashcard_generator import render_interactive_flashcards
from utils import (
    load_custom_css,
    init_session_state,
    reset_document_session,
    SAMPLE_DOCUMENTS,
    format_summary_for_download,
    format_keywords_for_download,
    format_eli5_for_download,
    logo_data_uri
)

# Initialize application styling and session state
load_custom_css()
init_session_state()

# Initialize AI service (credentials are read server-side only)
@st.cache_resource(show_spinner=False)
def get_ai_service() -> GeminiService:
    return GeminiService()


ai_service = get_ai_service()

NOT_CONFIGURED_MSG = NOT_CONFIGURED_MESSAGE
IS_ADMIN = is_admin_mode()


def load_sample_document(sample_name: str):
    """Loads a pre-packaged sample document into session state for instant testing."""
    sample_text = SAMPLE_DOCUMENTS.get(sample_name, "")
    if not sample_text:
        return

    reset_document_session()
    proc_doc = extract_from_txt(sample_text.encode("utf-8"), file_name=f"{sample_name}.txt")
    st.session_state["processed_doc"] = proc_doc

    # Build RAG index
    rag = RAGEngine()
    rag.index_document(proc_doc)
    st.session_state["rag_engine"] = rag


# ==========================================
# SIDEBAR NAVIGATION & STATUS
# ==========================================
with st.sidebar:
    st.markdown(
        f"""
        <div class="brand">
            <img src="{logo_data_uri()}" alt="DocuMind AI logo"/>
            <div>
                <div class="brand-name">DocuMind AI</div>
                <div class="brand-tag">Read less. Understand more.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")

    # Navigation Menu
    nav_options = [
        "🏠 Dashboard",
        "📤 Upload Document",
        "📝 Summarize",
        "🔍 Extract Keywords",
        "💬 Ask Your Document",
        "💡 Simplify Concepts",
        "🎯 Generate Quiz",
        "📇 Flashcards",
        "⚙️ Settings"
    ]

    # Stable-key navigation: a fixed widget key means ONE click always navigates.
    # (Changing `index` between reruns used to reset the widget and swallow every other click.)
    page_to_label = {opt.split(" ", 1)[-1]: opt for opt in nav_options}
    if st.session_state.get("nav_radio") not in nav_options:
        st.session_state["nav_radio"] = page_to_label.get(st.session_state["active_page"], nav_options[0])
    elif st.session_state.get("_nav_last") == st.session_state["nav_radio"]:
        # Radio untouched this run: if code (e.g. a dashboard button) changed the page, follow it.
        st.session_state["nav_radio"] = page_to_label.get(st.session_state["active_page"], st.session_state["nav_radio"])

    selected_nav = st.radio(
        "Navigation",
        options=nav_options,
        key="nav_radio",
        label_visibility="collapsed"
    )
    st.session_state["_nav_last"] = selected_nav

    # The user's click (if any) decides the page; otherwise the page set by code stands.
    if selected_nav != page_to_label.get(st.session_state["active_page"]) and \
            st.session_state.get("_nav_prev_selected") != selected_nav:
        st.session_state["active_page"] = selected_nav.split(" ", 1)[-1]
    st.session_state["_nav_prev_selected"] = selected_nav

    st.markdown("---")

    # Document Status Box
    st.markdown("### 📄 Document Status")
    doc: ProcessedDocument = st.session_state.get("processed_doc")
    if doc and doc.metadata and not doc.metadata.error_message:
        meta = doc.metadata
        st.success(f"**Loaded:** {meta.file_name}")
        st.caption(f"• **Type:** {meta.file_type.upper()}\n• **Pages:** {meta.total_pages}\n• **Words:** {meta.total_words:,}")
        if st.button("🔄 Unload Document", use_container_width=True):
            reset_document_session()
            st.rerun()
    else:
        st.info("No document loaded yet.")
        if st.button("➕ Upload or Pick Sample", use_container_width=True):
            st.session_state["active_page"] = "Upload Document"
            st.rerun()

    st.markdown("---")

    # AI assistant status (plain language for end users)
    st.markdown("### ✨ AI Assistant")
    if ai_service.is_configured():
        st.markdown("<span class='badge badge-green'>🟢 Ready</span>", unsafe_allow_html=True)
        if IS_ADMIN:
            st.caption(f"Admin: model `{ai_service.model}`")
    else:
        st.markdown("<span class='badge badge-amber'>🟡 Temporarily unavailable</span>", unsafe_allow_html=True)
        st.caption("Please try again later." if not IS_ADMIN else "Admin: API key not configured (see Settings).")

with st.sidebar:
    st.markdown("<div class='sidebar-foot'>DocuMind AI · Your documents stay in your session</div>", unsafe_allow_html=True)


# Helper to enforce document presence
def require_document() -> bool:
    doc = st.session_state.get("processed_doc")
    if not doc or not doc.full_text.strip():
        st.warning("⚠️ Please upload a document or select a sample document first to use this feature.")
        col_act1, col_act2 = st.columns([1, 2])
        with col_act1:
            if st.button("📤 Go to Upload Document", type="primary"):
                st.session_state["active_page"] = "Upload Document"
                st.rerun()
        with col_act2:
            if st.button("⚡ Quick-Load Sample Document ('AI in Healthcare')"):
                load_sample_document("Artificial Intelligence in Healthcare")
                st.success("Sample document loaded successfully!")
                st.rerun()
        return False
    return True


# ==========================================
# PAGE ROUTING
# ==========================================
active_page = st.session_state["active_page"]


# ------------------------------------------
# 1. DASHBOARD
# ------------------------------------------
if active_page == "Dashboard":
    st.markdown(
        f"""
        <div class="hero">
            <img src="{logo_data_uri()}" alt="DocuMind AI logo"/>
            <div class="hero-body">
                <h1>Welcome to DocuMind AI</h1>
                <p>Turn any PDF or text file into clear summaries, answers with page citations, quizzes and flashcards.</p>
                <span class="chip">📄 PDF &amp; TXT</span>
                <span class="chip">🌐 English · हिन्दी · ਪੰਜਾਬੀ</span>
                <span class="chip">🔎 Answers with citations</span>
                <span class="chip">🎯 Quizzes &amp; flashcards</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Quick Start Banner
    st.markdown("### 🚀 Quick Start Guide")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">Step 1</div>
                <h4 style="margin: 8px 0 4px 0;">Upload or Pick Sample</h4>
                <p style="color: #6b7280; font-size: 0.9rem;">Upload any PDF or TXT file, or launch an instant 1-click sample document below.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">Step 2</div>
                <h4 style="margin: 8px 0 4px 0;">Summarize & Ask</h4>
                <p style="color: #6b7280; font-size: 0.9rem;">Generate multilingual summaries, extract key facts, and chat directly with document citations.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">Step 3</div>
                <h4 style="margin: 8px 0 4px 0;">Master the Material</h4>
                <p style="color: #6b7280; font-size: 0.9rem;">Simplify dense paragraphs, take interactive scored quizzes, and review study flashcards.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 1-Click Instant Demo Section
    st.markdown("### ⚡ Try Instant Demo (No File Needed)")
    st.caption("Don't have a document handy? Click either button to load an expertly curated sample document instantly:")
    col_demo1, col_demo2 = st.columns(2)
    with col_demo1:
        if st.button("📄 Load 'AI in Modern Healthcare' Sample", use_container_width=True):
            load_sample_document("Artificial Intelligence in Healthcare")
            st.success("Loaded 'AI in Modern Healthcare'! You can now explore Summaries, Q&A, and Quizzes.")
            st.rerun()
    with col_demo2:
        if st.button("🌱 Load 'Renewable Energy Systems' Sample", use_container_width=True):
            load_sample_document("Renewable Energy & Climate Solutions")
            st.success("Loaded 'Renewable Energy Systems'! You can now explore Summaries, Q&A, and Quizzes.")
            st.rerun()

    st.markdown("---")

    # Feature Grid
    st.markdown("### 🌟 Powerful Features at Your Fingertips")
    row1_c1, row1_c2, row1_c3 = st.columns(3)

    with row1_c1:
        st.markdown(
            """
            <div class="feature-card">
                <div>
                    <h3>📝 Multilingual Summaries</h3>
                    <p>Generate Short, Medium, Detailed, Bulleted, or Section-wise summaries in English, Hindi, or Punjabi.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Open Summarizer ➡️", key="btn_dash_sum", use_container_width=True):
            st.session_state["active_page"] = "Summarize"
            st.rerun()

    with row1_c2:
        st.markdown(
            """
            <div class="feature-card">
                <div>
                    <h3>💬 Grounded Document Q&A</h3>
                    <p>Ask questions with zero hallucination. Retrieval-Augmented Generation (RAG) grounds answers with page citations.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Open Document Chat ➡️", key="btn_dash_qa", use_container_width=True):
            st.session_state["active_page"] = "Ask Your Document"
            st.rerun()

    with row1_c3:
        st.markdown(
            """
            <div class="feature-card">
                <div>
                    <h3>🔍 Keyword & Fact Extraction</h3>
                    <p>Automatically extract key concepts, formal definitions, dates, figures, and technical terminology.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Extract Information ➡️", key="btn_dash_kw", use_container_width=True):
            st.session_state["active_page"] = "Extract Keywords"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    row2_c1, row2_c2, row2_c3 = st.columns(3)

    with row2_c1:
        st.markdown(
            """
            <div class="feature-card">
                <div>
                    <h3>💡 Simplify Concepts (ELI5)</h3>
                    <p>Demystify difficult paragraphs with plain English, practical real-world examples, and memorable analogies.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Simplify a Concept ➡️", key="btn_dash_eli5", use_container_width=True):
            st.session_state["active_page"] = "Simplify Concepts"
            st.rerun()

    with row2_c2:
        st.markdown(
            """
            <div class="feature-card">
                <div>
                    <h3>🎯 Scored Quiz Generator</h3>
                    <p>Test your comprehension with auto-generated multiple-choice questions, instant scoring, and explanations.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Take a Quiz ➡️", key="btn_dash_quiz", use_container_width=True):
            st.session_state["active_page"] = "Generate Quiz"
            st.rerun()

    with row2_c3:
        st.markdown(
            """
            <div class="feature-card">
                <div>
                    <h3>📇 Interactive Flashcards</h3>
                    <p>Active recall study cards with flip-to-reveal answers, deck navigation, and export capability.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("View Flashcards ➡️", key="btn_dash_fc", use_container_width=True):
            st.session_state["active_page"] = "Flashcards"
            st.rerun()


# ------------------------------------------
# 2. UPLOAD DOCUMENT
# ------------------------------------------
elif active_page == "Upload Document":
    st.markdown("## 📤 Upload & Process Document")
    st.caption("Upload your PDF or Text document. DocuMind AI will extract text, analyze structure, and build the search index.")

    tab_upload, tab_sample = st.tabs(["📁 Upload File (PDF / TXT)", "⚡ Choose Sample Document"])

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Choose a PDF or TXT file",
            type=["pdf", "txt"],
            help="DocuMind AI processes digital text PDFs and TXT files up to 50MB."
        )

        if uploaded_file is not None:
            # Only process a given upload once (Streamlit reruns this script on every click)
            upload_sig = f"{uploaded_file.name}:{uploaded_file.size}:{getattr(uploaded_file, 'file_id', '')}"
            if st.session_state.get("_upload_sig") != upload_sig:
                reset_document_session()
                progress = st.progress(0.0, text="Reading document...")

                def _on_extract(done_pages: int, total_pages: int):
                    progress.progress(
                        min(0.9, 0.9 * done_pages / max(total_pages, 1)),
                        text=f"Extracting text: page {done_pages} of {total_pages}..."
                    )

                proc_doc = process_uploaded_file(uploaded_file, progress_callback=_on_extract)
                st.session_state["processed_doc"] = proc_doc

                # Index for RAG if text extraction succeeded
                if proc_doc.full_text.strip():
                    progress.progress(0.95, text="Building search index...")
                    rag = RAGEngine()
                    rag.index_document(proc_doc)
                    st.session_state["rag_engine"] = rag
                progress.progress(1.0, text="Document ready.")
                progress.empty()
                st.session_state["_upload_sig"] = upload_sig

    with tab_sample:
        st.markdown("#### Select a pre-loaded sample document to test immediately:")
        sample_choice = st.selectbox("Sample Documents", list(SAMPLE_DOCUMENTS.keys()))
        if st.button("Load Selected Sample", type="primary"):
            load_sample_document(sample_choice)
            st.success(f"Loaded '{sample_choice}' successfully!")
            st.rerun()

    # Display Document Details & Extracted Content if available
    doc = st.session_state.get("processed_doc")
    if doc:
        st.markdown("---")
        meta = doc.metadata

        # Error handling
        if meta.error_message:
            st.error(f"❌ {meta.error_message}")
        else:
            # Scanned / Image-only warning
            if meta.is_scanned_or_image_only:
                st.warning(meta.warning_message)

            st.markdown("### 📊 Document Overview")
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("File Name", meta.file_name)
            col_m2.metric("File Type", meta.file_type.upper())
            col_m3.metric("Total Pages", meta.total_pages)
            col_m4.metric("Word Count", f"{meta.total_words:,}")

            st.markdown("### 📖 Extracted Content Preview")
            tab_preview_full, tab_preview_pages = st.tabs(["Full Document", "Page-by-Page View"])

            with tab_preview_full:
                st.text_area(
                    "Extracted Text",
                    value=doc.full_text,
                    height=300,
                    disabled=True,
                    help="Extracted plain text from your document."
                )

            with tab_preview_pages:
                if doc.pages:
                    page_options = [f"Page {p.page_number} ({p.word_count} words)" for p in doc.pages]
                    selected_page_str = st.selectbox("Select Page", page_options)
                    selected_idx = page_options.index(selected_page_str)
                    st.markdown(f"**Content of Page {selected_idx + 1}:**")
                    st.text_area(
                        "Page Content",
                        value=doc.pages[selected_idx].text,
                        height=250,
                        disabled=True
                    )
                else:
                    st.info("Single page text document.")

            # Download Raw Text & Next Steps
            st.markdown("---")
            col_dl, col_next1, col_next2 = st.columns([1, 1, 1])

            with col_dl:
                st.download_button(
                    label="📥 Download Extracted Text (.txt)",
                    data=doc.full_text,
                    file_name=f"{meta.file_name}_extracted.txt",
                    mime="text/plain",
                    use_container_width=True
                )

            with col_next1:
                if st.button("📝 Summarize Document ➡️", type="primary", use_container_width=True):
                    st.session_state["active_page"] = "Summarize"
                    st.rerun()

            with col_next2:
                if st.button("💬 Ask Questions ➡️", use_container_width=True):
                    st.session_state["active_page"] = "Ask Your Document"
                    st.rerun()


# ------------------------------------------
# 3. SUMMARIZE
# ------------------------------------------
elif active_page == "Summarize":
    st.markdown("## 📝 AI Document Summarization")
    st.caption("Generate high-fidelity summaries customized by length, format, and language.")

    if require_document():
        doc = st.session_state["processed_doc"]

        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            summary_type = st.selectbox(
                "Summary Format & Length",
                options=["Short", "Medium", "Detailed", "Bullet-point", "Section-wise"],
                index=1,
                help="Short: 1-2 paragraphs | Medium: 3-4 paragraphs | Detailed: In-depth breakdown | Bullet-point: Key takeaways | Section-wise: Chapter/heading breakdown."
            )
        with col_opt2:
            summary_lang = st.selectbox(
                "Output Language",
                options=["English", "Hindi", "Punjabi"],
                index=0,
                help="Select between English, Hindi (हिंदी), and Punjabi (ਪੰਜਾਬੀ)."
            )

        if st.button("✨ Generate Summary", type="primary", use_container_width=True):
            if not ai_service.is_configured():
                st.warning(NOT_CONFIGURED_MSG)
            else:
                progress_bar = st.progress(0.0, text=f"Preparing your {summary_type.lower()} summary in {summary_lang}...")

                def _on_progress(step: int, total: int, message: str):
                    progress_bar.progress(min(1.0, step / max(total, 1)), text=message)

                with st.spinner(f"Writing your {summary_type.lower()} summary in {summary_lang}..."):
                    summary_result = ai_service.generate_summary(
                        document_text=doc.full_text,
                        summary_type=summary_type,
                        language=summary_lang,
                        temperature=st.session_state.get("temperature", 0.2),
                        progress_callback=_on_progress
                    )
                progress_bar.empty()
                if summary_result.startswith(("⚠️", "⏳", "❌", "🌐", "🚫")):
                    st.error(summary_result)  # friendly message; keep any previous summary intact
                else:
                    st.session_state["summary_text"] = summary_result
                    st.session_state["summary_type"] = summary_type
                    st.session_state["summary_lang"] = summary_lang

        # Display Summary
        saved_summary = st.session_state.get("summary_text")
        if saved_summary:
            st.markdown("---")
            st.markdown(f"### 📋 Summary Result ({st.session_state.get('summary_type')} • {st.session_state.get('summary_lang')})")

            # Metrics
            orig_words = doc.metadata.total_words
            sum_words = len(saved_summary.split())
            reduction = max(0, int((1 - (sum_words / orig_words)) * 100)) if orig_words > 0 else 0

            col_met1, col_met2, col_met3 = st.columns(3)
            col_met1.metric("Original Words", f"{orig_words:,}")
            col_met2.metric("Summary Words", f"{sum_words:,}")
            col_met3.metric("Reading Time Saved", f"{reduction}%")

            # Styled container for summary
            with st.container(border=True):
                st.markdown(saved_summary)

            # Download button
            st.markdown("<br>", unsafe_allow_html=True)
            export_content = format_summary_for_download(
                doc.metadata.file_name,
                st.session_state.get("summary_type", "Summary"),
                st.session_state.get("summary_lang", "English"),
                saved_summary
            )
            st.download_button(
                label="📥 Download Summary (.txt)",
                data=export_content,
                file_name=f"DocuMind_Summary_{doc.metadata.file_name}.txt",
                mime="text/plain",
                use_container_width=True
            )


# ------------------------------------------
# 4. EXTRACT KEYWORDS & INFORMATION
# ------------------------------------------
elif active_page == "Extract Keywords":
    st.markdown("## 🔍 Keyword & Key Information Extraction")
    st.caption("Identify core themes, formal definitions, facts, dates, names, and technical terms directly from the text.")

    if require_document():
        doc = st.session_state["processed_doc"]

        if st.button("🔎 Extract Important Information", type="primary", use_container_width=True):
            if not ai_service.is_configured():
                st.warning(NOT_CONFIGURED_MSG)
            else:
                with st.spinner("Analyzing document and extracting key entities (this takes a few seconds)..."):
                    keywords_result = ai_service.extract_keywords_and_facts(
                        document_text=doc.full_text,
                        temperature=0.1
                    )
                if keywords_result.startswith(("⚠️", "⏳", "❌", "🌐", "🚫")):
                    st.error(keywords_result)
                else:
                    st.session_state["keywords_text"] = keywords_result

        saved_keywords = st.session_state.get("keywords_text")
        if saved_keywords:
            st.markdown("---")
            st.markdown("### 📌 Extracted Information")
            with st.container(border=True):
                st.markdown(saved_keywords)

            # Download button
            st.markdown("<br>", unsafe_allow_html=True)
            export_kw = format_keywords_for_download(doc.metadata.file_name, saved_keywords)
            st.download_button(
                label="📥 Download Extracted Information (.txt)",
                data=export_kw,
                file_name=f"DocuMind_Keywords_{doc.metadata.file_name}.txt",
                mime="text/plain",
                use_container_width=True
            )


# ------------------------------------------
# 5. ASK YOUR DOCUMENT (RAG CHATBOT)
# ------------------------------------------
elif active_page == "Ask Your Document":
    st.markdown("## 💬 Ask Your Document (Grounded RAG Chatbot)")
    st.caption("Ask questions about your uploaded document. Answers are strictly grounded in document excerpts with page citations.")

    if require_document():
        doc = st.session_state["processed_doc"]
        rag: RAGEngine = st.session_state.get("rag_engine")

        if not rag or not rag.is_indexed():
            with st.spinner("Indexing document chunks for fast retrieval..."):
                rag = RAGEngine()
                rag.index_document(doc)
                st.session_state["rag_engine"] = rag

        # Suggested Questions
        with st.expander("💡 Click a Suggested Question to Ask Immediately", expanded=False):
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                if st.button("What is the primary topic or purpose of this document?", use_container_width=True):
                    st.session_state["pending_question"] = "What is the primary topic or purpose of this document?"
            with col_s2:
                if st.button("What are the most significant conclusions or recommendations?", use_container_width=True):
                    st.session_state["pending_question"] = "What are the most significant conclusions or recommendations?"

        # Display Chat History
        chat_container = st.container()
        with chat_container:
            for message in st.session_state["chat_history"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    # If this message has sources attached, show expandable accordion
                    if message.get("sources"):
                        with st.expander("🔍 View Supporting Document Excerpts & Pages", expanded=False):
                            for idx, src in enumerate(message["sources"], start=1):
                                st.markdown(f"**Excerpt {idx} (Page {src['page']}, Relevance: {src['score']}%)**")
                                st.caption(src["text"])
                                st.markdown("---")

        # Chat Input
        pending = st.session_state.pop("pending_question", None)
        user_query = st.chat_input("Ask any question about your document...") or pending

        if user_query:
            if not ai_service.is_configured():
                st.warning(NOT_CONFIGURED_MSG)
            else:
                # Add user query to chat history
                st.session_state["chat_history"].append({"role": "user", "content": user_query})

                with st.chat_message("user"):
                    st.markdown(user_query)

                # Retrieve relevant chunks via RAG
                retrieved_chunks = rag.retrieve(user_query, top_k=4)
                context_str = rag.format_context_for_prompt(retrieved_chunks)

                with st.chat_message("assistant"):
                    with st.spinner("Searching document excerpts and synthesizing grounded answer..."):
                        answer = ai_service.answer_document_question(
                            question=user_query,
                            context_string=context_str,
                            chat_history=st.session_state["chat_history"],
                            temperature=st.session_state.get("temperature", 0.2)
                        )
                        st.markdown(answer)

                        # Package sources for display and storage
                        sources = []
                        for chunk, score in retrieved_chunks:
                            sources.append({
                                "page": chunk.page_number,
                                "score": max(0, min(100, int(score * 100))),
                                "text": chunk.text
                            })

                        if sources:
                            with st.expander("🔍 View Supporting Document Excerpts & Pages", expanded=False):
                                for idx, src in enumerate(sources, start=1):
                                    st.markdown(f"**Excerpt {idx} (Page {src['page']}, Relevance: {src['score']}%)**")
                                    st.caption(src["text"])
                                    st.markdown("---")

                # Append assistant response to state
                st.session_state["chat_history"].append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

        # Clear Chat History Button
        if st.session_state["chat_history"]:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑️ Clear Chat History", type="secondary"):
                st.session_state["chat_history"] = []
                st.rerun()


# ------------------------------------------
# 6. SIMPLIFY CONCEPTS ("ELI5")
# ------------------------------------------
elif active_page == "Simplify Concepts":
    st.markdown("## 💡 Explain Like I'm a Beginner (ELI5)")
    st.caption("Stuck on a dense paragraph, technical term, or confusing formula? Get a crystal-clear explanation with real-world analogies.")

    if require_document():
        doc = st.session_state["processed_doc"]

        # Option to paste text or select an excerpt
        st.markdown("#### Enter or paste the difficult paragraph, concept, or term:")
        concept_input = st.text_area(
            "Concept or Excerpt to Explain",
            value=st.session_state.get("eli5_concept", ""),
            height=130,
            placeholder="e.g., Convolutional Neural Networks, Levelized Cost of Energy, Algorithmic Bias..."
        )

        # Quick prefill dropdown from document sections
        if st.button("✨ Explain in Simple English", type="primary", use_container_width=True):
            if not concept_input.strip():
                st.warning("Please enter a concept or paragraph to explain.")
            elif not ai_service.is_configured():
                st.warning(NOT_CONFIGURED_MSG)
            else:
                st.session_state["eli5_concept"] = concept_input
                with st.spinner("Breaking down concept into plain English, examples, and analogies..."):
                    explanation = ai_service.explain_like_beginner(
                        concept_or_excerpt=concept_input,
                        document_context=doc.full_text[:3000],
                        temperature=0.3
                    )
                if explanation.startswith(("⚠️", "⏳", "❌", "🌐", "🚫")):
                    st.error(explanation)
                else:
                    st.session_state["eli5_result"] = explanation

        saved_eli5 = st.session_state.get("eli5_result")
        if saved_eli5:
            st.markdown("---")
            st.markdown(f"### 🌟 Friendly Explanation for: *\"{st.session_state.get('eli5_concept')}\"*")
            with st.container(border=True):
                st.markdown(saved_eli5)

            # Download button
            st.markdown("<br>", unsafe_allow_html=True)
            export_eli5 = format_eli5_for_download(
                doc.metadata.file_name,
                st.session_state.get("eli5_concept", "Concept"),
                saved_eli5
            )
            st.download_button(
                label="📥 Download Explanation (.txt)",
                data=export_eli5,
                file_name="DocuMind_Concept_Explanation.txt",
                mime="text/plain",
                use_container_width=True
            )


# ------------------------------------------
# 7. GENERATE QUIZ
# ------------------------------------------
elif active_page == "Generate Quiz":
    st.markdown("## 🎯 Interactive Scored Quiz Generator")
    st.caption("Generate custom multiple-choice quizzes to test your comprehension with immediate scoring and feedback.")

    if require_document():
        doc = st.session_state["processed_doc"]

        col_q1, col_q2 = st.columns([1, 2])
        with col_q1:
            num_q = st.select_slider(
                "Number of Questions",
                options=[3, 5, 8, 10],
                value=5
            )
        with col_q2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            gen_quiz_btn = st.button("🎲 Generate New Quiz", type="primary", use_container_width=True)

        if gen_quiz_btn:
            if not ai_service.is_configured():
                st.warning(NOT_CONFIGURED_MSG)
            else:
                with st.spinner(f"Crafting {num_q} comprehensive multiple-choice questions from the document..."):
                    quiz_items, err = ai_service.generate_quiz(doc.full_text, num_questions=num_q)
                    if err:
                        st.error(f"❌ {err}")
                    else:
                        st.session_state["quiz_data"] = quiz_items
                        st.session_state["quiz_user_answers"] = {}
                        st.session_state["quiz_submitted"] = False
                        st.success(f"Generated {len(quiz_items)} quiz questions successfully!")
                        st.rerun()

        # Render Interactive Quiz Component
        quiz_data = st.session_state.get("quiz_data", [])
        if quiz_data:
            st.markdown("---")
            render_interactive_quiz(quiz_data)


# ------------------------------------------
# 8. FLASHCARDS
# ------------------------------------------
elif active_page == "Flashcards":
    st.markdown("## 📇 Interactive Study Flashcards")
    st.caption("Strengthen your memory and active recall with digital flip-to-reveal flashcards.")

    if require_document():
        doc = st.session_state["processed_doc"]

        col_f1, col_f2 = st.columns([1, 2])
        with col_f1:
            num_cards = st.select_slider(
                "Number of Flashcards",
                options=[5, 8, 10, 15],
                value=8
            )
        with col_f2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            gen_fc_btn = st.button("✨ Generate Flashcard Deck", type="primary", use_container_width=True)

        if gen_fc_btn:
            if not ai_service.is_configured():
                st.warning(NOT_CONFIGURED_MSG)
            else:
                with st.spinner(f"Creating {num_cards} study flashcards from key document concepts..."):
                    cards, err = ai_service.generate_flashcards(doc.full_text, num_cards=num_cards)
                    if err:
                        st.error(f"❌ {err}")
                    else:
                        st.session_state["flashcards_data"] = cards
                        st.session_state["flashcard_index"] = 0
                        st.session_state["flashcard_show_answer"] = False
                        st.success(f"Generated {len(cards)} flashcards successfully!")
                        st.rerun()

        # Render Interactive Flashcard Viewer Component
        flashcards_data = st.session_state.get("flashcards_data", [])
        if flashcards_data:
            st.markdown("---")
            render_interactive_flashcards(flashcards_data)


# ------------------------------------------
# 9. SETTINGS
# ------------------------------------------
elif active_page == "Settings":
    st.markdown("## ⚙️ Settings")
    st.caption("Personalize how DocuMind AI responds and manage your session.")

    st.markdown("### 🎛️ Answer Style")
    temp_val = st.slider(
        "Precise  ↔  Creative",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.get("temperature", 0.2)),
        step=0.05,
        help="Keep this low (0.0 - 0.3) for accurate, document-faithful answers. Raise it for more varied wording."
    )
    st.session_state["temperature"] = temp_val

    st.markdown("---")

    st.markdown("### 🔄 Session")
    doc_loaded = bool(st.session_state.get("processed_doc"))
    st.caption("Your document and results exist only in this browser session." if doc_loaded else "No document is loaded.")
    if st.button("🗑️ Clear Document & Results", type="secondary"):
        reset_document_session()
        st.success("Your document, chat history, and generated results have been cleared.")
        st.rerun()

    # Developer diagnostics: hidden from normal users (enable with DOCUMIND_ADMIN = true in secrets)
    if IS_ADMIN:
        st.markdown("---")
        with st.expander("🛠️ Administrator diagnostics", expanded=not ai_service.is_configured()):
            c1, c2 = st.columns(2)
            c1.success("API key detected") if ai_service.is_configured() else c1.warning("No API key detected")
            c2.info(f"Model: `{ai_service.model}`")
            if st.button("🧪 Test AI connection"):
                if not ai_service.is_configured():
                    st.error("No API key configured. Add GEMINI_API_KEY to .streamlit/secrets.toml (local) "
                             "or the app's Secrets panel (hosted), then restart.")
                else:
                    with st.spinner("Testing connection (retries automatically if busy)..."):
                        ok, msg = ai_service.test_connection()
                    (st.success if ok else st.error)(msg)
            st.caption("Key and model are configured server-side via GEMINI_API_KEY / GEMINI_MODEL and are never shown here.")
