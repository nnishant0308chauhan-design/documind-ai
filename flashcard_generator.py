"""
Flashcard Generator and Interactive Viewer Module for DocuMind AI
Manages study flashcards, single-card flip navigation, progress tracking,
and flashcard export.
"""

from typing import List, Dict, Any
import streamlit as st


def render_interactive_flashcards(flashcards: List[Dict[str, Any]]) -> None:
    """
    Renders an interactive flashcard deck in Streamlit.
    Features:
    - One card at a time with category badge.
    - Reveal / hide answer toggle.
    - Previous and Next navigation buttons.
    - Progress bar.
    - Toggle to view all cards in a study grid.
    - Export flashcards button.
    """
    if not flashcards:
        st.info("No flashcards available. Click 'Generate Flashcards' above to create study cards!")
        return

    num_cards = len(flashcards)

    # Initialize session state for navigation
    if "flashcard_index" not in st.session_state:
        st.session_state["flashcard_index"] = 0
    if "flashcard_show_answer" not in st.session_state:
        st.session_state["flashcard_show_answer"] = False

    # Ensure index is within bounds
    if st.session_state["flashcard_index"] >= num_cards:
        st.session_state["flashcard_index"] = 0

    current_idx = st.session_state["flashcard_index"]
    current_card = flashcards[current_idx]
    is_answer_revealed = st.session_state["flashcard_show_answer"]

    # Header controls
    col_hdr1, col_hdr2 = st.columns([3, 1])
    with col_hdr1:
        st.markdown(f"### 📇 Flashcard {current_idx + 1} of {num_cards}")
    with col_hdr2:
        view_mode = st.toggle("📋 Grid View", value=False, key="flashcard_grid_view_toggle")

    # Progress bar
    progress = (current_idx + 1) / num_cards
    st.progress(progress)

    if view_mode:
        # Show all cards in a grid
        st.markdown("#### 📚 Full Flashcard Deck Overview")
        for idx, card in enumerate(flashcards, start=1):
            category = card.get("category", "General")
            question = card.get("question", "")
            answer = card.get("answer", "")

            with st.expander(f"Card {idx}: [{category}] {question}", expanded=False):
                st.markdown(f"**Answer:** {answer}")
    else:
        # Single card interactive view
        category = current_card.get("category", "Core Concept")
        question = current_card.get("question", "")
        answer = current_card.get("answer", "")

        card_bg = "#f8f9fa" if not is_answer_revealed else "#eef6fc"
        border_color = "#3b82f6" if not is_answer_revealed else "#10b981"
        status_text = "QUESTION (Click 'Reveal Answer' below)" if not is_answer_revealed else "ANSWER"

        # Render styled card container
        st.markdown(
            f"""
            <div style="
                background-color: {card_bg};
                border: 2px solid {border_color};
                border-radius: 14px;
                padding: 30px;
                margin: 20px 0;
                box-shadow: 0 4px 12px rgba(0,0,0,0.06);
                min-height: 200px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            ">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                    <span style="
                        background-color: #e0e7ff;
                        color: #3730a3;
                        font-size: 0.82rem;
                        font-weight: 600;
                        padding: 4px 12px;
                        border-radius: 20px;
                        text-transform: uppercase;
                        letter-spacing: 0.5px;
                    ">{category}</span>
                    <span style="
                        font-size: 0.8rem;
                        font-weight: bold;
                        color: {'#6b7280' if not is_answer_revealed else '#059669'};
                    ">{status_text}</span>
                </div>
                <div style="font-size: 1.25rem; font-weight: 500; color: #111827; line-height: 1.6; padding: 10px 0;">
                    {question if not is_answer_revealed else answer}
                </div>
                <div style="font-size: 0.85rem; color: #9ca3af; text-align: right; margin-top: 15px;">
                    DocuMind AI Flashcards
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Interaction buttons
        col_prev, col_flip, col_next = st.columns([1, 2, 1])

        with col_prev:
            if st.button("⬅️ Previous", disabled=(current_idx == 0), use_container_width=True):
                st.session_state["flashcard_index"] -= 1
                st.session_state["flashcard_show_answer"] = False
                st.rerun()

        with col_flip:
            btn_label = "👁️ Reveal Answer" if not is_answer_revealed else "🙈 Hide Answer (Show Question)"
            if st.button(btn_label, type="primary", use_container_width=True):
                st.session_state["flashcard_show_answer"] = not is_answer_revealed
                st.rerun()

        with col_next:
            if st.button("Next ➡️", disabled=(current_idx == num_cards - 1), use_container_width=True):
                st.session_state["flashcard_index"] += 1
                st.session_state["flashcard_show_answer"] = False
                st.rerun()

    # Download button
    st.markdown("---")
    export_text = format_flashcards_to_text(flashcards)
    st.download_button(
        label="📥 Download All Flashcards (.txt)",
        data=export_text,
        file_name="DocuMind_Flashcards.txt",
        mime="text/plain",
        use_container_width=True
    )


def format_flashcards_to_text(flashcards: List[Dict[str, Any]]) -> str:
    """Formats all flashcards into a clean printable text format."""
    lines = [
        "=" * 60,
        "DocuMind AI - Study Flashcard Deck",
        "=" * 60,
        ""
    ]

    for idx, card in enumerate(flashcards, start=1):
        category = card.get("category", "General")
        question = card.get("question", "")
        answer = card.get("answer", "")

        lines.append(f"CARD #{idx} [{category.upper()}]")
        lines.append(f"FRONT (Question):")
        lines.append(f"  {question}")
        lines.append(f"BACK (Answer):")
        lines.append(f"  {answer}")
        lines.append("-" * 60)
        lines.append("")

    return "\n".join(lines)
