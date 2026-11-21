"""
Quiz Generator and Interactive Quiz Player Module for DocuMind AI
Manages multiple-choice quiz generation, stateful answer tracking,
instant scoring, feedback presentation, and quiz export.
"""

from typing import List, Dict, Any, Optional
import streamlit as st


def render_interactive_quiz(quiz_data: List[Dict[str, Any]]) -> None:
    """
    Renders an interactive quiz in Streamlit with radio selections,
    submission handling, score calculation, and detailed answer explanations.
    """
    if not quiz_data:
        st.info("No quiz questions available. Click 'Generate Quiz' above to create one!")
        return

    num_questions = len(quiz_data)

    # Initialize quiz session state variables if absent
    if "quiz_user_answers" not in st.session_state:
        st.session_state["quiz_user_answers"] = {}
    if "quiz_submitted" not in st.session_state:
        st.session_state["quiz_submitted"] = False

    st.markdown(f"### 📝 Knowledge Check ({num_questions} Questions)")
    st.caption("Select your answers below and click **Submit Quiz** to check your score.")

    # Form to group all questions neatly
    with st.form("documind_quiz_form"):
        for q_idx, item in enumerate(quiz_data):
            question_text = item.get("question", f"Question {q_idx + 1}")
            options = item.get("options", [])
            correct_idx = item.get("correct_answer_index", 0)
            explanation = item.get("explanation", "")

            st.markdown(f"#### **Q{q_idx + 1}. {question_text}**")

            # Determine radio selection index
            current_choice = st.session_state["quiz_user_answers"].get(q_idx, None)
            default_index = None
            if current_choice is not None and current_choice in options:
                default_index = options.index(current_choice)

            selected_option = st.radio(
                label=f"Options for Question {q_idx + 1}",
                options=options,
                index=default_index,
                key=f"quiz_radio_{q_idx}",
                label_visibility="collapsed",
                disabled=st.session_state["quiz_submitted"]
            )

            # Store in session state
            st.session_state["quiz_user_answers"][q_idx] = selected_option

            # If quiz has been submitted, show feedback immediately below question
            if st.session_state["quiz_submitted"]:
                correct_option_text = options[correct_idx] if 0 <= correct_idx < len(options) else "N/A"
                is_correct = (selected_option == correct_option_text)

                if is_correct:
                    st.success(
                        f"✅ **Correct!**\n\n"
                        f"**Explanation:** {explanation}"
                    )
                else:
                    st.error(
                        f"❌ **Incorrect.**\n\n"
                        f"- **Your Answer:** {selected_option or 'None selected'}\n"
                        f"- **Correct Answer:** {correct_option_text}\n\n"
                        f"**Explanation:** {explanation}"
                    )

            st.divider()

        # Submit button inside form
        submit_button = st.form_submit_button(
            "📊 Submit Quiz & Check Score",
            type="primary",
            use_container_width=True,
            disabled=st.session_state["quiz_submitted"]
        )

        if submit_button:
            st.session_state["quiz_submitted"] = True
            st.rerun()

    # Post-submission scoreboard and controls
    if st.session_state["quiz_submitted"]:
        # Calculate score
        score = 0
        for q_idx, item in enumerate(quiz_data):
            options = item.get("options", [])
            correct_idx = item.get("correct_answer_index", 0)
            correct_option_text = options[correct_idx] if 0 <= correct_idx < len(options) else None
            user_ans = st.session_state["quiz_user_answers"].get(q_idx)
            if user_ans and user_ans == correct_option_text:
                score += 1

        pct = int((score / num_questions) * 100) if num_questions > 0 else 0

        st.markdown("---")
        st.markdown("### 🏆 Quiz Results")

        col1, col2, col3 = st.columns([1, 1, 1])
        col1.metric("Score", f"{score} / {num_questions}")
        col2.metric("Accuracy", f"{pct}%")
        if pct >= 80:
            col3.success("🌟 Mastery Achieved!")
        elif pct >= 60:
            col3.info("👍 Good Effort! Review key points.")
        else:
            col3.warning("📚 Keep Learning! Re-read summary.")

        # Retake button
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🔄 Retake This Quiz", use_container_width=True):
                st.session_state["quiz_submitted"] = False
                st.session_state["quiz_user_answers"] = {}
                st.rerun()

        with col_btn2:
            export_text = format_quiz_to_text(quiz_data)
            st.download_button(
                label="📥 Download Quiz & Solutions (.txt)",
                data=export_text,
                file_name="DocuMind_Quiz.txt",
                mime="text/plain",
                use_container_width=True
            )


def format_quiz_to_text(quiz_data: List[Dict[str, Any]]) -> str:
    """Formats quiz questions, options, and solutions into a structured text document."""
    lines = [
        "=" * 60,
        "DocuMind AI - Generated Document Quiz",
        "=" * 60,
        ""
    ]

    for idx, item in enumerate(quiz_data, start=1):
        lines.append(f"Q{idx}. {item.get('question', '')}")
        options = item.get("options", [])
        for opt_idx, opt in enumerate(options):
            letter = chr(65 + opt_idx)
            lines.append(f"   [{letter}] {opt}")

        correct_idx = item.get("correct_answer_index", 0)
        correct_letter = chr(65 + correct_idx) if 0 <= correct_idx < len(options) else "?"
        correct_text = options[correct_idx] if 0 <= correct_idx < len(options) else "Unknown"

        lines.append("")
        lines.append(f"   Correct Answer: [{correct_letter}] {correct_text}")
        lines.append(f"   Explanation: {item.get('explanation', '')}")
        lines.append("-" * 60)
        lines.append("")

    return "\n".join(lines)
