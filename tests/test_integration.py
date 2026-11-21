"""
Integration Test Suite for DocuMind AI
Simulates end-to-end user workflows:
1. Document loading & text extraction
2. RAG chunking, indexing & semantic search
3. Quiz generation data flow & scoring logic
4. Flashcard navigation data flow
5. File export formatters
6. Scanned PDF fallback
7. GeminiService unconfigured & error reporting
"""

import unittest
import io
import pypdf
from document_processor import (
    extract_from_txt,
    extract_from_pdf,
    clean_text,
    process_uploaded_file,
    ProcessedDocument,
    DocumentMetadata
)
from rag_engine import RAGEngine
from ai_engine import GeminiService
from quiz_generator import format_quiz_to_text
from flashcard_generator import format_flashcards_to_text
from utils import (
    SAMPLE_DOCUMENTS,
    format_summary_for_download,
    format_keywords_for_download,
    format_eli5_for_download
)


class TestDocuMindIntegration(unittest.TestCase):

    def test_full_document_pipeline_with_sample(self):
        # 1. Load sample document
        sample_title = "Artificial Intelligence in Healthcare"
        raw_text = SAMPLE_DOCUMENTS[sample_title]
        self.assertGreater(len(raw_text), 500)

        # 2. Extract text & metadata
        proc_doc = extract_from_txt(raw_text.encode("utf-8"), file_name=f"{sample_title}.txt")
        self.assertIsNotNone(proc_doc)
        self.assertEqual(proc_doc.metadata.file_type, "txt")
        self.assertGreater(proc_doc.metadata.total_words, 100)
        self.assertFalse(proc_doc.metadata.is_scanned_or_image_only)

        # 3. RAG chunking and indexing
        rag = RAGEngine(target_chunk_chars=400, overlap_chars=80)
        chunks_indexed = rag.index_document(proc_doc)
        self.assertGreater(chunks_indexed, 1)
        self.assertTrue(rag.is_indexed())

        # 4. Query retrieval
        results = rag.retrieve("mammographic density and radiology CNNs", top_k=3)
        self.assertGreater(len(results), 0)
        top_chunk, score = results[0]
        self.assertTrue("mammographic" in top_chunk.text.lower() or "radiology" in top_chunk.text.lower())
        self.assertGreater(score, 0.1)

        # 5. Format prompt context
        context_str = rag.format_context_for_prompt(results)
        self.assertIn("EXCERPT 1", context_str)
        self.assertIn("Page 1", context_str)

    def test_quiz_scoring_logic(self):
        quiz = [
            {
                "question": "What is the primary role of CNNs in healthcare?",
                "options": ["Image analysis", "Sound recording", "Database backup", "Spreadsheet editing"],
                "correct_answer_index": 0,
                "explanation": "CNNs excel at radiological medical imaging analysis."
            },
            {
                "question": "Which regulation governs medical data privacy in the EU?",
                "options": ["HIPAA", "GDPR", "FERPA", "PCI-DSS"],
                "correct_answer_index": 1,
                "explanation": "GDPR mandates data protection in the EU."
            }
        ]

        # Simulate user selections
        user_answers_perfect = {0: "Image analysis", 1: "GDPR"}
        score_perfect = sum(1 for i, q in enumerate(quiz) if user_answers_perfect[i] == q["options"][q["correct_answer_index"]])
        self.assertEqual(score_perfect, 2)

        user_answers_partial = {0: "Image analysis", 1: "HIPAA"}
        score_partial = sum(1 for i, q in enumerate(quiz) if user_answers_partial[i] == q["options"][q["correct_answer_index"]])
        self.assertEqual(score_partial, 1)

        # Test export format
        formatted = format_quiz_to_text(quiz)
        self.assertIn("Q1. What is the primary role of CNNs in healthcare?", formatted)
        self.assertIn("[A] Image analysis", formatted)
        self.assertIn("Correct Answer: [B] GDPR", formatted)

    def test_flashcard_navigation_logic(self):
        flashcards = [
            {"question": "Q1", "answer": "A1", "category": "Concept"},
            {"question": "Q2", "answer": "A2", "category": "Fact"},
            {"question": "Q3", "answer": "A3", "category": "Definition"}
        ]
        # Simulate navigation
        idx = 0
        self.assertEqual(flashcards[idx]["question"], "Q1")
        # Next
        idx = min(idx + 1, len(flashcards) - 1)
        self.assertEqual(flashcards[idx]["question"], "Q2")
        # Prev
        idx = max(idx - 1, 0)
        self.assertEqual(flashcards[idx]["question"], "Q1")

        export_str = format_flashcards_to_text(flashcards)
        self.assertIn("CARD #1 [CONCEPT]", export_str)
        self.assertIn("CARD #2 [FACT]", export_str)

    def test_scanned_pdf_graceful_handling(self):
        writer = pypdf.PdfWriter()
        writer.add_blank_page(width=400, height=400)
        buf = io.BytesIO()
        writer.write(buf)
        buf.seek(0)

        doc = extract_from_pdf(buf.getvalue(), file_name="scanned_scan.pdf")
        self.assertTrue(doc.metadata.is_scanned_or_image_only)
        self.assertIn("Scanned / Image-only PDF Detected", doc.metadata.warning_message)
        self.assertIn("OCR", doc.metadata.warning_message)

    def test_gemini_service_safety_without_key(self):
        # Force "no key" even if a real key exists in .streamlit/secrets.toml or the environment
        import ai_engine
        original = ai_engine.resolve_api_key
        ai_engine.resolve_api_key = lambda: None
        try:
            service = GeminiService(api_key="")
        finally:
            ai_engine.resolve_api_key = original
        self.assertFalse(service.is_configured())

        # Calling operations without API key returns informative warning without crashing
        summary = service.generate_summary("Some sample text")
        self.assertIn("isn't available right now", summary)

        keywords = service.extract_keywords_and_facts("Some sample text")
        self.assertIn("isn't available right now", keywords)

        answer = service.answer_document_question("What is AI?", "AI is artificial intelligence")
        self.assertIn("isn't available right now", answer)

        eli5 = service.explain_like_beginner("Backpropagation")
        self.assertIn("isn't available right now", eli5)

        quiz, err = service.generate_quiz("Some sample text")
        self.assertIsNotNone(err)
        self.assertIn("isn't available right now", err)

        cards, err = service.generate_flashcards("Some sample text")
        self.assertIsNotNone(err)
        self.assertIn("isn't available right now", err)

    def test_gemini_model_configuration(self):
        from ai_engine import DEFAULT_MODEL, AVAILABLE_MODELS, FALLBACK_MODEL
        self.assertEqual(FALLBACK_MODEL, "gemini-3.5-flash")
        self.assertIn(DEFAULT_MODEL, AVAILABLE_MODELS)
        self.assertNotIn("gemini-2.0-flash", AVAILABLE_MODELS)
        self.assertNotIn("gemini-1.5-flash", AVAILABLE_MODELS)

    def test_error_formatting_is_friendly_and_non_technical(self):
        from ai_engine import FRIENDLY_BUSY_MESSAGE, CONFIG_PROBLEM_MESSAGE
        service = GeminiService(api_key="TEST", model="gemini-3.5-flash")
        invalid = service._format_error(Exception("API_KEY_INVALID: The key is not valid"))
        busy = service._format_error(Exception("429 RESOURCE_EXHAUSTED: Rate limit reached"))
        missing = service._format_error(Exception("404 NOT_FOUND: models/x is not found"))
        self.assertEqual(invalid, CONFIG_PROBLEM_MESSAGE)
        self.assertEqual(busy, FRIENDLY_BUSY_MESSAGE)
        self.assertEqual(missing, CONFIG_PROBLEM_MESSAGE)
        for msg in (invalid, busy, missing):
            for term in ("gemini", "api key", "secrets", "model", "429", "404"):
                self.assertNotIn(term, msg.lower())
        # technical detail is kept for admins/logs only
        self.assertIn("NOT_FOUND", service.last_error_detail)


if __name__ == "__main__":
    unittest.main()
