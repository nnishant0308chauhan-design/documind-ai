"""
Unit Tests for Utility Functions, Export Formatters, and Quiz/Flashcard Helpers
"""

import unittest
from utils import (
    SAMPLE_DOCUMENTS,
    format_summary_for_download,
    format_keywords_for_download,
    format_eli5_for_download
)
from quiz_generator import format_quiz_to_text
from flashcard_generator import format_flashcards_to_text


class TestUtils(unittest.TestCase):

    def test_sample_documents(self):
        self.assertIn("Artificial Intelligence in Healthcare", SAMPLE_DOCUMENTS)
        self.assertIn("Renewable Energy & Climate Solutions", SAMPLE_DOCUMENTS)
        for name, content in SAMPLE_DOCUMENTS.items():
            self.assertGreater(len(content), 200, f"Sample '{name}' is too short.")

    def test_format_summary_for_download(self):
        res = format_summary_for_download("test.pdf", "Short", "English", "This is a summary.")
        self.assertIn("DocuMind AI", res)
        self.assertIn("test.pdf", res)
        self.assertIn("This is a summary.", res)

    def test_format_keywords_for_download(self):
        res = format_keywords_for_download("test.pdf", "Keywords: AI, Machine Learning")
        self.assertIn("DocuMind AI", res)
        self.assertIn("test.pdf", res)
        self.assertIn("Keywords: AI, Machine Learning", res)

    def test_format_eli5_for_download(self):
        res = format_eli5_for_download("test.pdf", "Deep Learning", "Like a brain learning.")
        self.assertIn("Deep Learning", res)
        self.assertIn("Like a brain learning.", res)

    def test_format_quiz_to_text(self):
        quiz_data = [
            {
                "question": "What is Python?",
                "options": ["Snake", "Language", "Car", "Planet"],
                "correct_answer_index": 1,
                "explanation": "Python is a programming language."
            }
        ]
        text = format_quiz_to_text(quiz_data)
        self.assertIn("Q1. What is Python?", text)
        self.assertIn("[B] Language", text)
        self.assertIn("Correct Answer: [B] Language", text)
        self.assertIn("Python is a programming language.", text)

    def test_format_flashcards_to_text(self):
        cards = [
            {
                "question": "What is RAG?",
                "answer": "Retrieval-Augmented Generation.",
                "category": "Architecture"
            }
        ]
        text = format_flashcards_to_text(cards)
        self.assertIn("CARD #1 [ARCHITECTURE]", text)
        self.assertIn("What is RAG?", text)
        self.assertIn("Retrieval-Augmented Generation.", text)


if __name__ == "__main__":
    unittest.main()
