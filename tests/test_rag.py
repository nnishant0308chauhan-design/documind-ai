"""
Unit Tests for RAG Engine Module in DocuMind AI
"""

import unittest
from document_processor import ProcessedDocument, DocumentMetadata, DocumentPage
from rag_engine import RAGEngine, split_text_into_chunks, tokenize


class TestRAGEngine(unittest.TestCase):

    def setUp(self):
        self.doc = ProcessedDocument(
            metadata=DocumentMetadata(
                file_name="renewable_study.txt",
                file_type="txt",
                file_size_bytes=500,
                total_pages=2,
                total_words=80,
                total_chars=500
            ),
            pages=[
                DocumentPage(
                    page_number=1,
                    text="Solar photovoltaic panels have dropped dramatically in price over the past decade. Utility-scale solar installations are expanding globally."
                ),
                DocumentPage(
                    page_number=2,
                    text="Wind energy turbines generate significant electrical power offshore. Battery storage systems like lithium iron phosphate provide grid reliability."
                )
            ],
            full_text=(
                "Solar photovoltaic panels have dropped dramatically in price over the past decade. "
                "Utility-scale solar installations are expanding globally.\n\n"
                "Wind energy turbines generate significant electrical power offshore. "
                "Battery storage systems like lithium iron phosphate provide grid reliability."
            )
        )
        self.rag = RAGEngine(target_chunk_chars=150, overlap_chars=30)
        self.chunk_count = self.rag.index_document(self.doc)

    def test_chunking_creates_chunks_with_pages(self):
        self.assertGreater(self.chunk_count, 0)
        for chunk in self.rag.chunks:
            self.assertIn(chunk.page_number, [1, 2])
            self.assertTrue(len(chunk.text) > 0)

    def test_tokenize(self):
        tokens = tokenize("Solar PV cells generate 100% clean electricity!")
        self.assertIn("solar", tokens)
        self.assertIn("pv", tokens)
        self.assertIn("cells", tokens)
        self.assertIn("electricity", tokens)
        # Verify stopwords like 'generate' or short punctuation are filtered properly
        self.assertNotIn("!", tokens)

    def test_retrieve_solar_query(self):
        results = self.rag.retrieve("solar photovoltaic price decline", top_k=2)
        self.assertGreater(len(results), 0)
        top_chunk, score = results[0]
        self.assertEqual(top_chunk.page_number, 1)
        self.assertIn("solar", top_chunk.text.lower())
        self.assertGreater(score, 0.2)

    def test_retrieve_wind_query(self):
        results = self.rag.retrieve("wind turbine offshore power", top_k=2)
        self.assertGreater(len(results), 0)
        top_chunk, score = results[0]
        self.assertEqual(top_chunk.page_number, 2)
        self.assertIn("wind", top_chunk.text.lower())

    def test_empty_query(self):
        results = self.rag.retrieve("", top_k=2)
        self.assertEqual(len(results), 0)

    def test_format_context_for_prompt(self):
        results = self.rag.retrieve("solar", top_k=1)
        formatted = self.rag.format_context_for_prompt(results)
        self.assertIn("Page 1", formatted)
        self.assertIn("EXCERPT 1", formatted)


if __name__ == "__main__":
    unittest.main()
