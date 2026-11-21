"""
Unit Tests for Document Processor Module in DocuMind AI
"""

import io
import unittest
import pypdf
from document_processor import (
    extract_from_txt,
    extract_from_pdf,
    clean_text,
    process_uploaded_file,
    DocumentMetadata,
    ProcessedDocument
)


class TestDocumentProcessor(unittest.TestCase):

    def test_clean_text(self):
        raw = "Hello   world!\r\n\r\n\n\nThis is a   test.\xa0Done."
        cleaned = clean_text(raw)
        self.assertNotIn("\r", cleaned)
        self.assertNotIn("\xa0", cleaned)
        self.assertNotIn("   ", cleaned)
        self.assertIn("Hello world!", cleaned)

    def test_extract_txt_valid(self):
        sample_text = "This is a test document.\nIt contains multiple lines of text for validation."
        doc = extract_from_txt(sample_text.encode("utf-8"), file_name="sample.txt")
        self.assertEqual(doc.metadata.file_name, "sample.txt")
        self.assertEqual(doc.metadata.file_type, "txt")
        self.assertEqual(doc.metadata.total_pages, 1)
        self.assertGreater(doc.metadata.total_words, 0)
        self.assertIsNone(doc.metadata.error_message)
        self.assertIn("validation", doc.full_text)

    def test_extract_txt_empty(self):
        doc = extract_from_txt(b"", file_name="empty.txt")
        self.assertEqual(doc.metadata.total_words, 0)
        self.assertIsNotNone(doc.metadata.error_message)
        self.assertIn("empty", doc.metadata.error_message.lower())

    def test_extract_pdf_valid(self):
        # Generate an in-memory test PDF with text
        writer = pypdf.PdfWriter()
        # Create a page and add an annotation or text
        writer.add_blank_page(width=300, height=300)
        buf = io.BytesIO()
        writer.write(buf)
        buf.seek(0)

        doc = extract_from_pdf(buf.getvalue(), file_name="test.pdf")
        self.assertEqual(doc.metadata.file_name, "test.pdf")
        self.assertEqual(doc.metadata.file_type, "pdf")
        self.assertEqual(doc.metadata.total_pages, 1)
        # Blank page should be detected as scanned/image-only or minimal text
        self.assertTrue(doc.metadata.is_scanned_or_image_only)
        self.assertIsNotNone(doc.metadata.warning_message)

    def test_extract_pdf_empty_bytes(self):
        doc = extract_from_pdf(b"", file_name="empty.pdf")
        self.assertEqual(doc.metadata.total_pages, 0)
        self.assertIsNotNone(doc.metadata.error_message)

    def test_unsupported_file_format(self):
        class MockFile:
            name = "presentation.pptx"
            def read(self):
                return b"some content"

        doc = process_uploaded_file(MockFile())
        self.assertIsNotNone(doc.metadata.error_message)
        self.assertIn("unsupported", doc.metadata.error_message.lower())


if __name__ == "__main__":
    unittest.main()
