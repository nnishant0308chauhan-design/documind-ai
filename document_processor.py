"""
Document Processor Module for DocuMind AI
Handles text extraction from PDF and TXT files, metadata extraction,
and error handling for corrupted, empty, or scanned documents.
"""

import io
import re
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
import pypdf
from pypdf.errors import PdfStreamError, EmptyFileError


@dataclass
class DocumentPage:
    """Represents a single page or section of an extracted document."""
    page_number: int  # 1-indexed
    text: str
    char_count: int = 0
    word_count: int = 0

    def __post_init__(self):
        self.char_count = len(self.text)
        self.word_count = len(self.text.split()) if self.text else 0


@dataclass
class DocumentMetadata:
    """Metadata describing the uploaded document."""
    file_name: str
    file_type: str  # 'pdf', 'txt', etc.
    file_size_bytes: int
    total_pages: int
    total_words: int
    total_chars: int
    is_scanned_or_image_only: bool = False
    warning_message: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class ProcessedDocument:
    """Complete representation of a processed document."""
    metadata: DocumentMetadata
    pages: List[DocumentPage] = field(default_factory=list)
    full_text: str = ""

    def get_text_excerpt(self, max_chars: int = 1000) -> str:
        """Returns a truncated excerpt of the full text for preview."""
        if len(self.full_text) <= max_chars:
            return self.full_text
        return self.full_text[:max_chars].rstrip() + "..."


_RE_MULTI_NEWLINE = re.compile(r"\n{3,}")
_RE_MULTI_SPACE = re.compile(r"[ \t]+")


def clean_text(raw_text: str) -> str:
    """
    Cleans extracted text by normalizing whitespace and removing control characters.
    """
    if not raw_text:
        return ""
    # Replace non-breaking spaces and carriage returns
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    # Replace excessive consecutive newlines with two newlines
    text = _RE_MULTI_NEWLINE.sub("\n\n", text)
    # Replace excessive horizontal whitespace with a single space
    text = _RE_MULTI_SPACE.sub(" ", text)
    return text.strip()


def extract_from_pdf(
    file_input: Any,
    file_name: str = "document.pdf",
    progress_callback: Optional[Any] = None
) -> ProcessedDocument:
    """
    Extracts text and metadata from a PDF file (stream, BytesIO, or bytes).
    Handles password protection, scanned PDFs, empty pages, and stream errors.
    """
    try:
        # Normalize input to a stream
        if isinstance(file_input, bytes):
            stream = io.BytesIO(file_input)
            file_size = len(file_input)
        elif hasattr(file_input, "read"):
            # If it's a file-like object or UploadedFile from Streamlit
            pos = file_input.tell() if hasattr(file_input, "tell") else 0
            content = file_input.read()
            if hasattr(file_input, "seek"):
                file_input.seek(pos)
            file_size = len(content) if isinstance(content, bytes) else len(content.encode("utf-8"))
            stream = io.BytesIO(content) if isinstance(content, bytes) else io.BytesIO(content.encode("utf-8"))
        else:
            raise ValueError("Unsupported input format for PDF extraction.")

        if file_size == 0:
            metadata = DocumentMetadata(
                file_name=file_name,
                file_type="pdf",
                file_size_bytes=0,
                total_pages=0,
                total_words=0,
                total_chars=0,
                error_message="The uploaded PDF file is empty (0 bytes)."
            )
            return ProcessedDocument(metadata=metadata)

        reader = pypdf.PdfReader(stream)

        # Check for encryption
        if reader.is_encrypted:
            try:
                # Attempt to decrypt with empty password (some PDFs are encrypted with empty password)
                decrypt_success = reader.decrypt("")
                if not decrypt_success:
                    metadata = DocumentMetadata(
                        file_name=file_name,
                        file_type="pdf",
                        file_size_bytes=file_size,
                        total_pages=len(reader.pages) if reader.pages else 0,
                        total_words=0,
                        total_chars=0,
                        error_message="This PDF is password-protected. Please unlock or remove password before uploading."
                    )
                    return ProcessedDocument(metadata=metadata)
            except Exception:
                metadata = DocumentMetadata(
                    file_name=file_name,
                    file_type="pdf",
                    file_size_bytes=file_size,
                    total_pages=0,
                    total_words=0,
                    total_chars=0,
                    error_message="This PDF is password-protected and could not be decrypted."
                )
                return ProcessedDocument(metadata=metadata)

        total_pages = len(reader.pages)
        if total_pages == 0:
            metadata = DocumentMetadata(
                file_name=file_name,
                file_type="pdf",
                file_size_bytes=file_size,
                total_pages=0,
                total_words=0,
                total_chars=0,
                error_message="The uploaded PDF does not contain any readable pages."
            )
            return ProcessedDocument(metadata=metadata)

        pages_list: List[DocumentPage] = []
        full_text_parts: List[str] = []
        total_extracted_chars = 0

        for page_idx, page in enumerate(reader.pages):
            page_num = page_idx + 1
            try:
                raw_page_text = page.extract_text() or ""
                cleaned_page_text = clean_text(raw_page_text)
            except Exception:
                cleaned_page_text = ""

            doc_page = DocumentPage(page_number=page_num, text=cleaned_page_text)
            pages_list.append(doc_page)
            # Report progress ~every 5 pages (cheap; lets the UI show a progress bar on big PDFs)
            if progress_callback and (page_num % 5 == 0 or page_num == total_pages):
                try:
                    progress_callback(page_num, total_pages)
                except Exception:
                    pass
            if cleaned_page_text:
                full_text_parts.append(cleaned_page_text)
                total_extracted_chars += len(cleaned_page_text)

        full_text = "\n\n".join(full_text_parts)
        total_words = len(full_text.split()) if full_text else 0

        # Check for scanned or image-only PDF
        # If document has pages but average extracted characters per page is under 15
        is_scanned = False
        warning = None
        if total_pages > 0 and (total_extracted_chars / total_pages < 15):
            is_scanned = True
            warning = (
                "⚠️ Scanned / Image-only PDF Detected: This document has pages but virtually no "
                "digital text could be extracted. It is likely a scanned document or photograph. "
                "DocuMind AI requires selectable digital text or OCR preprocessing to analyze this document."
            )

        metadata = DocumentMetadata(
            file_name=file_name,
            file_type="pdf",
            file_size_bytes=file_size,
            total_pages=total_pages,
            total_words=total_words,
            total_chars=len(full_text),
            is_scanned_or_image_only=is_scanned,
            warning_message=warning
        )

        return ProcessedDocument(
            metadata=metadata,
            pages=pages_list,
            full_text=full_text
        )

    except (PdfStreamError, EmptyFileError) as e:
        return ProcessedDocument(
            metadata=DocumentMetadata(
                file_name=file_name,
                file_type="pdf",
                file_size_bytes=file_size if "file_size" in locals() else 0,
                total_pages=0,
                total_words=0,
                total_chars=0,
                error_message=f"PDF parsing error: The file appears to be corrupted or invalid ({str(e)})."
            )
        )
    except Exception as e:
        return ProcessedDocument(
            metadata=DocumentMetadata(
                file_name=file_name,
                file_type="pdf",
                file_size_bytes=file_size if "file_size" in locals() else 0,
                total_pages=0,
                total_words=0,
                total_chars=0,
                error_message=f"Unexpected error while reading PDF: {str(e)}"
            )
        )


def extract_from_txt(file_input: Any, file_name: str = "document.txt") -> ProcessedDocument:
    """
    Extracts text and metadata from a TXT file, trying multiple character encodings.
    """
    try:
        raw_bytes: bytes
        if isinstance(file_input, bytes):
            raw_bytes = file_input
        elif hasattr(file_input, "read"):
            pos = file_input.tell() if hasattr(file_input, "tell") else 0
            content = file_input.read()
            if hasattr(file_input, "seek"):
                file_input.seek(pos)
            raw_bytes = content if isinstance(content, bytes) else content.encode("utf-8")
        else:
            raise ValueError("Unsupported input format for TXT extraction.")

        file_size = len(raw_bytes)
        if file_size == 0:
            return ProcessedDocument(
                metadata=DocumentMetadata(
                    file_name=file_name,
                    file_type="txt",
                    file_size_bytes=0,
                    total_pages=0,
                    total_words=0,
                    total_chars=0,
                    error_message="The uploaded text file is empty (0 bytes)."
                )
            )

        # Try various encodings
        decoded_text = None
        for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252", "utf-16"]:
            try:
                decoded_text = raw_bytes.decode(enc)
                break
            except UnicodeDecodeError:
                continue

        if decoded_text is None:
            return ProcessedDocument(
                metadata=DocumentMetadata(
                    file_name=file_name,
                    file_type="txt",
                    file_size_bytes=file_size,
                    total_pages=0,
                    total_words=0,
                    total_chars=0,
                    error_message="Could not decode the text file. Unsupported encoding format."
                )
            )

        cleaned = clean_text(decoded_text)
        if not cleaned:
            return ProcessedDocument(
                metadata=DocumentMetadata(
                    file_name=file_name,
                    file_type="txt",
                    file_size_bytes=file_size,
                    total_pages=1,
                    total_words=0,
                    total_chars=0,
                    error_message="The uploaded text file contains only blank or invisible characters."
                )
            )

        # For TXT files, we treat the whole document as page 1
        page = DocumentPage(page_number=1, text=cleaned)
        metadata = DocumentMetadata(
            file_name=file_name,
            file_type="txt",
            file_size_bytes=file_size,
            total_pages=1,
            total_words=page.word_count,
            total_chars=page.char_count
        )

        return ProcessedDocument(
            metadata=metadata,
            pages=[page],
            full_text=cleaned
        )

    except Exception as e:
        return ProcessedDocument(
            metadata=DocumentMetadata(
                file_name=file_name,
                file_type="txt",
                file_size_bytes=0,
                total_pages=0,
                total_words=0,
                total_chars=0,
                error_message=f"Error reading text file: {str(e)}"
            )
        )


def process_uploaded_file(uploaded_file: Any, progress_callback: Optional[Any] = None) -> ProcessedDocument:
    """
    Main entry point for processing any uploaded file from Streamlit or tests.
    Validates file extension and routes to appropriate extractor.
    """
    if uploaded_file is None:
        raise ValueError("No file provided.")

    file_name = getattr(uploaded_file, "name", "uploaded_document")
    name_lower = file_name.lower()

    if name_lower.endswith(".pdf"):
        return extract_from_pdf(uploaded_file, file_name=file_name, progress_callback=progress_callback)
    elif name_lower.endswith(".txt"):
        return extract_from_txt(uploaded_file, file_name=file_name)
    else:
        return ProcessedDocument(
            metadata=DocumentMetadata(
                file_name=file_name,
                file_type="unsupported",
                file_size_bytes=0,
                total_pages=0,
                total_words=0,
                total_chars=0,
                error_message=f"Unsupported file format: '{file_name}'. DocuMind AI currently supports PDF (.pdf) and Text (.txt) files."
            )
        )
