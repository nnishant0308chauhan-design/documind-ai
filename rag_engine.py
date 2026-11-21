"""
RAG Engine Module for DocuMind AI
Provides document chunking with page provenance, TF-IDF / BM25-based semantic retrieval,
cosine similarity ranking, and context assembly for grounded Question Answering.
"""

import math
import re
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

from document_processor import ProcessedDocument, DocumentPage


# Minimal set of standard English stopwords for cleaner token indexing
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "cannot", "could", "did", "do", "does", "doing", "down", "during", "each",
    "few", "for", "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "my", "myself", "no", "nor", "not", "of", "off",
    "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "wasn't", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself", "yourselves"
}


@dataclass
class DocumentChunk:
    """Represents a text chunk with page reference and metadata."""
    chunk_id: int
    text: str
    page_number: int  # 1-indexed
    word_count: int
    char_count: int


def tokenize(text: str) -> List[str]:
    """Tokenizes text into lowercase alphanumeric words, filtering punctuation."""
    if not text:
        return []
    words = re.findall(r"\b[a-zA-Z0-9_\-\u0900-\u097F\u0A00-\u0A7F]+\b", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


def get_clean_overlap(text: str, max_overlap_chars: int) -> str:
    """Extracts the end of text for overlap, snapping to a word boundary."""
    if not text or max_overlap_chars <= 0:
        return ""
    if len(text) <= max_overlap_chars:
        return text.strip()
    slice_text = text[-max_overlap_chars:]
    first_space = slice_text.find(" ")
    if first_space != -1 and first_space < len(slice_text) - 1:
        return slice_text[first_space + 1:].strip()
    return slice_text.strip()


def split_text_into_chunks(
    text: str,
    page_number: int = 1,
    start_chunk_id: int = 0,
    target_chunk_chars: int = 700,
    overlap_chars: int = 120
) -> List[DocumentChunk]:
    """
    Splits a single block of text (e.g. from a page) into overlapping chunks
    while respecting sentence/paragraph boundaries and word boundaries.
    """
    if not text or not text.strip():
        return []

    # Split into paragraphs or sentences
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    chunks: List[DocumentChunk] = []
    current_chunk_parts: List[str] = []
    current_char_len = 0
    chunk_id = start_chunk_id

    for para in paragraphs:
        para_len = len(para)
        # If single paragraph is larger than target_chunk_chars, split by sentence
        if para_len > target_chunk_chars:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sentence in sentences:
                sent = sentence.strip()
                if not sent:
                    continue
                if current_char_len + len(sent) > target_chunk_chars and current_chunk_parts:
                    chunk_text = " ".join(current_chunk_parts)
                    chunks.append(DocumentChunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        page_number=page_number,
                        word_count=len(chunk_text.split()),
                        char_count=len(chunk_text)
                    ))
                    chunk_id += 1
                    overlap_tokens = get_clean_overlap(chunk_text, overlap_chars)
                    current_chunk_parts = [overlap_tokens, sent] if overlap_tokens else [sent]
                    current_char_len = sum(len(p) for p in current_chunk_parts)
                else:
                    current_chunk_parts.append(sent)
                    current_char_len += len(sent) + 1
        else:
            if current_char_len + para_len > target_chunk_chars and current_chunk_parts:
                chunk_text = "\n\n".join(current_chunk_parts)
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    page_number=page_number,
                    word_count=len(chunk_text.split()),
                    char_count=len(chunk_text)
                ))
                chunk_id += 1
                overlap_text = get_clean_overlap(chunk_text, overlap_chars)
                current_chunk_parts = [overlap_text, para] if overlap_text else [para]
                current_char_len = sum(len(p) for p in current_chunk_parts)
            else:
                current_chunk_parts.append(para)
                current_char_len += para_len + 2

    # Append any remaining text
    if current_chunk_parts:
        chunk_text = "\n\n".join(current_chunk_parts).strip()
        if chunk_text:
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                page_number=page_number,
                word_count=len(chunk_text.split()),
                char_count=len(chunk_text)
            ))

    return chunks


class RAGEngine:
    """
    Lightweight, high-speed Retrieval-Augmented Generation (RAG) engine.
    Chunks documents, creates a TF-IDF term-document matrix with cosine similarity,
    and retrieves the most relevant passages with page provenance.
    """

    def __init__(self, target_chunk_chars: int = 700, overlap_chars: int = 120):
        self.target_chunk_chars = target_chunk_chars
        self.overlap_chars = overlap_chars
        self.chunks: List[DocumentChunk] = []
        self.vocabulary: Dict[str, int] = {}
        self.idf_vector: Optional[np.ndarray] = None
        self.chunk_matrix: Optional[np.ndarray] = None  # Normalized TF-IDF matrix (num_chunks, vocab_size)
        self._chunk_token_sets: List[set] = []  # precomputed once at index time (speeds up retrieve)

    def is_indexed(self) -> bool:
        """Returns True if the engine has indexed chunks and is ready to query."""
        return len(self.chunks) > 0 and self.chunk_matrix is not None

    def index_document(self, processed_doc: ProcessedDocument) -> int:
        """
        Chunks the processed document across all pages and indexes the chunks.
        Returns the total number of chunks created.
        """
        self.chunks = []
        self._chunk_token_sets = []
        chunk_id = 0

        # Chunk page by page to guarantee accurate page provenance
        if processed_doc.pages:
            for page in processed_doc.pages:
                if not page.text.strip():
                    continue
                page_chunks = split_text_into_chunks(
                    text=page.text,
                    page_number=page.page_number,
                    start_chunk_id=chunk_id,
                    target_chunk_chars=self.target_chunk_chars,
                    overlap_chars=self.overlap_chars
                )
                self.chunks.extend(page_chunks)
                chunk_id += len(page_chunks)
        elif processed_doc.full_text.strip():
            # Fallback if pages list is empty
            self.chunks = split_text_into_chunks(
                text=processed_doc.full_text,
                page_number=1,
                start_chunk_id=0,
                target_chunk_chars=self.target_chunk_chars,
                overlap_chars=self.overlap_chars
            )

        if not self.chunks:
            self.vocabulary = {}
            self.idf_vector = None
            self.chunk_matrix = None
            return 0

        # Build TF-IDF index
        self._build_tfidf_index()
        return len(self.chunks)

    def _build_tfidf_index(self):
        """Constructs vocabulary, IDF vector, and normalized TF-IDF chunk matrix."""
        num_chunks = len(self.chunks)
        chunk_token_counts: List[Dict[str, int]] = []
        doc_freq: Dict[str, int] = {}

        # 1. Collect term frequencies per chunk and document frequencies across chunks
        for chunk in self.chunks:
            tokens = tokenize(chunk.text)
            tf_map: Dict[str, int] = {}
            for t in tokens:
                tf_map[t] = tf_map.get(t, 0) + 1
            chunk_token_counts.append(tf_map)
            self._chunk_token_sets.append(set(tf_map.keys()))
            for t in tf_map.keys():
                doc_freq[t] = doc_freq.get(t, 0) + 1

        # 2. Build vocabulary
        self.vocabulary = {term: idx for idx, term in enumerate(doc_freq.keys())}
        vocab_size = len(self.vocabulary)
        if vocab_size == 0:
            self.chunk_matrix = None
            return

        # 3. Compute smooth IDF: log((1 + N) / (1 + df)) + 1
        idf = np.zeros(vocab_size, dtype=np.float32)
        for term, idx in self.vocabulary.items():
            df = doc_freq[term]
            idf[idx] = math.log((1 + num_chunks) / (1 + df)) + 1.0
        self.idf_vector = idf

        # 4. Compute sublinear TF-IDF matrix: (1 + log(tf)) * idf
        matrix = np.zeros((num_chunks, vocab_size), dtype=np.float32)
        for chunk_idx, tf_map in enumerate(chunk_token_counts):
            for term, count in tf_map.items():
                if term in self.vocabulary:
                    term_idx = self.vocabulary[term]
                    sublinear_tf = 1.0 + math.log(count)
                    matrix[chunk_idx, term_idx] = sublinear_tf * idf[term_idx]

        # 5. L2-normalize each chunk row so dot product equals cosine similarity
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.chunk_matrix = matrix / norms

    def retrieve(self, query: str, top_k: int = 4) -> List[Tuple[DocumentChunk, float]]:
        """
        Retrieves the top_k most relevant chunks for a user query.
        Returns a list of (DocumentChunk, similarity_score) tuples sorted descending by score.
        """
        if not self.is_indexed() or not query.strip():
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            # If query has no indexed words, return first chunks
            return [(chunk, 0.1) for chunk in self.chunks[:top_k]]

        vocab_size = len(self.vocabulary)
        query_vec = np.zeros(vocab_size, dtype=np.float32)

        # Count query terms
        q_counts: Dict[str, int] = {}
        for t in query_tokens:
            q_counts[t] = q_counts.get(t, 0) + 1

        for term, count in q_counts.items():
            if term in self.vocabulary:
                t_idx = self.vocabulary[term]
                sublinear_tf = 1.0 + math.log(count)
                query_vec[t_idx] = sublinear_tf * self.idf_vector[t_idx]

        q_norm = np.linalg.norm(query_vec)
        if q_norm > 0:
            query_vec = query_vec / q_norm

        # Compute cosine similarity
        scores = np.dot(self.chunk_matrix, query_vec)

        # Add keyword overlap bonus for exact token matching
        query_set = set(query_tokens)
        for idx, chunk in enumerate(self.chunks):
            chunk_tokens_set = self._chunk_token_sets[idx] if idx < len(self._chunk_token_sets) else set(tokenize(chunk.text))
            overlap = len(query_set.intersection(chunk_tokens_set))
            if overlap > 0:
                # Small bonus proportional to fraction of query matched
                scores[idx] += 0.05 * (overlap / len(query_set))

        # Get top-k indices
        top_k = min(top_k, len(self.chunks))
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[Tuple[DocumentChunk, float]] = []
        for idx in top_indices:
            score = float(scores[idx])
            results.append((self.chunks[idx], score))

        return results

    def format_context_for_prompt(self, retrieved: List[Tuple[DocumentChunk, float]]) -> str:
        """
        Formats retrieved chunks into a clean, annotated context block for the LLM prompt.
        """
        if not retrieved:
            return "No relevant document excerpts found."

        context_blocks = []
        for rank, (chunk, score) in enumerate(retrieved, start=1):
            pct = max(0, min(100, int(score * 100)))
            block = (
                f"--- EXCERPT {rank} (Page {chunk.page_number}, Relevance: {pct}%) ---\n"
                f"{chunk.text}"
            )
            context_blocks.append(block)

        return "\n\n".join(context_blocks)
