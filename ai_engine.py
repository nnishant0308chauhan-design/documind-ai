"""
AI Engine Module for DocuMind AI
Interfaces with Google Gemini API using the official `google-genai` SDK.
Provides summarization, keyword extraction, grounded RAG Q&A, concept simplification,
quiz generation, and flashcard generation.
"""

import os
import json
import re
import time
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple, Callable
from dotenv import load_dotenv

# Load environment variables from .env file if available (optional fallback only)
load_dotenv()

logger = logging.getLogger("documind.ai")

# Google GenAI imports
try:
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None
    types = None
    APIError = Exception


# ---------------------------------------------------------------------------
# CENTRAL CONFIGURATION (single place to change model / retry / chunk settings)
# ---------------------------------------------------------------------------
# Fallback model used when neither .streamlit/secrets.toml nor the environment
# defines GEMINI_MODEL.
FALLBACK_MODEL = "gemini-3.5-flash"

# Retry policy for temporary Gemini errors (429 / 503 / resource exhausted)
MAX_RETRIES = 3              # maximum retries AFTER the first attempt
BACKOFF_BASE_SECONDS = 2.0   # waits 2s, 4s, 8s (+ small jitter)
BACKOFF_MAX_SECONDS = 20.0

# Summarization / chunking settings
SINGLE_PASS_CHAR_LIMIT = 60000   # docs up to this size are summarized in ONE API call
CHUNK_CHAR_LIMIT = 30000         # size of each section when a document is larger
MAX_CHUNKS = 12                  # hard cap on map-step calls for huge PDFs
MAX_DOC_CHARS = CHUNK_CHAR_LIMIT * MAX_CHUNKS  # text beyond this is sampled, not sent

# End-user messages: plain language, no technical terms, no keys, no model names.
FRIENDLY_BUSY_MESSAGE = (
    "⏳ Our AI assistant is very busy right now. We already retried automatically, "
    "but it's still unavailable. Please wait a minute and try again."
)
NOT_CONFIGURED_MESSAGE = (
    "⚠️ The AI assistant isn't available right now. Please try again later."
)
CONFIG_PROBLEM_MESSAGE = (
    "⚠️ The AI assistant is temporarily unavailable due to a service configuration issue. "
    "Please try again later or contact the site administrator."
)
NETWORK_MESSAGE = (
    "🌐 We couldn't reach the AI service. Please check your internet connection and try again."
)
GENERIC_ERROR_MESSAGE = (
    "⚠️ Something went wrong while generating your result. Please try again in a moment."
)


def _read_streamlit_secret(name: str) -> Optional[str]:
    """Reads a value from .streamlit/secrets.toml (server-side only). Never raises."""
    try:
        import streamlit as st
        value = st.secrets.get(name)  # raises if no secrets file exists
        if isinstance(value, str) and value.strip():
            return value.strip()
    except Exception:
        pass
    return None


def resolve_api_key() -> Optional[str]:
    """
    Resolves the Gemini API key on the Python backend ONLY, in this order:
    1. .streamlit/secrets.toml  -> GEMINI_API_KEY (or GOOGLE_API_KEY)
    2. GEMINI_API_KEY / GOOGLE_API_KEY environment variable (or .env file)
    The key is never accepted from the UI and never displayed.
    """
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        key = _read_streamlit_secret(name)
        if key:
            return key
    env_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()
    return None


def resolve_model() -> str:
    """Central model configuration: secrets.toml GEMINI_MODEL > env GEMINI_MODEL > fallback."""
    return _read_streamlit_secret("GEMINI_MODEL") or os.getenv("GEMINI_MODEL") or FALLBACK_MODEL


def is_admin_mode() -> bool:
    """Administrator diagnostics are shown only if DOCUMIND_ADMIN is truthy in secrets/env."""
    raw = _read_streamlit_secret("DOCUMIND_ADMIN") or os.getenv("DOCUMIND_ADMIN") or ""
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


# Kept for backwards compatibility with existing imports/tests
DEFAULT_MODEL = resolve_model()
AVAILABLE_MODELS = [DEFAULT_MODEL]


def is_retryable_error(error: Exception) -> bool:
    """
    True only for TEMPORARY errors: HTTP 429, 503, resource exhausted / overloaded.
    Invalid keys, permission, not-found and bad-request errors are permanent.
    """
    err_str = str(error)
    err_lower = err_str.lower()

    # Permanent errors are never retried (checked first so they always win)
    permanent_markers = (
        "api_key_invalid", "invalid api key", "api key not valid", "unauthenticated",
        "permission_denied", "not_found", "invalid_argument", "failed_precondition",
    )
    if any(m in err_lower for m in permanent_markers):
        return False
    code = getattr(error, "code", None)
    if isinstance(code, int) and code in (400, 401, 403, 404):
        return False

    if isinstance(code, int) and code in (429, 500, 502, 503, 504):
        return True
    temporary_markers = (
        "429", "503", "resource_exhausted", "resource exhausted", "unavailable",
        "overloaded", "high demand", "rate limit", "try again later",
    )
    return any(m in err_lower for m in temporary_markers)


def is_unavailable_error(error: Exception) -> bool:
    """True when an error is of the temporary kind (used to show the friendly busy message)."""
    return is_retryable_error(error)


# ---------------------------------------------------------------------------
# TEXT PREPARATION HELPERS (fast, no API calls)
# ---------------------------------------------------------------------------
_SUMMARY_CACHE: Dict[str, str] = {}
_SUMMARY_CACHE_MAX = 32


def _cache_key(*parts: Any) -> str:
    return hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def _cache_put(key: str, value: str) -> None:
    if len(_SUMMARY_CACHE) >= _SUMMARY_CACHE_MAX:
        _SUMMARY_CACHE.pop(next(iter(_SUMMARY_CACHE)))
    _SUMMARY_CACHE[key] = value


def split_into_sections(text: str, chunk_limit: int = CHUNK_CHAR_LIMIT) -> List[str]:
    """
    Splits long text into sections of at most ~chunk_limit characters, breaking on
    paragraph boundaries (falling back to sentence/word boundaries for huge paragraphs).
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_limit:
        return [text]

    sections: List[str] = []
    current: List[str] = []
    current_len = 0

    def flush():
        nonlocal current, current_len
        if current:
            sections.append("\n\n".join(current).strip())
        current, current_len = [], 0

    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        # A single oversized paragraph is cut into hard slices on whitespace
        while len(para) > chunk_limit:
            cut = para.rfind(" ", 0, chunk_limit)
            cut = cut if cut > chunk_limit // 2 else chunk_limit
            flush()
            sections.append(para[:cut].strip())
            para = para[cut:].strip()
        if current_len + len(para) + 2 > chunk_limit:
            flush()
        current.append(para)
        current_len += len(para) + 2
    flush()
    return [sec for sec in sections if sec]


def sample_text(text: str, limit: int) -> str:
    """
    Returns at most `limit` characters, sampled evenly from the start, middle and end
    so quizzes/flashcards/keywords cover the whole document rather than only page 1.
    """
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    parts = 4
    piece = limit // parts
    step = (len(text) - piece) // (parts - 1)
    out = []
    for i in range(parts):
        start = i * step
        seg = text[start:start + piece]
        if i > 0:  # avoid starting mid-word
            sp = seg.find(" ")
            seg = seg[sp + 1:] if 0 <= sp < 80 else seg
        out.append(seg.strip())
    return "\n\n[...]\n\n".join(out)


class GeminiService:
    """
    Service wrapper around Google GenAI client.
    Handles authentication, prompt execution, structured JSON parsing,
    and user-friendly error reporting.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        # The key comes from the backend (secrets.toml / environment). The optional
        # `api_key` argument exists only for programmatic/test use; the UI never passes one.
        self.api_key = (api_key.strip() if isinstance(api_key, str) and api_key.strip() else None) \
            or resolve_api_key()
        self.model = model or resolve_model()
        self._client: Optional[Any] = None
        self._sleep: Callable[[float], None] = time.sleep  # injectable for tests
        self.last_retry_count: int = 0
        self.last_error_detail: str = ""  # technical detail for administrators/logs only

        if self.api_key and GENAI_AVAILABLE:
            try:
                self._client = genai.Client(api_key=self.api_key)
            except Exception:
                self._client = None

    def is_configured(self) -> bool:
        """Returns True if the service has an API key and client initialized."""
        return bool(self.api_key and self._client)

    def _call_with_retry(self, fn: Callable[[], Any]) -> Any:
        """
        Runs `fn` and retries TEMPORARY failures (429 / 503 / resource exhausted)
        with exponential backoff: 2s, 4s, 8s (+jitter), at most MAX_RETRIES retries.
        Permanent errors (invalid key, permission, 404, bad request) raise immediately.
        """
        import random
        self.last_retry_count = 0
        attempt = 0
        while True:
            try:
                return fn()
            except Exception as e:
                if not is_retryable_error(e) or attempt >= MAX_RETRIES:
                    raise
                delay = min(BACKOFF_BASE_SECONDS * (2 ** attempt), BACKOFF_MAX_SECONDS)
                delay += random.uniform(0, 0.5)
                attempt += 1
                self.last_retry_count = attempt
                self._sleep(delay)

    def _generate(self, prompt: str, config: Any) -> Any:
        """Single entry point for every Gemini call (adds retry/backoff)."""
        return self._call_with_retry(
            lambda: self._client.models.generate_content(
                model=self.model, contents=prompt, config=config
            )
        )

    def _extract_response_text(self, response: Any) -> Tuple[Optional[str], Optional[str]]:
        """
        Robustly extracts generated text from a Gemini API response object.

        Root cause context: Thinking models (gemini-2.5-flash, gemini-3.5-flash, etc.)
        return "thought" parts (part.thought == True) alongside actual text parts.
        The SDK's response.text property SKIPS thought parts by design. When token
        budgets are low, the model may spend all tokens on thinking and produce zero
        output text, making response.text return None even though the API call succeeded.

        This method correctly handles:
        - Thought-only responses (thinking consumed all tokens)
        - Safety-blocked responses with specific block reason
        - STOP / MAX_TOKENS / OTHER finish reasons
        - Missing candidates or empty content parts
        - Both text and non-text parts mixed together

        Returns: (extracted_text_or_None, error_message_or_None)
        """
        if response is None:
            return None, "No response object returned by the API."

        # Check prompt-level blocking first
        if hasattr(response, "prompt_feedback") and response.prompt_feedback:
            block_reason = getattr(response.prompt_feedback, "block_reason", None)
            if block_reason:
                return None, (
                    f"⚠️ Your request was blocked by the AI safety filters (reason: {block_reason}). "
                    "Please rephrase your question or simplify the document content."
                )

        # Check candidates
        if not getattr(response, "candidates", None):
            return None, (
                "⚠️ The AI returned no answer. This may be due to safety filtering or a temporary issue. Please try again in a moment."
            )

        candidate = response.candidates[0]
        finish_reason = str(getattr(candidate, "finish_reason", "UNKNOWN"))

        # Inspect safety ratings for candidate-level blocking
        safety_ratings = getattr(candidate, "safety_ratings", None) or []
        blocked_categories = [
            str(getattr(r, "category", "UNKNOWN"))
            for r in safety_ratings
            if getattr(r, "blocked", False)
        ]
        if blocked_categories:
            return None, (
                f"⚠️ The response was blocked by safety filters for: {', '.join(blocked_categories)}. "
                "Please rephrase your request or use different content."
            )

        # Inspect content parts — collect only non-thought text parts
        content = getattr(candidate, "content", None)
        if not content or not getattr(content, "parts", None):
            if "MAX_TOKENS" in finish_reason:
                return None, (
                    "⚠️ The answer was cut off before it could be written. Please try again or use a shorter section."
                )
            return None, (
                f"⚠️ The AI returned an empty response (finish_reason: {finish_reason}). "
                "Please try again."
            )

        # Gather text from non-thought parts
        text_parts = []
        thought_parts_count = 0
        for part in content.parts:
            is_thought = isinstance(getattr(part, "thought", None), bool) and part.thought
            if is_thought:
                thought_parts_count += 1
                continue
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text:
                text_parts.append(part_text)

        extracted = "".join(text_parts).strip()

        if not extracted:
            if thought_parts_count > 0 and "MAX_TOKENS" in finish_reason:
                return None, (
                    "⚠️ The AI could not finish writing a response. Please try again."
                )
            elif thought_parts_count > 0:
                return None, (
                    "⚠️ The AI did not return any text. Please try again."
                )
            return None, (
                f"⚠️ The AI returned an empty response (finish_reason: {finish_reason}). "
                "Please try again or use a shorter document excerpt."
            )

        return extracted, None

    def _make_config(
        self,
        temperature: float = 0.2,
        max_output_tokens: int = 4000,
        response_mime_type: Optional[str] = None
    ) -> Any:
        """
        Builds a GenerateContentConfig with thinking disabled.

        WHY thinking_budget=0:
        Gemini 2.5/3.5 Flash are "thinking models" that spend output tokens on
        internal chain-of-thought reasoning before writing the actual response.
        With thinking enabled, a low max_output_tokens budget (e.g. 10 for connection
        test, or 3000 for quiz) may be fully consumed by thinking, leaving zero tokens
        for the actual text — which causes response.text to return None.
        Setting thinking_budget=0 disables this internal reasoning, ensuring all
        output tokens are used for the visible generated text.
        """
        kwargs = dict(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
        if response_mime_type:
            kwargs["response_mime_type"] = response_mime_type
        return types.GenerateContentConfig(**kwargs)

    def test_connection(self) -> Tuple[bool, str]:
        """
        Tests the API key by sending a minimal test request.
        Returns (success_boolean, message).
        """
        if not GENAI_AVAILABLE:
            return False, "The 'google-genai' package is not installed."
        if not self.is_configured():
            return False, NOT_CONFIGURED_MESSAGE

        try:
            response = self._generate("Respond with the exact word 'READY' to verify the API connection.", self._make_config(temperature=0.0, max_output_tokens=50))
            text, err = self._extract_response_text(response)
            if text:
                return True, f"✅ Connection successful! Model '{self.model}' is ready."
            return False, err or "The service responded but returned empty content."
        except Exception as e:
            msg = self._format_error(e)
            return False, f"{msg}  \n_Technical detail (admin only):_ `{self.last_error_detail}`"

    def _format_error(self, error: Exception) -> str:
        """
        Translates technical API exceptions into short, friendly, non-technical messages.
        The raw technical detail is kept in `last_error_detail` and written to the server log
        (never shown to end users).
        """
        err_str = str(error)
        err_lower = err_str.lower()
        self.last_error_detail = f"{type(error).__name__}: {err_str}"[:600]
        logger.warning("Gemini call failed: %s", self.last_error_detail)

        if (
            "api_key_invalid" in err_lower
            or "invalid api key" in err_lower
            or "api key not valid" in err_lower
            or "unauthenticated" in err_lower
            or "permission_denied" in err_lower
            or "not_found" in err_lower
            or "is not found" in err_lower
            or "not supported" in err_lower
            or "403" in err_str
            or "401" in err_str
            or "404" in err_str
            or ("400" in err_str and "key" in err_lower)
        ):
            return CONFIG_PROBLEM_MESSAGE

        if (
            "429" in err_str or "503" in err_str
            or "resource_exhausted" in err_lower or "quota" in err_lower
            or "rate limit" in err_lower or "unavailable" in err_lower
            or "overloaded" in err_lower or "high demand" in err_lower
        ):
            return FRIENDLY_BUSY_MESSAGE

        if (
            "timeout" in err_lower or "connection" in err_lower or "network" in err_lower
            or "getaddrinfo" in err_lower or "dns" in err_lower or "11001" in err_str
        ):
            return NETWORK_MESSAGE

        return GENERIC_ERROR_MESSAGE

    _STYLE_INSTRUCTIONS = {
        "Short": "Write a concise executive summary (1-2 paragraphs): core premise and main conclusion.",
        "Medium": "Write a balanced summary (3-4 paragraphs): background, key arguments, takeaways.",
        "Detailed": "Write an in-depth summary covering all major sections, arguments, methods and conclusions.",
        "Bullet-point": "Write a hierarchical markdown bullet-point summary of all key findings and takeaways.",
        "Section-wise": "Organize the summary by the document's chapters/sections/headings, summarizing each in order.",
    }

    @staticmethod
    def _language_instruction(language: str) -> str:
        if language == "Hindi":
            return "Write the ENTIRE summary in fluent, natural Hindi (हिंदी) using Devanagari script."
        if language == "Punjabi":
            return "Write the ENTIRE summary in fluent, natural Punjabi (ਪੰਜਾਬੀ) using Gurmukhi script."
        return "Write in clear, professional English."

    def generate_summary(
        self,
        document_text: str,
        summary_type: str = "Medium",
        language: str = "English",
        temperature: float = 0.2,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> str:
        """
        Generates a document summary (styles: Short, Medium, Detailed, Bullet-point,
        Section-wise; languages: English, Hindi, Punjabi).

        Speed strategy:
        - Documents up to SINGLE_PASS_CHAR_LIMIT chars -> exactly ONE Gemini call.
        - Larger documents -> split into ~CHUNK_CHAR_LIMIT sections, each condensed to short
          notes (max MAX_CHUNKS calls, 3 in parallel), then ONE final call writes the summary.
        - Identical repeat requests are served from an in-memory cache (zero API calls).
        `progress_callback(step, total, message)` is called from the calling thread.
        """
        if not self.is_configured():
            return NOT_CONFIGURED_MESSAGE

        text = (document_text or "").strip()
        if not text:
            return "⚠️ The document contains no readable text to summarize."

        def report(step: int, total: int, msg: str):
            if progress_callback:
                try:
                    progress_callback(step, total, msg)
                except Exception:
                    pass

        ckey = _cache_key("summary", self.model, summary_type, language, temperature, text)
        if ckey in _SUMMARY_CACHE:
            report(1, 1, "Summary ready.")
            return _SUMMARY_CACHE[ckey]

        style = self._STYLE_INSTRUCTIONS.get(summary_type, self._STYLE_INSTRUCTIONS["Medium"])
        lang = self._language_instruction(language)

        def final_prompt(body: str, from_notes: bool) -> str:
            source = "section notes of a long document" if from_notes else "document"
            return (
                f"You are DocuMind AI. Summarize the {source} below.\n"
                f"Style ({summary_type}): {style}\n"
                f"Language ({language}): {lang}\n"
                "Use only information in the text; do not invent facts. Use clean markdown.\n\n"
                f"TEXT:\n\"\"\"\n{body}\n\"\"\"\n"
            )

        try:
            # ---- Fast path: one call ----
            if len(text) <= SINGLE_PASS_CHAR_LIMIT:
                report(1, 1, "Writing your summary...")
                response = self._generate(
                    final_prompt(text, False),
                    self._make_config(temperature=temperature, max_output_tokens=4000),
                )
                result, err = self._extract_response_text(response)
                if not result:
                    return err or "⚠️ No summary could be generated. Please try again."
                _cache_put(ckey, result)
                return result

            # ---- Large document: condense sections, then one final call ----
            sections = split_into_sections(text, CHUNK_CHAR_LIMIT)
            note = ""
            if len(sections) > MAX_CHUNKS:
                # Keep first/last sections and sample the middle evenly so the whole document is represented
                idxs = sorted({round(i * (len(sections) - 1) / (MAX_CHUNKS - 1)) for i in range(MAX_CHUNKS)})
                sections = [sections[i] for i in idxs]
                note = "\n\n_Note: this document is very large, so evenly spaced sections were summarized._"
            total_steps = len(sections) + 1

            def condense(idx_sec: Tuple[int, str]) -> Tuple[int, str]:
                idx, sec = idx_sec
                prompt = (
                    "Condense this part of a longer document into concise bullet notes (max 250 words). "
                    "Keep key facts, figures, names and conclusions. Plain English, no preamble.\n\n"
                    f"PART {idx + 1} OF {len(sections)}:\n\"\"\"\n{sec}\n\"\"\"\n"
                )
                resp = self._generate(prompt, self._make_config(temperature=0.1, max_output_tokens=900))
                out, e = self._extract_response_text(resp)
                if not out:
                    raise RuntimeError(e or "empty section summary")
                return idx, out

            from concurrent.futures import ThreadPoolExecutor, as_completed
            notes: Dict[int, str] = {}
            done = 0
            report(0, total_steps, f"Large document: reading {len(sections)} sections...")
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = [pool.submit(condense, (i, sec)) for i, sec in enumerate(sections)]
                for fut in as_completed(futures):
                    idx, out = fut.result()  # raises -> handled below
                    notes[idx] = out
                    done += 1
                    report(done, total_steps, f"Read section {done} of {len(sections)}...")

            combined = "\n\n".join(f"[Section {i + 1}]\n{notes[i]}" for i in sorted(notes))
            report(total_steps - 1, total_steps, "Writing your final summary...")
            response = self._generate(
                final_prompt(combined, True),
                self._make_config(temperature=temperature, max_output_tokens=4000),
            )
            result, err = self._extract_response_text(response)
            if not result:
                return err or "⚠️ No summary could be generated. Please try again."
            result += note
            _cache_put(ckey, result)
            report(total_steps, total_steps, "Summary ready.")
            return result
        except Exception as e:
            return f"⚠️ Error generating summary: {self._format_error(e)}"

    def extract_keywords_and_facts(self, document_text: str, temperature: float = 0.1) -> str:
        """
        Extracts key concepts, definitions, dates, facts, names, and technical terms.
        Large documents are sampled evenly (start/middle/end) to keep this to ONE call.
        """
        if not self.is_configured():
            return NOT_CONFIGURED_MESSAGE

        text = sample_text(document_text, SINGLE_PASS_CHAR_LIMIT)
        prompt = (
            "You are DocuMind AI. Extract key information from the document below. "
            "Only use what the text states; never add outside facts. Output these markdown sections:\n"
            "### 🔑 1. Core Keywords & Main Themes\n- 8-15 keywords, each with a one-sentence relevance note.\n"
            "### 📖 2. Key Definitions & Terminology\n- Important terms with the document's definitions.\n"
            "### 📅 3. Crucial Facts, Dates & Statistics\n- Specific facts, dates, numbers, milestones.\n"
            "### 👤 4. Notable Names, Organizations & Entities\n- People, organizations, technologies and their roles.\n"
            "### ⚙️ 5. Technical Concepts & Glossary\n- Specialized terms in plain language.\n\n"
            f"DOCUMENT:\n\"\"\"\n{text}\n\"\"\"\n"
        )
        try:
            response = self._generate(prompt, self._make_config(temperature=temperature, max_output_tokens=4000))
            result, err = self._extract_response_text(response)
            if result:
                return result
            return err or "⚠️ No keywords could be extracted. Please try again."
        except Exception as e:
            return f"⚠️ Error extracting keywords: {self._format_error(e)}"

    def answer_document_question(
        self,
        question: str,
        context_string: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.2
    ) -> str:
        """
        Answers a user query strictly using retrieved document context (RAG).
        """
        if not self.is_configured():
            return NOT_CONFIGURED_MESSAGE

        # Format recent history if available
        history_str = ""
        if chat_history:
            formatted_history = []
            # Keep last 4 turns for context
            for msg in chat_history[-4:]:
                role = "User" if msg.get("role") == "user" else "Assistant"
                formatted_history.append(f"{role}: {msg.get('content', '')}")
            history_str = "\n".join(formatted_history)

        prompt = f"""You are DocuMind AI, an intelligent, grounded document Q&A assistant.

TASK:
Answer the user's question using ONLY the provided document excerpts below.

CRITICAL RULES:
1. STRICT GROUNDING: Answer using information found in the excerpts.
2. MISSING INFORMATION: If the answer is NOT present or cannot be directly inferred from the excerpts, state clearly:
   "I could not find the answer to this question in the provided document excerpts. Please check if your question relates to the uploaded document or try rephrasing."
   DO NOT make up an answer or pull external unverified facts.
3. CITATIONS: Whenever possible, reference the excerpt number or page number from which you obtained the information (e.g., "[Page 2]").
4. TONE: Clear, helpful, educational, and concise.

PREVIOUS CONVERSATION CONTEXT (if relevant):
{history_str if history_str else "None"}

RELEVANT DOCUMENT EXCERPTS:
\"\"\"
{context_string}
\"\"\"

USER'S QUESTION:
{question}
"""
        try:
            response = self._generate(prompt, self._make_config(temperature=temperature, max_output_tokens=2500))
            text, err = self._extract_response_text(response)
            if text:
                return text
            return err or "⚠️ No answer could be generated. Please try again."
        except Exception as e:
            return f"⚠️ Error answering question: {self._format_error(e)}"

    def explain_like_beginner(
        self,
        concept_or_excerpt: str,
        document_context: str = "",
        temperature: float = 0.3
    ) -> str:
        """
        Explains a complex concept or paragraph like explaining to a beginner (ELI5),
        with plain English, real-world analogies, and a glossary.
        """
        if not self.is_configured():
            return NOT_CONFIGURED_MESSAGE

        prompt = f"""You are DocuMind AI's "Explain Like I'm a Beginner" tutor.

TASK:
Take the concept or paragraph provided below and explain it so that anyone, even someone with zero background knowledge, can understand it immediately.

FORMAT YOUR RESPONSE IN THESE 4 STRUCTURED SECTIONS:

### 🌟 1. The Big Picture (Plain English)
- Explain the concept in super simple, conversational English.
- Use short sentences and eliminate all confusing jargon.

### 💡 2. Real-World Practical Example
- Show how this concept actually works in real daily life or in a real scenario that anyone can visualize.

### 🚲 3. A Relatable Analogy
- Create a vivid, memorable comparison (e.g. "Think of it like a restaurant kitchen..." or "Imagine you are packing a backpack...").

### 📖 4. Mini-Glossary of Tricky Words
- Bullet points defining any technical or intimidating words used in the original text in 1 friendly sentence each.

TARGET CONCEPT / PARAGRAPH TO EXPLAIN:
\"\"\"
{concept_or_excerpt}
\"\"\"

DOCUMENT BACKGROUND CONTEXT (for reference):
\"\"\"
{document_context[:2000] if document_context else "No extra background"}
\"\"\"
"""
        try:
            response = self._generate(prompt, self._make_config(temperature=temperature, max_output_tokens=2500))
            text, err = self._extract_response_text(response)
            if text:
                return text
            return err or "⚠️ No explanation could be generated. Please try again."
        except Exception as e:
            return f"⚠️ Error explaining concept: {self._format_error(e)}"

    def generate_quiz(self, document_text: str, num_questions: int = 5) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Generates multiple-choice questions with 4 options, answer key, and explanations.
        Returns (list_of_questions, error_message).
        """
        if not self.is_configured():
            return [], NOT_CONFIGURED_MESSAGE

        # Sample up to ~12,000 characters evenly across the whole document (fast, covers all parts)
        trimmed_text = sample_text(document_text, 12000)

        prompt = f"""You are DocuMind AI's educational quiz generator.

TASK:
Create exactly {num_questions} high-quality, educational Multiple Choice Questions (MCQs) based strictly on the document text provided below.

REQUIREMENTS FOR EACH QUESTION:
1. Clear question testing genuine comprehension (not trivial phrasing).
2. Exactly 4 plausible options (A, B, C, D).
3. Exactly one correct answer index (0 for Option A, 1 for Option B, 2 for Option C, 3 for Option D).
4. A concise explanation stating why the answer is correct based on the text.

OUTPUT FORMAT:
Return ONLY a valid JSON array of objects. Do NOT include any markdown introductory or concluding text outside the JSON.
Schema:
[
  {{
    "question": "What is ...?",
    "options": ["Option A text", "Option B text", "Option C text", "Option D text"],
    "correct_answer_index": 0,
    "explanation": "Brief explanation referencing the document."
  }}
]

DOCUMENT TEXT:
\"\"\"
{trimmed_text}
\"\"\"
"""
        try:
            response = self._generate(prompt, self._make_config(
                    temperature=0.2,
                    max_output_tokens=4000,
                    response_mime_type="application/json"
                ))

            raw_text, err = self._extract_response_text(response)
            if not raw_text:
                return [], err or "API returned an empty response for quiz generation."

            # Parse JSON safely
            quiz_data = self._parse_json_array(raw_text)
            if not quiz_data:
                return [], "Failed to parse the generated quiz into valid questions. Please try generating again."

            # Validate structure
            validated_quiz = []
            for item in quiz_data:
                if (
                    isinstance(item, dict)
                    and "question" in item
                    and "options" in item
                    and isinstance(item["options"], list)
                    and len(item["options"]) >= 2
                    and "correct_answer_index" in item
                ):
                    # Ensure correct_answer_index is valid integer
                    idx = int(item["correct_answer_index"])
                    if 0 <= idx < len(item["options"]):
                        validated_quiz.append({
                            "question": str(item["question"]),
                            "options": [str(opt) for opt in item["options"]],
                            "correct_answer_index": idx,
                            "explanation": str(item.get("explanation", "Correct based on the document."))
                        })

            if not validated_quiz:
                return [], "Quiz questions were generated but did not match the expected format."

            return validated_quiz, None

        except Exception as e:
            return [], self._format_error(e)

    def generate_flashcards(self, document_text: str, num_cards: int = 8) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Generates study flashcards (Question/Prompt on front, Answer on back) from the text.
        Returns (list_of_flashcards, error_message).
        """
        if not self.is_configured():
            return [], NOT_CONFIGURED_MESSAGE

        trimmed_text = sample_text(document_text, 12000)

        prompt = f"""You are DocuMind AI's flashcard creator.

TASK:
Generate exactly {num_cards} high-impact study flashcards based on the key concepts, definitions, and facts in the document text provided below.

REQUIREMENTS:
1. "question": A targeted question, term, or prompt for the front of the flashcard.
2. "answer": A clear, concise, self-contained explanation for the back of the flashcard.
3. "category": A tag such as "Concept", "Definition", "Key Fact", or "Formula".

OUTPUT FORMAT:
Return ONLY a valid JSON array of objects. Do NOT include any markdown intro or outro text outside the JSON.
Schema:
[
  {{
    "question": "Front of card (term or question)",
    "answer": "Back of card (explanation or definition)",
    "category": "Definition"
  }}
]

DOCUMENT TEXT:
\"\"\"
{trimmed_text}
\"\"\"
"""
        try:
            response = self._generate(prompt, self._make_config(
                    temperature=0.2,
                    max_output_tokens=4000,
                    response_mime_type="application/json"
                ))

            raw_text, err = self._extract_response_text(response)
            if not raw_text:
                return [], err or "API returned an empty response for flashcard generation."

            card_data = self._parse_json_array(raw_text)
            if not card_data:
                return [], "Failed to parse the generated flashcards into valid JSON. Please try again."

            validated_cards = []
            for item in card_data:
                if isinstance(item, dict) and "question" in item and "answer" in item:
                    validated_cards.append({
                        "question": str(item["question"]),
                        "answer": str(item["answer"]),
                        "category": str(item.get("category", "Concept"))
                    })

            if not validated_cards:
                return [], "No valid flashcards could be formatted."

            return validated_cards, None

        except Exception as e:
            return [], self._format_error(e)

    def _parse_json_array(self, text: str) -> Optional[List[Dict[str, Any]]]:
        """Safely parses JSON array from text, stripping markdown code blocks if present."""
        cleaned = text.strip()
        # Remove markdown code fences like ```json ... ```
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
        cleaned = cleaned.strip()

        # Try direct parse
        try:
            data = json.loads(cleaned)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                # Check if it was wrapped in a key like {"questions": [...]} or {"flashcards": [...]}
                for key in ["questions", "quiz", "flashcards", "cards", "data"]:
                    if key in data and isinstance(data[key], list):
                        return data[key]
                return [data]
        except json.JSONDecodeError:
            pass

        # Regex search for [ ... ]
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass

        return None
