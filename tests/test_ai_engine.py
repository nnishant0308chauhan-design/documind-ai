"""
Tests for retry/backoff, chunking and key handling in ai_engine.

These use a stub client that raises the SDK's real error classes. They verify
control flow only (retry counts, no-retry on permanent errors, call counts).
No AI output is simulated or claimed as real Gemini output.
"""

import unittest
from unittest import mock

from google.genai import errors

import ai_engine
from ai_engine import (
    GeminiService, is_retryable_error, split_into_sections, sample_text,
    MAX_RETRIES, SINGLE_PASS_CHAR_LIMIT, CHUNK_CHAR_LIMIT, MAX_CHUNKS,
)


def server_err(code=503, status="UNAVAILABLE"):
    return errors.ServerError(code, {"error": {"code": code, "message": "busy", "status": status}})


def client_err(code, status, msg="x"):
    return errors.ClientError(code, {"error": {"code": code, "message": msg, "status": status}})


class _Part:
    def __init__(self, text):
        self.text, self.thought = text, False


class _Resp:
    """Minimal response shape understood by _extract_response_text (test double)."""
    def __init__(self, text):
        content = type("C", (), {"parts": [_Part(text)]})()
        cand = type("Cand", (), {"content": content, "finish_reason": "STOP", "safety_ratings": []})()
        self.candidates = [cand]
        self.prompt_feedback = None


class _Models:
    def __init__(self, behaviours):
        self.behaviours = list(behaviours)
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        b = self.behaviours.pop(0) if len(self.behaviours) > 1 else self.behaviours[0]
        if isinstance(b, Exception):
            raise b
        return _Resp(b)


def make_service(behaviours):
    svc = GeminiService(api_key="test-key-not-real", model="test-model")
    svc._client = type("Client", (), {})()
    svc._client.models = _Models(behaviours)
    svc.sleeps = []
    svc._sleep = svc.sleeps.append  # no real waiting in tests
    return svc


class TestRetryClassification(unittest.TestCase):
    def test_temporary_errors_are_retryable(self):
        self.assertTrue(is_retryable_error(server_err(503)))
        self.assertTrue(is_retryable_error(client_err(429, "RESOURCE_EXHAUSTED")))
        self.assertTrue(is_retryable_error(Exception("503 UNAVAILABLE model overloaded")))

    def test_permanent_errors_are_not_retryable(self):
        self.assertFalse(is_retryable_error(client_err(400, "INVALID_ARGUMENT", "API key not valid")))
        self.assertFalse(is_retryable_error(client_err(401, "UNAUTHENTICATED")))
        self.assertFalse(is_retryable_error(client_err(403, "PERMISSION_DENIED")))
        self.assertFalse(is_retryable_error(client_err(404, "NOT_FOUND")))
        self.assertFalse(is_retryable_error(Exception("API_KEY_INVALID")))


class TestRetryBehaviour(unittest.TestCase):
    def test_recovers_after_temporary_errors(self):
        svc = make_service([server_err(503), client_err(429, "RESOURCE_EXHAUSTED"), "READY"])
        ok, msg = svc.test_connection()
        self.assertTrue(ok, msg)
        self.assertEqual(svc._client.models.calls, 3)
        self.assertEqual(len(svc.sleeps), 2)

    def test_exponential_backoff_delays_increase(self):
        svc = make_service([server_err(503)])
        svc.test_connection()
        self.assertEqual(len(svc.sleeps), MAX_RETRIES)
        base = [2.0, 4.0, 8.0]
        for delay, expected in zip(svc.sleeps, base):
            self.assertGreaterEqual(delay, expected)
            self.assertLess(delay, expected + 0.6)

    def test_max_three_retries_then_friendly_message(self):
        svc = make_service([server_err(503)])
        ok, msg = svc.test_connection()
        self.assertFalse(ok)
        self.assertEqual(svc._client.models.calls, 1 + MAX_RETRIES)  # 1 try + 3 retries
        self.assertIn("busy", msg.lower())
        self.assertNotIn("Traceback", msg)
        self.assertNotIn("gemini", msg.lower())

    def test_rate_limit_exhaustion_gives_friendly_message(self):
        svc = make_service([client_err(429, "RESOURCE_EXHAUSTED")])
        ok, msg = svc.test_connection()
        self.assertFalse(ok)
        self.assertEqual(svc._client.models.calls, 1 + MAX_RETRIES)
        self.assertIn("busy", msg.lower())

    def test_invalid_key_is_not_retried(self):
        svc = make_service([client_err(400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key.")])
        ok, msg = svc.test_connection()
        self.assertFalse(ok)
        self.assertEqual(svc._client.models.calls, 1)
        self.assertEqual(svc.sleeps, [])
        self.assertIn("configuration", msg.lower())

    def test_permission_and_not_found_not_retried(self):
        for code, status in ((403, "PERMISSION_DENIED"), (404, "NOT_FOUND")):
            svc = make_service([client_err(code, status)])
            ok, _ = svc.test_connection()
            self.assertFalse(ok)
            self.assertEqual(svc._client.models.calls, 1)

    def test_quiz_and_flashcards_report_busy_message(self):
        svc = make_service([server_err(503)])
        items, err = svc.generate_quiz("Some document text.")
        self.assertEqual(items, [])
        self.assertIn("busy", err.lower())
        items, err = make_service([server_err(503)]).generate_flashcards("Some document text.")
        self.assertEqual(items, [])
        self.assertIn("busy", err.lower())


class TestSummarizationEfficiency(unittest.TestCase):
    def setUp(self):
        ai_engine._SUMMARY_CACHE.clear()

    def test_small_document_uses_single_call(self):
        svc = make_service(["stub-summary"])
        out = svc.generate_summary("word " * 500, "Short", "English")
        self.assertEqual(out, "stub-summary")
        self.assertEqual(svc._client.models.calls, 1)

    def test_repeat_request_served_from_cache(self):
        svc = make_service(["stub-summary"])
        text = "alpha beta gamma. " * 200
        svc.generate_summary(text, "Medium", "English")
        svc.generate_summary(text, "Medium", "English")
        self.assertEqual(svc._client.models.calls, 1)

    def test_large_document_is_chunked_with_bounded_calls(self):
        # ~ 400k chars => many sections, capped at MAX_CHUNKS + 1 final call
        text = "\n\n".join(f"Paragraph {i}. " + ("lorem ipsum " * 60) for i in range(600))
        self.assertGreater(len(text), SINGLE_PASS_CHAR_LIMIT)
        svc = make_service(["stub-notes"])
        progress = []
        out = svc.generate_summary(text, "Medium", "English", progress_callback=lambda s, t, m: progress.append((s, t, m)))
        self.assertTrue(out.startswith("stub-notes"))
        calls = svc._client.models.calls
        self.assertLessEqual(calls, MAX_CHUNKS + 1)
        self.assertGreaterEqual(calls, 3)
        self.assertTrue(progress and progress[-1][0] == progress[-1][1])

    def test_failure_in_large_document_returns_friendly_message(self):
        text = "\n\n".join("lorem ipsum " * 100 for _ in range(300))
        svc = make_service([server_err(503)])
        out = svc.generate_summary(text, "Short", "English")
        self.assertIn("busy", out.lower())

    def test_empty_document(self):
        svc = make_service(["x"])
        self.assertIn("no readable text", svc.generate_summary("   "))
        self.assertEqual(svc._client.models.calls, 0)


class TestTextHelpers(unittest.TestCase):
    def test_split_respects_limit_and_loses_nothing(self):
        text = "\n\n".join(f"para{i} " + "x" * 500 for i in range(200))
        sections = split_into_sections(text, 5000)
        self.assertTrue(all(len(s) <= 5000 for s in sections))
        self.assertEqual(sum(s.count("para") for s in sections), 200)

    def test_split_handles_giant_single_paragraph(self):
        text = "word " * 50000
        sections = split_into_sections(text, 10000)
        self.assertTrue(all(len(s) <= 10000 for s in sections))
        self.assertGreater(len(sections), 20)

    def test_sample_text_covers_whole_document(self):
        text = "START " + "a " * 40000 + "MIDDLE " + "b " * 40000 + "END"
        out = sample_text(text, 2000)
        self.assertLessEqual(len(out), 2200)
        self.assertIn("START", out)
        self.assertIn("END", out)


class TestKeyHandling(unittest.TestCase):
    def test_key_comes_from_environment_when_no_secret(self):
        with mock.patch.object(ai_engine, "_read_streamlit_secret", return_value=None), \
             mock.patch.dict("os.environ", {"GEMINI_API_KEY": "env-key"}, clear=False):
            self.assertEqual(ai_engine.resolve_api_key(), "env-key")

    def test_secrets_take_priority_over_environment(self):
        def fake(name):
            return "secret-key" if name == "GEMINI_API_KEY" else None
        with mock.patch.object(ai_engine, "_read_streamlit_secret", side_effect=fake), \
             mock.patch.dict("os.environ", {"GEMINI_API_KEY": "env-key"}, clear=False):
            self.assertEqual(ai_engine.resolve_api_key(), "secret-key")

    def test_model_from_secrets(self):
        with mock.patch.object(ai_engine, "_read_streamlit_secret", side_effect=lambda n: "custom-model" if n == "GEMINI_MODEL" else None):
            self.assertEqual(ai_engine.resolve_model(), "custom-model")

    def test_resolve_api_key_takes_no_ui_argument(self):
        with self.assertRaises(TypeError):
            ai_engine.resolve_api_key("typed-in-ui")


if __name__ == "__main__":
    unittest.main()
