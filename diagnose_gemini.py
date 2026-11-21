"""
Diagnostic Test Script for Google Gemini API in DocuMind AI
Sends the prompt: "Explain what artificial intelligence is in 3 sentences."
Inspects the response structure:
- Candidates presence
- Candidate content and text parts
- Candidate finish_reason
- Prompt feedback and safety blocking
- Response text extraction
"""

import sys
import os
from typing import Optional, Any
from dotenv import load_dotenv

# Load local .env
load_dotenv()

from ai_engine import GeminiService, resolve_api_key, DEFAULT_MODEL
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


def run_gemini_diagnostic():
    print("=" * 70)
    print("DOCUMIND AI - GEMINI API DIAGNOSTIC TEST")
    print("=" * 70)

    # 1. Resolve API Key securely (without printing it)
    key = resolve_api_key()
    if not key:
        print("\n❌ No API key found.")
        print("Add it to .streamlit/secrets.toml:")
        print('  GEMINI_API_KEY = "your_key"')
        print("(or set the GEMINI_API_KEY environment variable / .env file)")
        return False

    print("✅ API key detected (not displayed)")
    print(f"Target Model: {DEFAULT_MODEL}")
    print("Prompt: 'Explain what artificial intelligence is in 3 sentences.'")
    print("-" * 70)

    # 2. Initialize Client
    try:
        client = genai.Client(api_key=key)
    except Exception as e:
        print(f"❌ Failed to initialize genai.Client: {e}")
        return False

    # 3. Send Diagnostic Request
    prompt = "Explain what artificial intelligence is in 3 sentences."
    try:
        print("Sending request to Gemini API...")
        # Note: We test with sufficient output tokens and thinking_budget=0 for reliable fast response
        config = types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=1000,
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=prompt,
            config=config
        )
    except Exception as e:
        print(f"\n❌ API Call Failed with exception: {type(e).__name__}: {e}")
        return False

    # 4. Detailed Response Inspection
    print("\n🔍 INSPECTING GEMINI RESPONSE:")
    print("-" * 70)

    # A. Prompt Feedback
    if hasattr(response, "prompt_feedback") and response.prompt_feedback:
        block_reason = getattr(response.prompt_feedback, "block_reason", None)
        print(f"• Prompt Feedback Block Reason: {block_reason or 'None (Not blocked)'}")
    else:
        print("• Prompt Feedback: None (No prompt blocking)")

    # B. Candidates
    has_candidates = hasattr(response, "candidates") and bool(response.candidates)
    num_candidates = len(response.candidates) if has_candidates else 0
    print(f"• Candidates Present: {has_candidates} (Count: {num_candidates})")

    if not has_candidates:
        print("❌ No candidates were returned by the model.")
        return False

    candidate = response.candidates[0]

    # C. Finish Reason
    finish_reason = getattr(candidate, "finish_reason", None)
    print(f"• Candidate Finish Reason: {finish_reason}")

    # D. Safety Ratings
    safety_ratings = getattr(candidate, "safety_ratings", None)
    if safety_ratings:
        blocked_ratings = [r for r in safety_ratings if getattr(r, "blocked", False)]
        print(f"• Safety Ratings: {len(safety_ratings)} categories checked ({len(blocked_ratings)} blocked)")
    else:
        print("• Safety Ratings: Clean / None")

    # E. Content & Text Parts
    has_content = hasattr(candidate, "content") and candidate.content is not None
    print(f"• Candidate Content Present: {has_content}")

    if has_content:
        parts = getattr(candidate.content, "parts", []) or []
        print(f"• Content Parts Count: {len(parts)}")
        for i, part in enumerate(parts):
            is_thought = getattr(part, "thought", False)
            text_val = getattr(part, "text", None)
            text_len = len(text_val) if text_val else 0
            part_type = "Thought/Reasoning" if is_thought else "Text/Response"
            print(f"    - Part {i+1}: Type={part_type}, Length={text_len} chars")

    # F. response.text accessor check
    try:
        raw_text_prop = response.text
        print(f"• response.text property value: {'Present (' + str(len(raw_text_prop)) + ' chars)' if raw_text_prop else 'None or Empty'}")
    except Exception as e:
        print(f"• response.text raised exception: {e}")

    # G. Robust Extracted Text via GeminiService helper
    service = GeminiService()
    extracted_text, extract_err = service._extract_response_text(response)

    print("-" * 70)
    if extracted_text:
        print("✅ SUCCESSFUL TEXT EXTRACTION:")
        print(extracted_text)
        print("=" * 70)
        return True
    else:
        print(f"❌ EXTRACTION FAILED: {extract_err}")
        print("=" * 70)
        return False


if __name__ == "__main__":
    sys.exit(0 if run_gemini_diagnostic() else 1)
