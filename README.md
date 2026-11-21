# 🧠 DocuMind AI – Intelligent Document Summarization & Learning Assistant

**DocuMind AI** is a full-stack, AI-powered web application that turns static PDF and Text documents into interactive, dynamic learning experiences. Powered by **Google Gemini** generative AI models and an internal **Retrieval-Augmented Generation (RAG)** engine, DocuMind AI allows students, researchers, educators, and professionals to summarize documents, extract factual intelligence, ask grounded questions with page citations, generate self-testing quizzes, flip study flashcards, and simplify dense concepts into everyday English.

---

## 🌟 Key Features

### 📄 1. Document Upload & Processing
- **Format Support:** Upload digital `.pdf` and `.txt` files.
- **Smart Metadata Extraction:** Displays file name, format, total page count, word count, and character count.
- **Scanned PDF Detection:** Intelligently recognizes scanned or image-based PDFs where digital text is absent and gently alerts the user that OCR preprocessing is required.
- **Error Handling:** Robustly handles empty files, password-encrypted PDFs, unsupported formats, and corrupted files with helpful guidance.
- **1-Click Instant Demo:** Built-in curated sample documents (*"AI in Modern Healthcare"* and *"Renewable Energy Systems"*) to test all features instantly without uploading a file.

### 📝 2. AI Document Summarization
- **5 Custom Styles:**
  - **Short Summary:** 1–2 concise paragraphs capturing core premises.
  - **Medium Summary:** Balanced overview with major arguments and takeaways.
  - **Detailed Summary:** Comprehensive, in-depth breakdown of all document sections.
  - **Bullet-Point Summary:** Clean, hierarchical bullet points of all key takeaways.
  - **Section-Wise Summary:** Sequential breakdown corresponding to document chapters/headings.
- **Multilingual Support:** Generate fluent, grammatically authentic summaries in:
  - 🇬🇧 **English**
  - 🇮🇳 **Hindi (हिंदी)** in native Devanagari script
  - 🇮🇳 **Punjabi (ਪੰਜਾਬੀ)** in native Gurmukhi script
- **Word Reduction Metrics:** Live comparison of original word count vs. summary word count and reading time saved.
- **Instant Export:** Download summaries directly as `.txt` files.

### 🔍 3. Keyword and Key Information Extraction
- Automatically extracts and organizes:
  1. **Core Keywords & Themes** (with contextual explanations)
  2. **Key Definitions & Terminology**
  3. **Crucial Facts, Dates & Statistics**
  4. **Notable Names, Organizations & Entities**
  5. **Technical Glossary**
- **Strict Grounding:** Adheres strictly to the uploaded text without hallucinating external trivia.
- **Instant Export:** Download extracted insights as `.txt`.

### 💬 4. Ask Your Document (Grounded RAG Chatbot)
- **Retrieval-Augmented Generation (RAG):**
  - Chunks documents with overlap while preserving exact page numbers.
  - Computes TF-IDF / BM25 term matrices and cosine similarity rankings in pure Python/NumPy for fast, deterministic retrieval.
- **Grounded Answers:** Answers are synthesized exclusively from the most relevant retrieved excerpts.
- **No Hallucinations:** If the answer is not contained in the text, the assistant explicitly states it cannot find the information in the document.
- **Source Transparency:** Expandable accordion under every response displays the exact matching excerpts, page numbers, and relevance match scores.
- **Session History:** Maintains conversation context across queries.

### 💡 5. Explain Like I'm a Beginner ("ELI5")
- Select any difficult paragraph, technical term, or confusing formula from the document.
- DocuMind AI translates it into:
  1. **The Big Picture:** Plain, conversational English without confusing jargon.
  2. **Practical Real-World Example:** Visualizable daily life application.
  3. **Relatable Analogy:** Memorable, vivid comparison (*"Think of it like..."*).
  4. **Mini-Glossary:** Friendly definitions of tricky words.
- Download explanations as `.txt`.

### 🎯 6. Scored Interactive Quiz Generator
- Generates 3, 5, 8, or 10 multiple-choice questions (MCQs) directly from the document.
- 4 options per question (A, B, C, D).
- Interactive in-browser quiz solver:
  - Select your answers with radio buttons.
  - Instant scoring with percentage calculations.
  - Color-coded feedback: Green for correct answers; Red for incorrect answers displaying what you selected, the correct choice, and a grounded explanation.
- Retake or re-generate quizzes anytime.
- Download questions, options, and full answer keys as `.txt`.

### 📇 7. Interactive Study Flashcards
- Automatically extracts core concepts into question/prompt and answer pairs.
- **Interactive Deck View:**
  - Displays one card at a time with category badges (*Concept*, *Definition*, *Fact*).
  - Flip card interaction (*Reveal Answer* / *Hide Answer*).
  - Previous and Next navigation buttons with live progress tracking.
- **Grid View:** Toggle to browse all flashcards simultaneously for quick review.
- Download full flashcard decks as `.txt`.

---

## 🛠️ Technology Stack

| Technology | Purpose |
| :--- | :--- |
| **Python 3.10+ / 3.14** | Core programming language |
| **Streamlit** | Modern, responsive web application framework |
| **Google GenAI SDK (`google-genai`)** | Official Python SDK for Google Gemini models |
| **Google Gemini Models** | Configurable generative intelligence (`gemini-3.5-flash`, `gemini-2.5-flash`, etc.) |
| **PyPDF (`pypdf`)** | High-performance text and page metadata extraction from PDFs |
| **NumPy** | High-speed vectorized term calculation and cosine similarity for RAG |
| **python-dotenv** | Secure environment variable and credential management |

---

## 📂 Project Folder Structure

```
backend/
└── DocuMind-AI/
    ├── app.py                      # Main Streamlit web application & UI routing
    ├── document_processor.py       # PDF/TXT extraction, metadata, scanned PDF detection
    ├── rag_engine.py               # Chunking, TF-IDF / BM25 index, semantic retrieval
    ├── ai_engine.py                # GeminiService wrapper with google-genai SDK
    ├── quiz_generator.py           # Quiz state handling, scoring, UI & export formatters
    ├── flashcard_generator.py      # Flashcard flip navigation, UI & export formatters
    ├── utils.py                    # Custom CSS styling, session management, sample documents
    │
    ├── requirements.txt            # Python dependencies
    ├── README.md                   # Comprehensive documentation & beginner guide
    ├── .env.example                # Template for Gemini API key configuration
    ├── .gitignore                  # Security rules preventing API key leaks
    │
    ├── assets/                     # Static media and assets
    │
    ├── .streamlit/
    │   └── config.toml             # Streamlit visual theme, primary colors & server settings
    │
    └── tests/
        ├── test_document_processor.py  # Unit tests for document extraction & edge cases
        ├── test_rag.py                 # Unit tests for chunking, indexing & similarity
        ├── test_utils.py               # Unit tests for helpers & formatters
        └── test_integration.py         # End-to-end integration tests
```

---

## ⚙️ Installation and Setup Guide

### 1. Prerequisites
- Python **3.10 or higher** installed on your system.
- Check your Python version in terminal:
  ```bash
  python --version
  ```

### 2. Navigate to the Project Directory
```bash
cd backend\DocuMind-AI
```

### 3. Install Dependencies
Install all required libraries using `pip`:
```bash
pip install -r requirements.txt
```

---

## 🔑 Configuring Your Google Gemini API Key

DocuMind AI requires a free Google Gemini API key to run its generative AI features.

### Step 1: Obtain a Free Key
1. Visit **[Google AI Studio](https://aistudio.google.com/)**.
2. Sign in with your Google account.
3. Click the **"Get API key"** button in the left sidebar.
4. Click **"Create API key"** and copy the generated key.

### Step 2: Put the key in `.streamlit/secrets.toml`

The key is read **only by the Python backend**. It is never typed into, stored in, or shown by the web interface.

1. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`:
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml      # macOS / Linux
   copy .streamlit\secrets.toml.example .streamlit\secrets.toml    # Windows
   ```
2. Open `.streamlit/secrets.toml` and set your values:
   ```toml
   GEMINI_API_KEY = "your_actual_gemini_api_key_here"
   GEMINI_MODEL = "gemini-3.5-flash"   # optional, central model setting
   ```
3. **Restart the app**, open **⚙️ Settings**, and click **🧪 Test API Connection**.

Fallback (only if `secrets.toml` has no key): a `GEMINI_API_KEY` environment variable or `.env` file.

> **Security Note:** `.streamlit/secrets.toml` and `.env` are in `.gitignore`, so your key is never committed.

### Reliability & speed
- Temporary Gemini errors (429 / 503 / resource exhausted) are retried automatically up to **3 times** with exponential backoff (about 2s, 4s, 8s). Invalid keys and other permanent errors are never retried.
- Documents up to ~60,000 characters are summarized with a single API call; larger PDFs are condensed section by section (max 12 sections) and then summarized once. Repeat requests are cached.

---

## 🌐 Deploying Publicly (Streamlit Community Cloud)

1. Push the project to a **private or public GitHub repo**. `.streamlit/secrets.toml` and `.env` are git-ignored, so your key is not uploaded. Check with `git status` before pushing.
2. Go to **share.streamlit.io** → *New app* → pick your repo, branch and `app.py`.
3. Open **Advanced settings → Secrets** and paste:
   ```toml
   GEMINI_API_KEY = "your_actual_gemini_api_key_here"
   GEMINI_MODEL = "gemini-3.5-flash"
   ```
4. Deploy. Visitors see a clean app: no Deploy button, no developer menu, no API/model wording, and friendly messages instead of technical errors.
5. To see diagnostics yourself (connection test, model), add `DOCUMIND_ADMIN = true` to the Secrets panel, use it, then remove it again.

**Cost and abuse:** a public app spends *your* Gemini quota. Watch usage in Google AI Studio, and consider restricting the app's viewers (Streamlit Cloud → Sharing) if you only want certain people to use it.

---

## 🚀 Running the Application

To launch DocuMind AI, execute the following command in your terminal:

```bash
streamlit run app.py
```

The application will start immediately and automatically open in your default browser at:
```
http://localhost:8501
```

---

## 🧪 Running the Automated Test Suite

DocuMind AI includes a complete suite of unit and integration tests covering text extraction, scanned PDF detection, RAG retrieval accuracy, scoring logic, and formatters.

To run all tests:
```bash
python -m unittest discover -s tests -v
```

All 23 tests should pass in under 0.1 seconds.

---

## ❓ Common Errors & Troubleshooting

### 1. `Invalid API Key: The provided Google Gemini API key is incorrect...`
- **Cause:** The key has a typo, leading/trailing space, or was deleted in Google AI Studio.
- **Solution:** Go to **⚙️ Settings** in the app, re-paste your key, and click **"Test API Connection"**.

### 2. `⚠️ Scanned / Image-only PDF Detected...`
- **Cause:** The uploaded PDF contains scanned images or photos rather than selectable digital text.
- **Solution:** Upload a digital text PDF, convert the scanned PDF using an OCR utility first, or upload a `.txt` file. You can also test all features with the built-in sample documents.

### 3. `Rate Limit / Quota Exceeded (HTTP 429)`
- **Cause:** You have made many rapid requests on the free tier of the Gemini API.
- **Solution:** Wait 60 seconds before retrying. Alternatively, reduce the temperature or test with shorter excerpts.

### 4. `The file appears to be corrupted or invalid`
- **Cause:** The PDF file was interrupted during download or is password-protected.
- **Solution:** Ensure the PDF opens normally in your PDF viewer, remove any password, and re-upload.

---

## 📄 License
This project is open-source under the MIT License. Feel free to customize and extend it!
