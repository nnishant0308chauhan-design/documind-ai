"""
Utility and Helper Functions for DocuMind AI
Provides CSS styling, export formatters, sample documents for instant testing,
and session state initialization.
"""

import os
import base64
from typing import Dict, Any, Tuple
import streamlit as st

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def logo_path(kind: str = "favicon") -> str:
    """Absolute path to a logo asset (favicon.png / logo.png / logo.svg)."""
    return os.path.join(ASSETS_DIR, {"favicon": "favicon.png", "png": "logo.png", "svg": "logo.svg"}[kind])


def logo_data_uri() -> str:
    """Logo as an inline data URI so it renders anywhere without static file hosting."""
    try:
        with open(logo_path("svg"), "rb") as f:
            return "data:image/svg+xml;base64," + base64.b64encode(f.read()).decode("ascii")
    except OSError:
        return ""



# Sample documents for 1-click beginner testing
SAMPLE_DOCUMENTS = {
    "Artificial Intelligence in Healthcare": """# ARTIFICIAL INTELLIGENCE IN MODERN HEALTHCARE: ADVANCEMENTS AND CHALLENGES
By Dr. Elena Vance and Dr. Marcus Chen, Institute of Medical Informatics (2025)

## 1. Executive Summary
Artificial Intelligence (AI) and Machine Learning (ML) are radically transforming contemporary medical diagnosis, clinical treatment planning, and pharmaceutical drug discovery. By analyzing vast multi-modal datasets—including radiological imaging, genomic sequencing, electronic health records (EHRs), and real-time biometric telemetry—AI models are enabling personalized precision medicine while drastically reducing diagnostic turnaround times.

## 2. Key Diagnostic Breakthroughs
### Deep Learning in Medical Imaging
Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) have achieved diagnostic parity with senior radiologists in detecting early-stage malignancies. Specifically, in diabetic retinopathy screening and mammographic density assessment, modern deep models have demonstrated sensitivity metrics exceeding 94.5% with specificity of 91.2%. Automated diagnostic systems are now approved in over 40 jurisdictions for primary triage in emergency radiology.

### Genomic Sequencing and Oncology
Next-generation sequencing paired with transformer architectures allows oncologists to identify rare genetic mutations within hours rather than weeks. Targeted molecular therapies can now be matched to patient tumor profiles with high statistical efficacy, markedly improving 5-year survival rates in non-small cell lung cancer and metastatic melanoma.

## 3. Natural Language Processing in Clinical Workflows
Clinicians spend up to 35% of their working hours on administrative documentation. Large Language Models (LLMs) tuned on biomedical corpora (such as BioBERT and Med-PaLM) can synthesize patient histories, generate structured clinical summaries from unconstrained physician-patient consultations, and draft prior-authorization requests. Studies at Massachusetts General Hospital demonstrated a 42% reduction in clinician administrative burden following the deployment of ambient clinical documentation assistants in late 2024.

## 4. Ethical Challenges, Algorithmic Bias, and Governance
Despite rapid adoption, medical AI faces significant socioeconomic and technical challenges:
1. Algorithmic Bias: Models trained predominantly on Caucasian demographic datasets frequently exhibit reduced diagnostic accuracy when evaluated on underrepresented ethnic populations.
2. Data Privacy: Compliance with HIPAA in the United States and GDPR in the European Union mandates rigorous de-identification of Protected Health Information (PHI) before model training.
3. The "Black Box" Dilemma: Deep neural networks often lack mechanistic interpretability. Clinicians cannot ethically accept diagnostic recommendations without understandable rationales, driving current research into Explainable AI (XAI).

## 5. Conclusion and Strategic Outlook
The future of healthcare AI is collaborative, pairing human clinical expertise with computational precision. Rather than replacing medical practitioners, AI serves as a cognitive multiplier, democratizing high-quality diagnostic intelligence across resource-constrained rural and global clinics.""",

    "Renewable Energy & Climate Solutions": """# GLOBAL TRANSITION TO RENEWABLE ENERGY SYSTEMS (2025-2035)
International Clean Tech Advisory Council

## 1. Introduction
The global energy sector accounts for roughly 73% of anthropogenic greenhouse gas emissions. Meeting the Paris Agreement target of limiting global warming to 1.5 degrees Celsius requires an unprecedented transition from fossil hydrocarbons (coal, oil, and natural gas) to renewable and carbon-neutral generation sources by 2050.

## 2. Solar Photovoltaics and Wind Energy
Solar Photovoltaic (PV) technology has witnessed a dramatic 89% cost decline over the past decade, making levelized cost of energy (LCOE) for utility-scale solar cheaper than new coal-fired generation across 85% of world regions. Simultaneously, offshore wind turbine capacities have reached 15 megawatts per unit, capturing higher and more consistent oceanic wind speeds.

## 3. Energy Storage and Grid Modernization
The primary technical bottleneck for renewable penetration is resource intermittency: solar generation peaks during midday, while wind speeds fluctuate seasonally.
Key solutions include:
- Lithium Iron Phosphate (LFP) Batteries: Dominant in short-duration stationary storage (2 to 6 hours), offering superior thermal stability and lower raw material costs compared to nickel-cobalt chemistries.
- Sodium-ion Batteries: Emerging commercial alternatives that replace scarce lithium with ubiquitous sodium salts, particularly attractive for grid-scale bulk storage.
- Long-Duration Energy Storage (LDES): Pumped hydroelectric storage and green hydrogen electrolysis (using surplus renewable electricity to split water into hydrogen and oxygen for long-term fuel cells).

## 4. Smart Grids and Decentralization
Modernizing legacy electrical grids into bidirectional 'Smart Grids' leverages edge computing, IoT sensors, and dynamic demand-response algorithms. Virtual Power Plants (VPPs) aggregate thousands of residential rooftop solar arrays, domestic battery banks, and electric vehicle (EV) charging stations, dynamically stabilizing grid frequency during peak load spikes.

## 5. Policy Imperatives
Accelerating the clean energy transition requires three synchronized policy levers:
1. Carbon pricing mechanisms to internalize environmental externalities.
2. Expedited transmission line permitting to connect remote renewable zones to urban demand centers.
3. Subsidies and workforce training initiatives for just transitions in legacy fossil fuel communities."""
}


def load_custom_css():
    """Injects custom CSS to create a clean, modern, cohesive dashboard design."""
    st.markdown(
        """
        <style>
        /* Import clean modern font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Top header bar accent */
        .main-header {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            padding: 24px 30px;
            border-radius: 14px;
            color: white;
            margin-bottom: 25px;
            box-shadow: 0 4px 15px rgba(79, 70, 229, 0.15);
        }
        .main-header h1 {
            color: white !important;
            margin: 0;
            font-size: 2.1rem;
            font-weight: 700;
        }
        .main-header p {
            color: #e0e7ff !important;
            margin: 6px 0 0 0;
            font-size: 1.05rem;
        }

        /* Metric cards */
        .metric-card {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 16px 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }
        .metric-title {
            color: #6b7280;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .metric-value {
            color: #111827;
            font-size: 1.6rem;
            font-weight: 700;
            margin-top: 4px;
        }

        /* Feature action cards on Dashboard */
        .feature-card {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 22px;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-shadow: 0 2px 4px rgba(0,0,0,0.04);
            border-top: 4px solid #4f46e5;
        }
        .feature-card h3 {
            margin: 0 0 8px 0;
            font-size: 1.15rem;
            color: #1f2937;
        }
        .feature-card p {
            color: #4b5563;
            font-size: 0.92rem;
            line-height: 1.5;
            margin-bottom: 12px;
        }

        /* Badge pills */
        .badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.78rem;
            font-weight: 600;
        }
        .badge-blue { background-color: #dbeafe; color: #1e40af; }
        .badge-green { background-color: #d1fae5; color: #065f46; }
        .badge-amber { background-color: #fef3c7; color: #92400e; }
        .badge-purple { background-color: #ede9fe; color: #5b21b6; }

        /* Sidebar enhancement */
        [data-testid="stSidebar"] {
            background-color: #f9fafb;
            border-right: 1px solid #e5e7eb;
        }

        /* Improve button styling */
        .stButton button {
            border-radius: 8px;
            font-weight: 500;
        }

        /* ---------- Hide developer chrome (Deploy button, menu, footer) ---------- */
        [data-testid="stAppDeployButton"], .stDeployButton, #MainMenu,
        [data-testid="stMainMenu"], [data-testid="stToolbarActions"], footer,
        [data-testid="stDecoration"] { display: none !important; visibility: hidden !important; }
        header[data-testid="stHeader"] { background: transparent; }

        /* ---------- UI polish ---------- */
        .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }
        h2 { font-weight: 700; letter-spacing: -0.01em; }
        h3 { font-weight: 600; }

        .main-header {
            position: relative; overflow: hidden;
            box-shadow: 0 8px 24px rgba(79, 70, 229, 0.22);
        }
        .main-header::after {
            content: ""; position: absolute; right: -40px; top: -40px;
            width: 180px; height: 180px; border-radius: 50%;
            background: rgba(255,255,255,0.10);
        }

        .feature-card, .metric-card { transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease; }
        .feature-card:hover { transform: translateY(-3px); box-shadow: 0 10px 22px rgba(79,70,229,0.12); border-color: #c7d2fe; }
        .metric-card { border-left: 4px solid #7c3aed; }

        /* Buttons */
        .stButton button, .stDownloadButton button {
            border-radius: 10px; font-weight: 600;
            transition: transform .12s ease, box-shadow .12s ease;
        }
        .stButton button:hover, .stDownloadButton button:hover {
            transform: translateY(-1px); box-shadow: 0 4px 12px rgba(79,70,229,0.18);
        }
        .stButton button[kind="primary"] {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); border: none;
        }

        /* Sidebar navigation as pills */
        [data-testid="stSidebar"] .stRadio > div { gap: 4px; }
        [data-testid="stSidebar"] .stRadio label {
            padding: 8px 12px; border-radius: 10px; width: 100%;
            transition: background .15s ease;
        }
        [data-testid="stSidebar"] .stRadio label:hover { background: #eef2ff; }
        [data-testid="stSidebar"] .stRadio label:has(input:checked) {
            background: #e0e7ff; font-weight: 600; color: #3730a3;
        }
        [data-testid="stSidebar"] .stRadio label > div:first-child { display: none; }

        /* Tabs, inputs, progress, chat */
        .stTabs [data-baseweb="tab-list"] { gap: 6px; }
        .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: 8px 16px; font-weight: 600; }
        .stTextArea textarea, .stTextInput input, .stSelectbox > div > div { border-radius: 10px; }
        .stProgress > div > div > div > div { background: linear-gradient(90deg, #4f46e5, #7c3aed); }
        [data-testid="stChatMessage"] {
            border: 1px solid #e5e7eb; border-radius: 14px; padding: 12px 16px;
            background: #ffffff; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }
        [data-testid="stMetric"] {
            background: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }
        [data-testid="stExpander"] { border-radius: 12px; border-color: #e5e7eb; }
        [data-testid="stFileUploader"] section {
            border: 2px dashed #c7d2fe; border-radius: 14px; background: #f5f7ff;
        }
        [data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"] .result-anchor) {
            background: #f8fafc; border-radius: 14px;
        }
        ::-webkit-scrollbar { width: 10px; height: 10px; }
        ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 8px; }

        /* ---------- Brand + hero ---------- */
        .brand { display: flex; align-items: center; gap: 12px; padding: 4px 2px 2px 2px; }
        .brand img { width: 46px; height: 46px; filter: drop-shadow(0 4px 8px rgba(79,70,229,.35)); }
        .brand-name {
            font-size: 1.35rem; font-weight: 800; letter-spacing: -0.02em; line-height: 1.1;
            background: linear-gradient(90deg, #4338ca, #7c3aed 55%, #db2777);
            -webkit-background-clip: text; background-clip: text; color: transparent;
        }
        .brand-tag { font-size: .74rem; color: #6b7280; margin-top: 2px; }

        .hero {
            display: flex; align-items: center; gap: 26px; position: relative; overflow: hidden;
            background: linear-gradient(135deg, #312e81 0%, #4f46e5 45%, #7c3aed 100%);
            border-radius: 20px; padding: 30px 34px; margin-bottom: 26px; color: #fff;
            box-shadow: 0 14px 34px rgba(79,70,229,.28);
        }
        .hero::before {
            content: ""; position: absolute; right: -60px; top: -70px; width: 260px; height: 260px;
            border-radius: 50%; background: radial-gradient(circle, rgba(236,72,153,.45), transparent 70%);
        }
        .hero::after {
            content: ""; position: absolute; left: 38%; bottom: -90px; width: 220px; height: 220px;
            border-radius: 50%; background: radial-gradient(circle, rgba(255,255,255,.14), transparent 70%);
        }
        .hero img { width: 96px; height: 96px; flex-shrink: 0; z-index: 1; filter: drop-shadow(0 8px 18px rgba(0,0,0,.35)); }
        .hero-body { z-index: 1; }
        .hero h1 { color: #fff !important; margin: 0 0 6px 0; font-size: 2.2rem; font-weight: 800; letter-spacing: -0.02em; }
        .hero p { color: #e0e7ff !important; margin: 0 0 14px 0; font-size: 1.05rem; max-width: 640px; }
        .chip {
            display: inline-block; margin: 0 8px 6px 0; padding: 5px 13px; border-radius: 999px;
            background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.28);
            font-size: .8rem; font-weight: 600; color: #fff; backdrop-filter: blur(4px);
        }
        @media (max-width: 700px) { .hero { flex-direction: column; align-items: flex-start; padding: 22px; } .hero h1 { font-size: 1.7rem; } }

        .metric-card { min-height: 212px; }
        .feature-card { min-height: 160px; }
        .sidebar-foot { color: #9ca3af; font-size: .72rem; text-align: center; padding-top: 6px; }
        [data-testid="stSidebar"] .stRadio label:has(input:checked) { box-shadow: inset 4px 0 0 #7c3aed; }
        [data-testid="stSidebar"] .stRadio label { cursor: pointer; }
        .section-title { font-size: 1.25rem; font-weight: 700; margin: 6px 0 2px 0; }
        </style>
        """,
        unsafe_allow_html=True
    )


def init_session_state():
    """Initializes all required session state variables with clean defaults."""
    defaults = {
        "active_page": "Dashboard",
        "processed_doc": None,
        "rag_engine": None,
        "chat_history": [],
        "summary_text": "",
        "summary_type": "Medium",
        "summary_lang": "English",
        "keywords_text": "",
        "quiz_data": [],
        "quiz_user_answers": {},
        "quiz_submitted": False,
        "flashcards_data": [],
        "flashcard_index": 0,
        "flashcard_show_answer": False,
        "eli5_concept": "",
        "eli5_result": "",
        "temperature": 0.2
    }

    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def reset_document_session():
    """Clears the currently loaded document and all derived AI results."""
    st.session_state["processed_doc"] = None
    st.session_state["rag_engine"] = None
    st.session_state["chat_history"] = []
    st.session_state["summary_text"] = ""
    st.session_state["keywords_text"] = ""
    st.session_state["quiz_data"] = []
    st.session_state["quiz_user_answers"] = {}
    st.session_state["quiz_submitted"] = False
    st.session_state["flashcards_data"] = []
    st.session_state["flashcard_index"] = 0
    st.session_state["flashcard_show_answer"] = False
    st.session_state["eli5_result"] = ""


def format_summary_for_download(doc_name: str, summary_type: str, language: str, summary_text: str) -> str:
    """Formats summary for file download."""
    return f"""============================================================
DocuMind AI - Document Summary
============================================================
Source Document: {doc_name}
Summary Type:    {summary_type}
Language:        {language}
Generated By:    DocuMind AI
============================================================

{summary_text}
"""


def format_keywords_for_download(doc_name: str, keywords_text: str) -> str:
    """Formats keywords extraction for file download."""
    return f"""============================================================
DocuMind AI - Key Concepts, Definitions & Information
============================================================
Source Document: {doc_name}
Generated By:    DocuMind AI
============================================================

{keywords_text}
"""


def format_eli5_for_download(doc_name: str, concept: str, explanation: str) -> str:
    """Formats concept explanation for file download."""
    return f"""============================================================
DocuMind AI - Concept Explanation (Beginner Friendly)
============================================================
Source Document: {doc_name}
Target Concept:  {concept}
Generated By:    DocuMind AI
============================================================

{explanation}
"""
