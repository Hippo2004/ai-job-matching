import os
import time
import json
import numpy as np
import pandas as pd
import streamlit as st
import requests

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# =========================
# Streamlit page config
# =========================
st.set_page_config(page_title="AI Job Match", layout="wide")
st.title("🤖 AI-Powered Job Matching Dashboard")
st.write("This is a test version with Hippolyte's CV injected directly.")

# =========================
# Robust HTTP session
# =========================
def make_session():
    s = requests.Session()
    retries = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; hippo2004-job-matcher/0.1; +https://streamlit.app)"
    })
    return s

SESSION = make_session()

# =========================
# Candidate profile (Phase 1: injected)
# =========================
CV_TEXT = """
Hippolyte Guermonprez
Currently seeking a 6-month internship in Private Jet Charter Sales in Switzerland, with the goal of converting it to a full-time role. Experienced in business development, customer service, and international environments.

Experience:
- Lemonway – Paris – Fintech – Business Development (Intern)
- Autonomos – Paris – Account Manager – 2023
- Décathlon – Sales Associate – 2022

Education:
- Paris School of Business – Bachelor's in Business
- Languages: French (native), English (fluent), German (conversational)

Skills:
- B2B Sales
- CRM & Outreach Tools (Lemlist, HubSpot)
- Cold Emailing, Lead Generation
- Client Management & Communication
""".strip()

# =========================
# Optional ST embedder (fallback to TF-IDF)
# =========================
@st.cache_resource(show_spinner=False)
def get_embedder(prefer_tfidf: bool = False):
    if prefer_tfidf:
        return ("tfidf", None)

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        return ("st", model)
    except Exception as e:
        st.info("Using TF-IDF fallback (Sentence-Transformers unavailable).")
        return ("tfidf", None)

# =========================
# Data fetchers (with caching)
# =========================
@st.cache_data(ttl=1800, show_spinner=False)  # 30 minutes
def fetch_remotive():
    url = "https://remotive.io/api/remote-jobs"
    try:
        r = SESSION.get(url, timeout=15)
        if r.status_code != 200:
            raise RuntimeError(f"{r.status_code} {r.reason}")
        data = r.json().get("jobs", [])
        df = pd.json_normalize(data)
        # Normalize fields
        df = df.rename(columns={
            "title": "title",
            "company_name": "company",
            "candidate_required_location": "location",
            "description": "description",
            "url": "url"
        })
        df["source"] = "Remotive"
        needed = ["title", "company", "location", "description", "url", "source"]
        return df[needed].dropna(subset=["title", "description"])
    except Exception as e:
        return pd.DataFrame(), f"Remotive error: {e}"

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_arbeitnow():
    all_rows = []
    page = 1
    try:
        while True:
            url = f"https://www.arbeitnow.com/api/job-board-api?page={page}"
            r = SESSION.get(url, timeout=15)
            if r.status_code != 200:
                break
            payload = r.json()
            rows = payload.get("data", [])
            if not rows:
                break
            all_rows.extend(rows)
            if not payload.get("links", {}).get("next"):
                break
            page += 1

        if not all_rows:
            return pd.DataFrame(), None

        df = pd.DataFrame(all_rows)
        df = df.rename(columns={
            "title": "title",
            "company_name": "company",
            "location": "location",
            "description": "description",
            "url": "url"
        })
        df["source"] = "Arbeitnow"
        needed = ["title", "company", "location", "description", "url", "source"]
        return df[needed].dropna(subset=["title", "description"]), None
    except Exception as e:
        return pd.DataFrame(), f"Arbeitnow error: {e}"

# Small static fallback so the UI never looks empty
def fallback_jobs():
    return pd.DataFrame([
        {
            "title": "Investment Intern (m/f/d)",
            "company": "Signature Ventures GmbH",
            "location": "Munich / Remote",
            "description": "Support investment team, deal flow, analysis, and portfolio research.",
            "url": "https://example.com/job/investment-intern",
            "source": "Sample"
        },
        {
            "title": "Social Media / Content Intern (BENELUX)",
            "company": "KoRo Handels GmbH",
            "location": "Berlin / Remote",
            "description": "Help plan, create, and publish content. Community engagement, analytics.",
            "url": "https://example.com/job/social-media-intern",
            "source": "Sample"
        }
    ])

# =========================
# Matching
# =========================
def score_with_st(model, texts):
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    cv = embs[0]
    jobs = np.array(embs[1:])
    scores = jobs @ cv  # cosine because normalized
    return scores

def score_with_tfidf(texts):
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
    X = vec.fit_transform(texts)
    cv = X[0]
    jobs = X[1:]
    scores = cosine_similarity(jobs, cv).ravel()
    return scores

def compute_match_scores(cv_text: str, df: pd.DataFrame, method, model):
    df = df.copy()
    df["combined"] = (df["title"].fillna("") + " " + df["description"].fillna("")).str.strip()
    texts = [cv_text] + df["combined"].tolist()

    if method == "st":
        try:
            raw = score_with_st(model, texts)
        except Exception:
            # fall back if model errors
            raw = score_with_tfidf(texts)
    else:
        raw = score_with_tfidf(texts)

    # scale 0..1 for display
    scaler = MinMaxScaler()
    df["match_score"] = scaler.fit_transform(raw.reshape(-1, 1))
    return df.sort_values("match_score", ascending=False)

def top_keywords(texts, n=12):
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=8000)
    X = vec.fit_transform(texts)
    sums = np.asarray(X.sum(axis=0)).ravel()
    terms = np.array(vec.get_feature_names_out())
    idx = np.argsort(-sums)[:n]
    return terms[idx].tolist()

# =========================
# UI controls
# =========================
st.sidebar.subheader("Settings")
use_sources = st.sidebar.multiselect(
    "Job sources", ["Remotive", "Arbeitnow"], default=["Remotive", "Arbeitnow"]
)
safe_mode = st.sidebar.toggle("Safe mode (use TF-IDF only)", value=False)
top_k = st.sidebar.slider("How many matches to show", 5, 50, 20, 5)

method, model = get_embedder(prefer_tfidf=safe_mode)

# =========================
# Fetch data
# =========================
frames = []
alerts = []

if "Remotive" in use_sources:
    remotive_df, err = fetch_remotive()
    if err:
        alerts.append(err)
    if not remotive_df.empty:
        frames.append(remotive_df)

if "Arbeitnow" in use_sources:
    arbeit_df, err = fetch_arbeitnow()
    if err:
        alerts.append(err)
    if not arbeit_df.empty:
        frames.append(arbeit_df)

if alerts:
    for a in alerts:
        st.warning(f"⚠️ {a}")

if frames:
    all_jobs = pd.concat(frames, ignore_index=True)
else:
    st.error("No jobs returned right now. Showing a small sample so you can test the flow.")
    all_jobs = fallback_jobs()

# =========================
# Match + Display
# =========================
with st.spinner("🔎 Scoring matches..."):
    results = compute_match_scores(CV_TEXT, all_jobs, method, model)
    top = results.head(top_k)

st.success("✅ Matching complete!")

if top.empty:
    st.info("No results to show.")
else:
    st.write("Here are top matches:")
    for i, row in top.reset_index(drop=True).iterrows():
        st.markdown(f"### {row['title']} @ {row.get('company', 'Unknown')}")
        st.markdown(
            f"📍 **{row.get('location', 'N/A')}**  |  🏷️ **{row.get('source', '')}**  |  💡 Match: **{int(round(row['match_score']*100))}%**"
        )
        st.markdown(f"[🔗 View Job Posting]({row.get('url', '#')})", unsafe_allow_html=True)
        st.button("✅ I’m Interested", key=f"btn_{i}")
        st.markdown("---")

    # Lightweight “improve profile” ideas
    st.subheader("💡 Improve Your Profile")
    job_keywords = top_keywords(top["description"].tolist(), n=15)
    cv_tokens = set([t for t in CV_TEXT.lower().split() if t.isalpha()])
    missing = [kw for kw in job_keywords if kw.split()[0] not in cv_tokens]
    if missing:
        st.write("These keywords are common in matched jobs but missing from your CV:")
        st.write(", ".join(missing))
    else:
        st.write("Your CV already covers most common keywords in these jobs.")

st.caption("Tip: If Remotive errors again, deselect it in the sidebar and rely on Arbeitnow.")


