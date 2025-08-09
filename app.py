import os
import uuid
import time
from typing import Optional

import streamlit as st
import pandas as pd
import requests

from io import StringIO
from pypdf import PdfReader

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- SUPABASE ----------
from supabase import create_client, Client

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="AI Job Match", layout="wide")
st.title("🤖 AI-Powered Job Matching Dashboard")
st.write("This is a test version with Hippolyte's CV injected directly.")

# ---------- SECRETS ----------
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "")

@st.cache_resource(show_spinner=False)
def get_supabase() -> Optional[Client]:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        return None

sb = get_supabase()

# A stable per-session candidate id
if "candidate_id" not in st.session_state:
    st.session_state["candidate_id"] = str(uuid.uuid4())

# ---------- DEFAULT CV (fallback) ----------
DEFAULT_CV = """
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
"""

# ======================================================
# Sidebar – sources, filters, controls
# ======================================================
with st.sidebar:
    st.header("Settings")
    source = st.selectbox("Job source", ["Arbeitnow"], index=0)
    top_n = st.slider("How many matches to show", min_value=5, max_value=50, value=20, step=1)
    location_filter = st.text_input("Filter by location (optional)", value="")
    keyword_filter = st.text_input("Filter by keyword in title/desc (optional)", value="")
    st.caption("Results are cached for performance. Use filters to refine quickly.")

# ======================================================
# CV Upload (CSV or PDF)
# ======================================================
st.subheader("1) Upload CV (CSV or PDF)")
uploaded = st.file_uploader("Upload your CV (CSV with a 'text' column or PDF). Leave empty to use the default CV.", type=["csv", "pdf"])

def load_cv_text(file) -> str:
    if file is None:
        return DEFAULT_CV.strip()

    if file.name.lower().endswith(".csv"):
        try:
            df = pd.read_csv(file)
            if "text" in df.columns and not df["text"].dropna().empty:
                return "\n".join(df["text"].dropna().astype(str).tolist())[:50000]
            else:
                # If no 'text' column, join everything
                return "\n".join(
                    " ".join(map(str, row.dropna().tolist()))
                    for _, row in df.iterrows()
                )[:50000]
        except Exception as e:
            st.warning(f"Could not read CSV: {e}. Using default CV.")
            return DEFAULT_CV.strip()

    if file.name.lower().endswith(".pdf"):
        try:
            reader = PdfReader(file)
            pages = []
            for p in reader.pages:
                pages.append(p.extract_text() or "")
            text = "\n".join(pages)
            return (text or DEFAULT_CV).strip()[:50000]
        except Exception as e:
            st.warning(f"Could not read PDF: {e}. Using default CV.")
            return DEFAULT_CV.strip()

    return DEFAULT_CV.strip()

cv_text = load_cv_text(uploaded)

# ======================================================
# Fetch jobs (Arbeitnow) – cached + retry + dedupe
# ======================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_arbeitnow() -> pd.DataFrame:
    jobs = []
    page = 1
    while True:
        url = f"https://www.arbeitnow.com/api/job-board-api?page={page}"
        ok = False
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=10)
                if r.status_code == 200:
                    data = r.json()
                    jobs.extend(data.get("data", []))
                    ok = True
                    if not data.get("links", {}).get("next"):
                        break
                    page += 1
                    time.sleep(0.2)
                else:
                    time.sleep(0.8)
            except Exception:
                time.sleep(0.8)
        if not ok:
            break
        if not data.get("links", {}).get("next"):
            break

    if not jobs:
        return pd.DataFrame()

    df = pd.DataFrame(jobs)
    # Harmonize columns
    df = df.rename(columns={
        "title": "job_title",
        "description": "job_description",
        "location": "location",
        "url": "url",
        "company_name": "company_name",
    })
    # Dedupe by URL or title+company
    if "url" in df.columns:
        df = df.drop_duplicates(subset=["url"], keep="first")
    if {"job_title", "company_name"} <= set(df.columns):
        df = df.drop_duplicates(subset=["job_title", "company_name"], keep="first")
    return df.reset_index(drop=True)

# ======================================================
# Matching via TF-IDF (fast & safe for Streamlit Cloud)
# ======================================================
def compute_matches(cv_text: str, df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    for col in ["job_title", "job_description", "company_name", "location", "url"]:
        if col not in df.columns:
            df[col] = ""

    # Apply filters
    if location_filter:
        df = df[df["location"].fillna("").str.contains(location_filter, case=False, na=False)]
    if keyword_filter:
        mask = (
            df["job_title"].fillna("").str.contains(keyword_filter, case=False, na=False) |
            df["job_description"].fillna("").str.contains(keyword_filter, case=False, na=False)
        )
        df = df[mask]

    if df.empty:
        return df

    combined = (df["job_title"].fillna("") + " " + df["job_description"].fillna("")).tolist()
    corpus = [cv_text] + combined  # index 0 is CV
    tfidf = TfidfVectorizer(min_df=2, max_df=0.9, stop_words="english")
    try:
        X = tfidf.fit_transform(corpus)
    except ValueError:
        # Very small corpus edge case – fall back to raw strings
        df["match_score"] = 0.0
        return df.head(top_n)

    sims = cosine_similarity(X[0:1], X[1:]).flatten()
    df["match_score"] = sims
    df = df.sort_values("match_score", ascending=False)
    return df.head(top_n)

# ======================================================
# Supabase writes
# ======================================================
def ensure_candidate(sb: Client, candidate_id: str, cv_text: str):
    try:
        # Upsert candidate (by id)
        sb.table("candidates").upsert({
            "id": candidate_id,
            "cv_text": cv_text[:60000]
        }).execute()
    except Exception as e:
        st.warning(f"Could not upsert candidate: {e}")

def log_interest(sb: Client, candidate_id: str, row: pd.Series, source: str):
    try:
        payload = {
            "candidate_id": candidate_id,
            "job_title": str(row.get("job_title", ""))[:500],
            "company": str(row.get("company_name", ""))[:300],
            "url": str(row.get("url", row.get("job_url", "")))[:1000],
            "location": str(row.get("location", ""))[:300],
            "source": source,
            "match_score": float(row.get("match_score", 0.0))
        }
        sb.table("interests").insert(payload).execute()
        st.toast("Saved your interest ✅", icon="✅")
    except Exception as e:
        st.error(f"Could not save interest: {e}")

# ======================================================
# MAIN
# ======================================================
with st.spinner("🔍 Fetching jobs and computing matches…"):
    if source == "Arbeitnow":
        jobs_df = fetch_arbeitnow()
    else:
        jobs_df = pd.DataFrame()

    if jobs_df.empty:
        st.warning("No jobs returned from the source right now. Try again later or adjust filters.")
        st.stop()

    top_matches = compute_matches(cv_text, jobs_df, top_n)

st.success("✅ Matching complete!")
st.write("Here are top matches:")

# Ensure candidate exists in DB
if sb:
    ensure_candidate(sb, st.session_state["candidate_id"], cv_text)

for i, row in top_matches.reset_index(drop=True).iterrows():
    st.markdown(f"### {row['job_title']} @ {row.get('company_name', 'Unknown')}")
    st.markdown(
        f"📍 {row.get('location','N/A')} &nbsp;&nbsp;|&nbsp;&nbsp; 🏷️ {source} "
        f"&nbsp;&nbsp;|&nbsp;&nbsp; 🔢 Match: **{round(100*row.get('match_score',0))}%**",
        unsafe_allow_html=True
    )
    url = row.get("url", row.get("job_url", ""))
    if url:
        st.markdown(f"[🔗 View Job Posting]({url})", unsafe_allow_html=True)

    cols = st.columns(2)
    with cols[0]:
        if st.button("✅ I'm Interested", key=f"interest_{i}"):
            if sb:
                log_interest(sb, st.session_state["candidate_id"], row, source)
            else:
                st.error("Supabase is not configured (missing secrets).")
    with cols[1]:
        st.caption((row.get("job_description") or "")[:200] + ("…" if len((row.get("job_description") or "")) > 200 else ""))
    st.divider()



