import os
import uuid
import time
from typing import Optional, List

import streamlit as st
import pandas as pd
import requests

# ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Optional: Supabase ----------
try:
    from supabase import create_client, Client  # type: ignore
except Exception:
    create_client = None
    Client = None

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Job Match", layout="wide")
st.title("🤖 AI-Powered Job Matching Dashboard")
st.caption("Phase-1 MVP • Live jobs from Arbeitnow • TF-IDF matching • Optional Supabase logging")

# =========================
# DEFAULT CV (fallback)
# =========================
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
""".strip()

# =========================
# SECRETS / SUPABASE
# =========================
SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL", ""))
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY", ""))

@st.cache_resource(show_spinner=False)
def get_supabase() -> Optional["Client"]:
    if not create_client:
        return None
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        return None

sb = get_supabase()

# stable per-session candidate id
if "candidate_id" not in st.session_state:
    st.session_state["candidate_id"] = str(uuid.uuid4())

# =========================
# SIDEBAR FILTERS
# =========================
with st.sidebar:
    st.header("Filters & Settings")
    source = st.selectbox("Job source", ["Arbeitnow"], index=0)
    top_n = st.slider("Top matches to show", 5, 50, 20, 1)
    min_score = st.slider("Minimum match (%)", 0, 100, 0, 1)
    location_filter = st.text_input("Filter by location (optional)", value="")
    keyword_filter = st.text_input("Filter by keyword in title/desc (optional)", value="")
    st.caption("Results are cached. Adjust filters and re-run.")

# =========================
# CV UPLOAD (CSV / PDF / TXT)
# =========================
st.subheader("1) Upload your CV (CSV / PDF / TXT)")
cv_file = st.file_uploader(
    "Upload a CSV with a 'text' column, a PDF, or a plain .txt file. Leave empty to use the default CV.",
    type=["csv", "pdf", "txt"], accept_multiple_files=False
)

def load_cv_text(file) -> str:
    if file is None:
        return DEFAULT_CV

    name = (file.name or "").lower()

    # CSV
    if name.endswith(".csv"):
        try:
            df = pd.read_csv(file)
            if "text" in df.columns and not df["text"].dropna().empty:
                return "\n".join(df["text"].dropna().astype(str).tolist())[:60000]
            # no 'text' column: join all columns
            rows: List[str] = []
            for _, row in df.iterrows():
                rows.append(" ".join(map(str, row.dropna().tolist())))
            return "\n".join(rows)[:60000]
        except Exception as e:
            st.warning(f"Could not read CSV: {e}. Using default CV.")
            return DEFAULT_CV

    # PDF (lazy import so app can still boot even if pypdf not yet installed)
    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader  # lazy import
            reader = PdfReader(file)
            pages = [(p.extract_text() or "") for p in reader.pages]
            text = "\n".join(pages).strip()
            return (text or DEFAULT_CV)[:60000]
        except ImportError:
            st.error("PDF support not installed. Add `pypdf` to requirements.txt and reboot the app.")
            return DEFAULT_CV
        except Exception as e:
            st.warning(f"Could not read PDF: {e}. Using default CV.")
            return DEFAULT_CV

    # TXT
    if name.endswith(".txt"):
        try:
            text = file.read().decode("utf-8", errors="ignore")
            return (text or DEFAULT_CV)[:60000]
        except Exception as e:
            st.warning(f"Could not read TXT: {e}. Using default CV.")
            return DEFAULT_CV

    return DEFAULT_CV

cv_text = load_cv_text(cv_file)

# =========================
# JOB FETCH (Arbeitnow)
# =========================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_arbeitnow(max_pages: int = 5) -> pd.DataFrame:
    jobs = []
    page = 1
    while page <= max_pages:
        url = f"https://www.arbeitnow.com/api/job-board-api?page={page}"
        ok = False
        for _ in range(3):
            try:
                r = requests.get(url, timeout=10)
                if r.status_code == 200:
                    data = r.json()
                    jobs.extend(data.get("data", []))
                    ok = True
                    if not data.get("links", {}).get("next"):
                        page = max_pages + 1  # exit outer loop
                        break
                    page += 1
                    time.sleep(0.2)
                else:
                    time.sleep(0.6)
            except Exception:
                time.sleep(0.6)
        if not ok:
            break

    if not jobs:
        return pd.DataFrame()

    df = pd.DataFrame(jobs)
    df = df.rename(columns={
        "title": "job_title",
        "description": "job_description",
        "location": "location",
        "url": "url",
        "company_name": "company_name",
    })

    # normalize required columns
    for col in ["job_title", "job_description", "company_name", "location", "url"]:
        if col not in df.columns:
            df[col] = ""

    # dedupe
    if "url" in df.columns:
        df = df.drop_duplicates(subset=["url"], keep="first")
    if {"job_title", "company_name"} <= set(df.columns):
        df = df.drop_duplicates(subset=["job_title", "company_name"], keep="first")

    return df.reset_index(drop=True)

# =========================
# MATCHING (TF-IDF)
# =========================
def compute_matches(cv: str, df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if df.empty:
        return df

    data = df.copy()

    # Filters
    if location_filter:
        data = data[data["location"].fillna("").str.contains(location_filter, case=False, na=False)]
    if keyword_filter:
        mask = (
            data["job_title"].fillna("").str.contains(keyword_filter, case=False, na=False) |
            data["job_description"].fillna("").str.contains(keyword_filter, case=False, na=False)
        )
        data = data[mask]

    if data.empty:
        return data

    combined = (data["job_title"].fillna("") + " " + data["job_description"].fillna("")).tolist()
    corpus = [cv] + combined  # 0 = CV
    tfidf = TfidfVectorizer(stop_words="english", max_df=0.9, min_df=2)
    try:
        X = tfidf.fit_transform(corpus)
    except ValueError:
        data["match_score"] = 0.0
        return data.head(top_k)

    sims = cosine_similarity(X[0:1], X[1:]).flatten()
    data["match_score"] = sims
    data = data.sort_values("match_score", ascending=False)

    # keep top_k and apply min score threshold
    data = data.head(top_k)
    if min_score > 0:
        data = data[data["match_score"] * 100 >= min_score]
    return data

# =========================
# KEYWORD SUGGESTIONS (simple TF-IDF)
# =========================
def suggest_keywords(cv: str, job_texts: List[str], top_n: int = 12) -> List[str]:
    if not job_texts:
        return []
    # Build TF-IDF across job descriptions
    tfidf = TfidfVectorizer(stop_words="english", max_df=0.95)
    X = tfidf.fit_transform(job_texts)
    # Mean TF-IDF across jobs for each term
    import numpy as np
    mean_scores = np.asarray(X.mean(axis=0)).ravel()
    terms = tfidf.get_feature_names_out()
    # Top job keywords
    top_idx = mean_scores.argsort()[::-1][:top_n * 3]  # a bit more before filtering by CV
    job_keywords = [terms[i] for i in top_idx]

    # Remove keywords already in CV (very rough check)
    cv_lower = cv.lower()
    missing = [kw for kw in job_keywords if kw not in cv_lower]
    # return top_n unique
    seen = set()
    out = []
    for kw in missing:
        if kw not in seen:
            out.append(kw)
            seen.add(kw)
        if len(out) >= top_n:
            break
    return out

# =========================
# SUPABASE HELPERS
# =========================
def ensure_candidate(sb_client: Optional["Client"], candidate_id: str, cv: str):
    if not sb_client:
        return
    try:
        sb_client.table("candidates").upsert({
            "id": candidate_id,
            "cv_text": cv[:60000]
        }).execute()
    except Exception as e:
        st.warning(f"Could not upsert candidate: {e}")

def log_interest(sb_client: Optional["Client"], candidate_id: str, row: pd.Series, src: str):
    if not sb_client:
        st.error("Supabase is not configured (missing SUPABASE_URL / SUPABASE_KEY).")
        return
    try:
        payload = {
            "candidate_id": candidate_id,
            "job_title": str(row.get("job_title", ""))[:500],
            "company": str(row.get("company_name", ""))[:300],
            "url": str(row.get("url", row.get("job_url", "")))[:1000],
            "location": str(row.get("location", ""))[:300],
            "source": src,
            "match_score": float(row.get("match_score", 0.0))
        }
        sb_client.table("interests").insert(payload).execute()
        st.toast("Saved your interest ✅", icon="✅")
    except Exception as e:
        st.error(f"Could not save interest: {e}")

# =========================
# MAIN FLOW
# =========================
with st.spinner("🔍 Fetching jobs and computing matches…"):
    if source == "Arbeitnow":
        jobs_df = fetch_arbeitnow()
    else:
        jobs_df = pd.DataFrame()

    if jobs_df.empty:
        st.error("No jobs returned from the source right now. Try again later or tweak filters.")
        st.stop()

    top_matches = compute_matches(cv_text, jobs_df, top_n)

# Save candidate profile (optional)
if sb:
    ensure_candidate(sb, st.session_state["candidate_id"], cv_text)

# Header
st.success("✅ Matching complete!")
st.write(f"Found **{len(top_matches)}** top matches{f' (min {min_score}%)' if min_score else ''}:")

# Download
if not top_matches.empty:
    dl_cols = ["job_title", "company_name", "location", "url", "match_score"]
    st.download_button(
        "⬇️ Download matches as CSV",
        data=top_matches[dl_cols].to_csv(index=False),
        file_name="matches.csv",
        mime="text/csv"
    )

# Cards
for i, row in top_matches.reset_index(drop=True).iterrows():
    st.markdown(f"### {row['job_title']} @ {row.get('company_name', 'Unknown')}")
    st.markdown(
        f"📍 {row.get('location','N/A')} &nbsp;&nbsp;|&nbsp;&nbsp; 🏷️ {source}"
        f" &nbsp;&nbsp;|&nbsp;&nbsp; 🔢 Match: **{round(100*row.get('match_score',0))}%**",
        unsafe_allow_html=True
    )
    url = row.get("url", row.get("job_url", ""))
    if url:
        st.markdown(f"[🔗 View Job Posting]({url})", unsafe_allow_html=True)

    cols = st.columns(2)
    with cols[0]:
        if st.button("✅ I'm Interested", key=f"interest_{i}"):
            log_interest(sb, st.session_state["candidate_id"], row, source)
    with cols[1]:
        desc = (row.get("job_description") or "")[:220]
        st.caption(desc + ("…" if len((row.get("job_description") or "")) > 220 else ""))
    st.divider()

# Suggestions
if not top_matches.empty:
    st.subheader("💡 Improve your CV for these roles")
    suggestions = suggest_keywords(cv_text, top_matches["job_description"].fillna("").tolist(), top_n=12)
    if suggestions:
        st.write("These terms appear often in matched roles but not in your CV:")
        st.write(", ".join(suggestions))
    else:
        st.write("Your CV already covers most frequent terms in matched roles — great!")

