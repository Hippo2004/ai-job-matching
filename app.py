import os
import uuid
import time
from typing import Optional, List, Dict
from collections import Counter
import math

import streamlit as st
import pandas as pd
import requests

# Try scikit-learn; fall back if unavailable
try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity        # type: ignore
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# Optional: Supabase
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
st.caption("Phase-1 MVP • Arbeitnow live jobs • TF-IDF matching (with pure-Python fallback) • Optional Supabase logging")

# =========================
# DEFAULT CV
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
    if not create_client or not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        return None

sb = get_supabase()
if "candidate_id" not in st.session_state:
    st.session_state["candidate_id"] = str(uuid.uuid4())

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("Filters & Settings")
    top_n = st.slider("Top matches to show", 5, 50, 20, 1)
    min_score = st.slider("Minimum match (%)", 0, 100, 0, 1)
    location_filter = st.text_input("Filter by location (optional)", value="")
    keyword_filter = st.text_input("Filter in title/description (optional)", value="")
    st.caption(f"Matcher: {'scikit-learn TF-IDF' if SKLEARN_OK else 'pure-Python bag-of-words'}")

# =========================
# CV UPLOAD (CSV / PDF / TXT)
# =========================
st.subheader("1) Upload your CV (CSV / PDF / TXT)")
cv_file = st.file_uploader(
    "Upload a CSV with a 'text' column, a PDF, or a .txt file. Leave empty to use the default CV.",
    type=["csv", "pdf", "txt"]
)

def load_cv_text(file) -> str:
    if file is None:
        return DEFAULT_CV
    name = (file.name or "").lower()

    if name.endswith(".csv"):
        try:
            df = pd.read_csv(file)
            if "text" in df.columns and not df["text"].dropna().empty:
                return "\n".join(df["text"].dropna().astype(str).tolist())[:60000]
            rows: List[str] = []
            for _, row in df.iterrows():
                rows.append(" ".join(map(str, row.dropna().tolist())))
            return "\n".join(rows)[:60000]
        except Exception as e:
            st.warning(f"Could not read CSV: {e}. Using default CV.")
            return DEFAULT_CV

    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader  # lazy import so app still boots if pypdf not ready
            reader = PdfReader(file)
            text = "\n".join([(p.extract_text() or "") for p in reader.pages]).strip()
            return (text or DEFAULT_CV)[:60000]
        except ImportError:
            st.error("PDF support not installed. Ensure `pypdf` is in requirements.txt and reboot.")
            return DEFAULT_CV
        except Exception as e:
            st.warning(f"Could not read PDF: {e}. Using default CV.")
            return DEFAULT_CV

    if name.endswith(".txt"):
        try:
            return (file.read().decode("utf-8", errors="ignore") or DEFAULT_CV)[:60000]
        except Exception as e:
            st.warning(f"Could not read TXT: {e}. Using default CV.")
            return DEFAULT_CV

    return DEFAULT_CV

cv_text = load_cv_text(cv_file)

# =========================
# FETCH JOBS (Arbeitnow)
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
                        page = max_pages + 1
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

    df = pd.DataFrame(jobs).rename(columns={
        "title": "job_title",
        "description": "job_description",
        "location": "location",
        "url": "url",
        "company_name": "company_name",
    })

    for col in ["job_title", "job_description", "company_name", "location", "url"]:
        if col not in df.columns:
            df[col] = ""

    if "url" in df.columns:
        df = df.drop_duplicates(subset=["url"], keep="first")
    if {"job_title", "company_name"} <= set(df.columns):
        df = df.drop_duplicates(subset=["job_title", "company_name"], keep="first")
    return df.reset_index(drop=True)

# =========================
# MATCHING
# =========================
def tokenize(text: str) -> List[str]:
    return [t for t in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if t]

def cosine_counts(a: Dict[str, int], b: Dict[str, int]) -> float:
    # cosine similarity on term counts
    common = set(a).intersection(b)
    dot = sum(a[t] * b[t] for t in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def compute_matches(cv: str, df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if df.empty:
        return df

    data = df.copy()

    # filters
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

    if SKLEARN_OK:
        combined = (data["job_title"].fillna("") + " " + data["job_description"].fillna("")).tolist()
        corpus = [cv] + combined
        try:
            vec = TfidfVectorizer(stop_words="english", max_df=0.9, min_df=2)
            X = vec.fit_transform(corpus)
            sims = cosine_similarity(X[0:1], X[1:]).flatten()
        except Exception:
            # fall back if vectorizer fails for tiny corpora
            sims = [0.0] * len(combined)
    else:
        # pure-Python bag-of-words cosine
        cv_counts = Counter(tokenize(cv))
        sims = []
        for txt in (data["job_title"].fillna("") + " " + data["job_description"].fillna("")).tolist():
            sims.append(cosine_counts(cv_counts, Counter(tokenize(txt))))

    data["match_score"] = sims
    data = data.sort_values("match_score", ascending=False).head(top_k)
    if min_score:
        data = data[data["match_score"] * 100 >= min_score]
    return data

def suggest_keywords(cv: str, job_texts: List[str], top_n: int = 12) -> List[str]:
    if not job_texts:
        return []
    cv_words = set(tokenize(cv))
    # simple frequency across jobs
    freq = Counter()
    for txt in job_texts:
        freq.update(set(tokenize(txt)))  # set to avoid over-counting repeated terms in the same posting
    # remove words already in CV and very short tokens
    candidates = [(w, c) for w, c in freq.items() if w not in cv_words and len(w) > 2]
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [w for w, _ in candidates[:top_n]]

# =========================
# SUPABASE HELPERS
# =========================
def ensure_candidate(sb_client: Optional["Client"], candidate_id: str, cv: str):
    if not sb_client:
        return
    try:
        sb_client.table("candidates").upsert({"id": candidate_id, "cv_text": cv[:60000]}).execute()
    except Exception as e:
        st.warning(f"Could not upsert candidate: {e}")

def log_interest(sb_client: Optional["Client"], candidate_id: str, row: pd.Series):
    if not sb_client:
        st.error("Supabase not configured (missing secrets).")
        return
    try:
        payload = {
            "candidate_id": candidate_id,
            "job_title": str(row.get("job_title", ""))[:500],
            "company": str(row.get("company_name", ""))[:300],
            "url": str(row.get("url", row.get("job_url", "")))[:1000],
            "location": str(row.get("location", ""))[:300],
            "source": "Arbeitnow",
            "match_score": float(row.get("match_score", 0.0)),
        }
        sb_client.table("interests").insert(payload).execute()
        st.toast("Saved your interest ✅", icon="✅")
    except Exception as e:
        st.error(f"Could not save interest: {e}")

# =========================
# MAIN
# =========================
with st.spinner("🔍 Fetching jobs and computing matches…"):
    jobs_df = fetch_arbeitnow()
    if jobs_df.empty:
        st.error("No jobs returned from Arbeitnow right now. Try again later.")
        st.stop()
    matches = compute_matches(cv_text, jobs_df, top_n)

if sb:
    ensure_candidate(sb, st.session_state["candidate_id"], cv_text)

st.success("✅ Matching complete!")
st.write(f"Showing **{len(matches)}** matches{f' (min {min_score}%)' if min_score else ''}.")

if not matches.empty:
    dl_cols = ["job_title", "company_name", "location", "url", "match_score"]
    st.download_button(
        "⬇️ Download matches as CSV",
        matches[dl_cols].to_csv(index=False),
        "matches.csv",
        "text/csv",
    )

for i, row in matches.reset_index(drop=True).iterrows():
    st.markdown(f"### {row['job_title']} @ {row.get('company_name','Unknown')}")
    st.markdown(
        f"📍 {row.get('location','N/A')} &nbsp;|&nbsp; 🔢 Match: **{round(100*row.get('match_score',0))}%** &nbsp;|&nbsp; 🏷️ Arbeitnow",
        unsafe_allow_html=True,
    )
    url = row.get("url", row.get("job_url", ""))
    if url:
        st.markdown(f"[🔗 View Job Posting]({url})", unsafe_allow_html=True)
    cols = st.columns(2)
    with cols[0]:
        if st.button("✅ I'm Interested", key=f"interest_{i}"):
            log_interest(sb, st.session_state["candidate_id"], row)
    with cols[1]:
        desc = (row.get("job_description") or "")[:220]
        st.caption(desc + ("…" if len((row.get("job_description") or "")) > 220 else ""))
    st.divider()

if not matches.empty:
    st.subheader("💡 Improve your CV for these roles")
    sugg = suggest_keywords(cv_text, matches["job_description"].fillna("").tolist(), top_n=12)
    if sugg:
        st.write(", ".join(sugg))
    else:
        st.write("Your CV already covers the frequent terms in these roles — nice!")


