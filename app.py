# =========================
# AI Job Matching - Phase 1 MVP
# Fast build (no sklearn), modern UI, robust PDF support (pypdf + pdfminer fallback)
# =========================

import os
import io
import uuid
import time
from typing import Optional

import streamlit as st
import pandas as pd
import requests

# PDF support (primary + fallback)
PDF_ENABLED = False
PDF_BACKUP = False
PDF_IMPORT_ERR = ""
try:
    from pypdf import PdfReader  # primary
    PDF_ENABLED = True
except Exception as e:
    PDF_ENABLED = False
    PDF_IMPORT_ERR = str(e)

try:
    # fallback parser
    from pdfminer.high_level import extract_text as pdfminer_extract_text
    PDF_BACKUP = True
except Exception:
    PDF_BACKUP = False

# Fast string matcher
from rapidfuzz import fuzz

# (Optional) Supabase – safe to include even if secrets absent
try:
    from supabase import create_client, Client
except Exception:
    create_client = None
    Client = None

# -------------------------
# PAGE CONFIG + THEME
# -------------------------
st.set_page_config(page_title="AI Job Match", layout="wide")

# Small CSS polish for modern cards / chips
st.markdown(
    """
    <style>
    .app-header h1 { font-size: 2.1rem !important; margin-bottom: .25rem; }
    .subtitle { color: #9aa3ab; font-size: .95rem; margin-bottom: 1.25rem; }
    .chip { display:inline-block; background:#111827; border:1px solid #2b3440;
            color:#c9d1d9; padding:2px 10px; border-radius:999px; font-size:.8rem; margin-right:6px;}
    .card { border:1px solid #2b3440; background:#0b1220; border-radius:14px; padding:18px; }
    .muted { color: #9aa3ab; }
    .job-title { font-size:1.05rem; font-weight:700; }
    .btn-row { display:flex; gap:10px; flex-wrap:wrap; }
    .match { font-weight:700; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown('<div class="app-header"><h1>🤖 AI-Powered Job Matching Dashboard</h1></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Phase-1 MVP · Live jobs from Arbeitnow · Fast fuzzy matching · Optional Supabase logging</div>',
    unsafe_allow_html=True,
)

# -------------------------
# SECRETS / SUPABASE (optional)
# -------------------------
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "")

@st.cache_resource(show_spinner=False)
def get_supabase() -> Optional["Client"]:
    if not (SUPABASE_URL and SUPABASE_KEY and create_client):
        return None
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        return None

sb = get_supabase()
if "candidate_id" not in st.session_state:
    st.session_state["candidate_id"] = str(uuid.uuid4())

# -------------------------
# DEFAULT CV (fallback)
# -------------------------
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

# -------------------------
# SIDEBAR – Filters / Controls
# -------------------------
with st.sidebar:
    st.header("Filters & Settings")
    top_n = st.slider("Top matches to show", 5, 50, 20, 1)
    min_match = st.slider("Minimum match (%)", 0, 100, 0, 1)
    location_filter = st.text_input("Filter by location (optional)", "")
    keyword_filter = st.text_input("Filter in title/description (optional)", "")
    st.caption("Tip: Leave filters empty for broader results.")

# -------------------------
# FILE UPLOADER (CSV / PDF / TXT)
# -------------------------
st.subheader("1) Upload your CV (CSV / PDF / TXT)")
uploaded = st.file_uploader(
    "Upload a CSV with a 'text' column, a PDF, or a .txt file. Leave empty to use the default CV.",
    type=["csv", "pdf", "txt"],
)

def _read_pdf_bytes(file_obj: io.BytesIO) -> str:
    """Try pypdf first, then fallback to pdfminer.six if needed."""
    # pypdf
    if PDF_ENABLED:
        try:
            reader = PdfReader(file_obj)
            pages = []
            for p in reader.pages:
                pages.append(p.extract_text() or "")
            text = "\n".join(pages).strip()
            if text:
                return text
        except Exception:
            pass
    # pdfminer.fallback
    if PDF_BACKUP:
        try:
            file_obj.seek(0)
            txt = pdfminer_extract_text(file_obj)
            if txt and txt.strip():
                return txt.strip()
        except Exception:
            pass
    return ""

def load_cv_text(upload) -> str:
    if upload is None:
        return DEFAULT_CV

    name = upload.name.lower()

    # CSV
    if name.endswith(".csv"):
        try:
            df = pd.read_csv(upload)
            if "text" in df.columns and not df["text"].dropna().empty:
                return "\n".join(df["text"].dropna().astype(str).tolist())[:60000]
            # Join all columns if no 'text'
            rows = [" ".join(map(str, r.dropna().tolist())) for _, r in df.iterrows()]
            return "\n".join(rows)[:60000]
        except Exception as e:
            st.warning(f"CSV read failed ({e}). Using default CV.")
            return DEFAULT_CV

    # TXT
    if name.endswith(".txt"):
        try:
            return upload.read().decode(errors="ignore")[:60000]
        except Exception as e:
            st.warning(f"TXT read failed ({e}). Using default CV.")
            return DEFAULT_CV

    # PDF
    if name.endswith(".pdf"):
        try:
            data = upload.read()
            buf = io.BytesIO(data)
            text = _read_pdf_bytes(buf)
            if text:
                st.toast("PDF parsed successfully ✅", icon="✅")
                return text[:60000]
            else:
                msg = "Could not parse PDF. "
                if not (PDF_ENABLED or PDF_BACKUP):
                    msg += "PDF support packages not available."
                st.error(msg + " Using default CV.")
                return DEFAULT_CV
        except Exception as e:
            st.error(f"PDF error: {e}. Using default CV.")
            return DEFAULT_CV

    # Anything else – fallback
    return DEFAULT_CV

cv_text = load_cv_text(uploaded)

# If the user uploaded a PDF and our import failed, show a clear, accurate notice
if uploaded is not None and uploaded.name.lower().endswith(".pdf") and not (PDF_ENABLED or PDF_BACKUP):
    st.warning(
        "PDF support packages were not available at runtime. "
        "Ensure `pypdf` (and optionally `pdfminer.six`) are in `requirements.txt`, then **Manage app → Advanced → Reset environment**."
    )

# -------------------------
# FETCH JOBS (Arbeitnow) – cached + resilient
# -------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_arbeitnow() -> pd.DataFrame:
    jobs = []
    page = 1
    while True:
        url = f"https://www.arbeitnow.com/api/job-board-api?page={page}"
        ok = False
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                jobs.extend(data.get("data", []))
                ok = True
                if not data.get("links", {}).get("next"):
                    break
                page += 1
                time.sleep(0.15)
        except Exception:
            pass
        if not ok:
            break

    if not jobs:
        return pd.DataFrame()

    df = pd.DataFrame(jobs).rename(
        columns={
            "title": "job_title",
            "description": "job_description",
            "location": "location",
            "url": "url",
            "company_name": "company_name",
        }
    )
    # Dedupe
    if "url" in df.columns:
        df = df.drop_duplicates(subset=["url"], keep="first")
    if {"job_title", "company_name"} <= set(df.columns):
        df = df.drop_duplicates(subset=["job_title", "company_name"], keep="first")
    return df.reset_index(drop=True)

# -------------------------
# MATCHING (rapidfuzz – fast, tiny)
# -------------------------
def _norm(s: str) -> str:
    return (s or "").lower().strip()

def compute_matches(cv_text: str, df: pd.DataFrame, top_n: int, min_match_pct: int) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    for col in ["job_title", "job_description", "company_name", "location", "url"]:
        if col not in df.columns:
            df[col] = ""

    # Filters
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

    cvn = _norm(cv_text)

    def score_row(row):
        title = _norm(row.get("job_title", ""))
        desc  = _norm(row.get("job_description", ""))
        s1 = fuzz.partial_ratio(cvn, title)
        s2 = fuzz.token_set_ratio(cvn, f"{title} {desc}")
        return 0.4 * s1 + 0.6 * s2

    df["match_pct"] = df.apply(score_row, axis=1).astype(float)
    df = df[df["match_pct"] >= float(min_match_pct)]
    df = df.sort_values("match_pct", ascending=False)
    return df.head(top_n)

# -------------------------
# SUPABASE WRITE HELPERS (optional)
# -------------------------
def ensure_candidate(sb_client: "Client", candidate_id: str, cv_text: str):
    try:
        sb_client.table("candidates").upsert({"id": candidate_id, "cv_text": cv_text[:60000]}).execute()
    except Exception as e:
        st.warning(f"Could not upsert candidate: {e}")

def log_interest(sb_client: "Client", candidate_id: str, row: pd.Series):
    try:
        payload = {
            "candidate_id": candidate_id,
            "job_title": str(row.get("job_title", ""))[:500],
            "company": str(row.get("company_name", ""))[:300],
            "url": str(row.get("url", row.get("job_url", "")))[:1000],
            "location": str(row.get("location", ""))[:300],
            "source": "Arbeitnow",
            "match_score": float(row.get("match_pct", 0.0)) / 100.0,
        }
        sb_client.table("interests").insert(payload).execute()
        st.toast("Saved your interest ✅", icon="✅")
    except Exception as e:
        st.error(f"Could not save interest: {e}")

# -------------------------
# MAIN – fetch → match → display
# -------------------------
with st.spinner("🔍 Fetching jobs and computing matches…"):
    jobs_df = fetch_arbeitnow()
    if jobs_df.empty:
        st.error("No jobs returned from Arbeitnow right now. Try again in a bit.")
        st.stop()

    top_matches = compute_matches(cv_text, jobs_df, top_n, min_match)
    if top_matches.empty:
        st.warning("No matches after filters. Loosen filters or lower the minimum match%.")
        st.stop()

st.success(f"✅ Matching complete!")
st.caption(f"Showing {len(top_matches)} matches (min {min_match}% match).")

# Save candidate once (optional)
if sb:
    ensure_candidate(sb, st.session_state["candidate_id"], cv_text)

# Download as CSV
csv_col = st.columns([1, 3, 1])[0]
with csv_col:
    csv_data = top_matches[["job_title", "company_name", "location", "url", "match_pct"]].rename(
        columns={"match_pct": "match_percent"}
    ).to_csv(index=False)
    st.download_button("⬇️ Download matches as CSV", csv_data, "matches.csv", "text/csv")

# Render cards
for _, row in top_matches.iterrows():
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)

        # Header row
        c1, c2 = st.columns([4, 1])
        with c1:
            st.markdown(f"<div class='job-title'>{row['job_title']} @ {row.get('company_name','Unknown')}</div>", unsafe_allow_html=True)
            loc = row.get("location", "N/A") or "N/A"
            st.markdown(f"<div class='muted'>📍 {loc} &nbsp; &nbsp; <span class='chip'>Arbeitnow</span></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='chip match'>{round(row.get('match_pct',0))}% match</div>", unsafe_allow_html=True)

        # Action row
        url = row.get("url", row.get("job_url", ""))
        colA, colB = st.columns([1, 1])
        with colA:
            if url:
                st.link_button("🔗 View Job", url)
            else:
                st.button("🔗 View Job", disabled=True)
        with colB:
            if st.button("✅ I'm Interested", key=f"interest_{uuid.uuid4().hex[:8]}"):
                if sb:
                    log_interest(sb, st.session_state["candidate_id"], row)
                else:
                    st.info("Interest noted locally (Supabase not configured).")

        # Summary
        desc = (row.get("job_description") or "").strip()
        if desc:
            st.markdown(f"<div class='muted' style='margin-top:.5rem'>{desc[:300]}{'…' if len(desc)>300 else ''}</div>", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("")  # space




