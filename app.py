import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Job Match", layout="wide")
st.title("🤖 AI-Powered Job Matching Dashboard")
st.write("This is a test version with Hippolyte's CV injected directly.")

# ---- Hippolyte CV (injected for Phase 1) ----
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
"""

# ---- Data sources ----
REMOTIVE_API = "https://remotive.io/api/remote-jobs"

@st.cache_data(ttl=1800)
def fetch_remotive_jobs():
    try:
        r = requests.get(REMOTIVE_API, timeout=10)
        r.raise_for_status()
        df = pd.json_normalize(r.json().get("jobs", []))
        if df.empty:
            return df
        # standardize columns we’ll use
        keep = {
            "title": "job_title",
            "company_name": "company",
            "description": "description",
            "candidate_required_location": "location",
            "url": "url"
        }
        df = df.rename(columns=keep)[list(keep.values())]
        df["job_title"] = df["job_title"].fillna("")
        df["description"] = df["description"].fillna("")
        df["location"] = df["location"].fillna("N/A")
        return df
    except Exception as e:
        st.warning(f"⚠️ Failed to fetch jobs from Remotive: {e}")
        return pd.DataFrame(columns=["job_title","company","description","location","url"])

def score_with_tfidf(cv_text: str, jobs_df: pd.DataFrame) -> pd.DataFrame:
    if jobs_df.empty:
        return jobs_df
    docs = [cv_text] + (jobs_df["job_title"] + " " + jobs_df["description"]).tolist()
    vect = TfidfVectorizer(stop_words="english", max_features=20000)
    X = vect.fit_transform(docs)
    cv_vec = X[0:1]
    job_vecs = X[1:]
    sims = cosine_similarity(cv_vec, job_vecs).ravel()  # 0..1
    jobs_df = jobs_df.copy()
    jobs_df["match_score"] = sims
    return jobs_df.sort_values("match_score", ascending=False)

# ---- Main ----
with st.spinner("🔍 Fetching jobs & computing matches..."):
    jobs = fetch_remotive_jobs()
    if jobs.empty:
        st.error("No jobs returned right now. Try again in a bit.")
    else:
        results = score_with_tfidf(CV_TEXT, jobs)
        top = results.head(20)

        st.success("✅ Matching complete!")
        for _, row in top.iterrows():
            title = row["job_title"] or "Untitled role"
            company = row.get("company") or "Unknown"
            score_pct = int(round(row["match_score"] * 100))
            st.markdown(f"### {title} @ {company}")
            st.markdown(f"📍 {row.get('location','N/A')}  |  💡 Match Score: **{score_pct}%**")
            if row.get("url"):
                st.markdown(f"[🔗 View Job Posting]({row['url']})", unsafe_allow_html=True)
            st.divider()


