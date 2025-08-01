import streamlit as st
import pandas as pd
import requests
import io
from difflib import SequenceMatcher

st.set_page_config(page_title="AI Job Matcher", layout="wide")

st.title("🔍 AI Job Matching Platform — Candidate Dashboard")

st.markdown("Upload your CV and let AI match you with real jobs from **Remotive**!")

# --- Upload CV ---
uploaded_file = st.file_uploader("📄 Upload your CV (CSV format)", type=["csv"])

if uploaded_file:
    try:
        cv_df = pd.read_csv(uploaded_file)
        st.success("CV uploaded and parsed successfully!")
        st.dataframe(cv_df.head())
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

# --- Load Live Jobs from Remotive API ---
@st.cache_data(show_spinner=False)
def fetch_remotive_jobs():
    url = "https://remotive.io/api/remote-jobs"
    try:
        response = requests.get(url)
        data = response.json()
        return pd.DataFrame(data["jobs"])
    except Exception as e:
        st.error("Failed to fetch jobs from Remotive.")
        return pd.DataFrame()

jobs_df = fetch_remotive_jobs()
st.subheader("💼 Found {} Live Remote Jobs".format(len(jobs_df)))

# --- Matching Logic ---
def match_score(cv_text, job_title, job_desc):
    combined_job = job_title + " " + job_desc
    return round(SequenceMatcher(None, cv_text.lower(), combined_job.lower()).ratio() * 100, 2)

if uploaded_file:
    st.subheader("🎯 Top Matches")

    # Convert full CV to text (basic)
    cv_text_blob = " ".join(str(x) for x in cv_df.values.flatten())

    # Score all jobs
    jobs_df["match_score"] = jobs_df.apply(
        lambda row: match_score(cv_text_blob, row["title"], row["description"]), axis=1
    )

    top_matches = jobs_df.sort_values(by="match_score", ascending=False).head(10)

    for _, row in top_matches.iterrows():
        with st.expander(f"{row['title']} at {row['company_name']} — Match: {row['match_score']}%"):
            st.write(row["description"])
            st.write(f"🌍 Location: {row['candidate_required_location']}")
            st.write(f"🔗 [Apply here]({row['url']})")
            st.button("✅ I'm Interested", key=row['id'])

