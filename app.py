import streamlit as st
import pandas as pd
import requests
from io import StringIO
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import MinMaxScaler

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Job Match", layout="wide")
st.title("🤖 AI-Powered Job Matching Dashboard")
st.markdown("Upload your CV (CSV format) or use Hippolyte's CV by default for demo purposes.")

# --- BERT MODEL ---
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- STEP 1: Upload or Use Default CV ---
uploaded_file = st.file_uploader("📄 Upload your CV (CSV format)", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        cv_text = " ".join(df.astype(str).fillna('').values.flatten())
        st.success("✅ CV uploaded and parsed successfully.")
    except Exception as e:
        st.error(f"❌ Failed to parse uploaded file: {e}")
        st.stop()
else:
    st.info("ℹ️ No file uploaded — using Hippolyte's CV for testing.")
    cv_text = """
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

# --- STEP 2: Fetch Live Jobs ---
@st.cache_data(show_spinner=False)
def fetch_remotive_jobs():
    try:
        r = requests.get("https://remotive.io/api/remote-jobs", timeout=10)
        return pd.json_normalize(r.json()["jobs"])
    except:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def fetch_arbeitnow_jobs():
    jobs = []
    page = 1
    try:
        while True:
            url = f"https://www.arbeitnow.com/api/job-board-api?page={page}"
            r = requests.get(url, timeout=10)
            data = r.json()
            jobs.extend(data.get("data", []))
            if not data.get("links", {}).get("next"):
                break
            page += 1
    except:
        pass
    return pd.DataFrame(jobs)

# --- STEP 3: Match Scoring ---
def calculate_similarity(cv_text, job_text):
    cv_embed = model.encode(cv_text, convert_to_tensor=True)
    job_embed = model.encode(job_text, convert_to_tensor=True)
    return float(util.cos_sim(cv_embed, job_embed).item())

def match_jobs(cv_text, jobs_df, title_col, desc_col):
    jobs_df = jobs_df.copy()
    jobs_df["combined"] = jobs_df[title_col].fillna("") + " " + jobs_df[desc_col].fillna("")
    jobs_df["match_score"] = jobs_df["combined"].apply(lambda x: calculate_similarity(cv_text, x))
    jobs_df["match_score"] = MinMaxScaler().fit_transform(jobs_df[["match_score"]])
    return jobs_df.sort_values("match_score", ascending=False)

# --- MAIN LOGIC ---
with st.spinner("🔍 Analyzing CV and fetching jobs..."):
    remotive = fetch_remotive_jobs()
    arbeitnow = fetch_arbeitnow_jobs()

    # Harmonize
    remotive = remotive.rename(columns={"title": "job_title", "description": "job_description"})
    arbeitnow = arbeitnow.rename(columns={"title": "job_title", "description": "job_description"})

    jobs_df = pd.concat([remotive, arbeitnow], ignore_index=True)
    jobs_df = jobs_df.dropna(subset=["job_title", "job_description"])

    if jobs_df.empty:
        st.warning("⚠️ No jobs available from APIs at this time.")
    else:
        top_matches = match_jobs(cv_text, jobs_df, "job_title", "job_description").head(20)

        # --- DISPLAY RESULTS ---
        st.success("✅ Matching complete!")
        st.write("Here are your top matches:")

        for _, row in top_matches.iterrows():
            st.markdown(f"### {row['job_title']} @ {row.get('company_name', 'Unknown')}")
            st.markdown(f"📍 Location: {row.get('location', 'N/A')} | 💡 Match Score: **{round(row['match_score']*100)}%**")
            st.markdown(f"[🔗 View Job Posting]({row.get('url', row.get('job_url', '#'))})", unsafe_allow_html=True)
            st.button("✅ I’m Interested", key=row['job_title'] + str(row['match_score']))
            st.markdown("---")


