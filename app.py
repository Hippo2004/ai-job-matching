import streamlit as st
import pandas as pd
import requests
from io import StringIO
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import MinMaxScaler

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Job Match", layout="wide")
st.title("🤖 AI-Powered Job Matching Dashboard")
st.write("This is a test version with Hippolyte's CV injected directly.")

# --- BERT MODEL ---
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- STEP 1: Injected CV TEXT ---
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
def fetch_remotive_jobs():
    response = requests.get("https://remotive.io/api/remote-jobs")
    if response.status_code == 200:
        return pd.json_normalize(response.json()["jobs"])
    return pd.DataFrame()

def fetch_arbeitnow_jobs():
    jobs = []
    page = 1
    while True:
        url = f"https://www.arbeitnow.com/api/job-board-api?page={page}"
        response = requests.get(url)
        data = response.json()
        jobs.extend(data.get("data", []))
        if not data.get("links", {}).get("next"):
            break
        page += 1
    return pd.DataFrame(jobs)

# --- STEP 3: Match Scoring ---
def calculate_similarity(cv_text, job_text):
    cv_embed = model.encode(cv_text, convert_to_tensor=True)
    job_embed = model.encode(job_text, convert_to_tensor=True)
    return float(util.cos_sim(cv_embed, job_embed).item())

def match_jobs(cv_text, jobs_df, title_col, desc_col):
    jobs_df = jobs_df.copy()
    jobs_df['combined'] = jobs_df[title_col].fillna('') + " " + jobs_df[desc_col].fillna('')
    jobs_df['match_score'] = jobs_df['combined'].apply(lambda job: calculate_similarity(cv_text, job))
    scaler = MinMaxScaler()
    jobs_df['match_score'] = scaler.fit_transform(jobs_df[['match_score']])
    return jobs_df.sort_values("match_score", ascending=False)

# --- MAIN LOGIC ---
with st.spinner("🔍 Analyzing Hippolyte's CV and fetching jobs..."):
    # Fetch jobs
    remotive_df = fetch_remotive_jobs()
    arbeitnow_df = fetch_arbeitnow_jobs()

    # Harmonize columns
    remotive_df = remotive_df.rename(columns={"title": "job_title", "description": "job_description"})
    arbeitnow_df = arbeitnow_df.rename(columns={"title": "job_title", "description": "job_description"})

    all_jobs = pd.concat([remotive_df, arbeitnow_df], ignore_index=True)
    all_jobs = all_jobs.dropna(subset=['job_title', 'job_description'])

    # Match jobs
    results_df = match_jobs(cv_text, all_jobs, 'job_title', 'job_description')
    top_matches = results_df.head(20)

# --- DISPLAY RESULTS ---
st.success("✅ Matching complete!")
st.write("Here are Hippolyte's top matches:")

for _, row in top_matches.iterrows():
    st.markdown(f"### {row['job_title']} @ {row.get('company_name', 'Unknown')}")
    st.markdown(f"📍 Location: {row.get('location', 'N/A')}  |  💡 Match Score: **{round(row['match_score']*100)}%**")
    st.markdown(f"[🔗 View Job Posting]({row.get('url', row.get('job_url', '#'))})", unsafe_allow_html=True)
    st.button("✅ I’m Interested", key=row['job_title'] + str(row['match_score']))
    st.markdown("---")


