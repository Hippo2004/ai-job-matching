import streamlit as st
import pandas as pd
import requests
from io import StringIO
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import MinMaxScaler
import spacy
from collections import Counter

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Job Match", layout="wide")
st.title("🤖 AI-Powered Job Matching Dashboard")
st.write("This is a test version with Hippolyte's CV injected directly.")

# --- BERT MODEL ---
model = SentenceTransformer("all-MiniLM-L6-v2")

try:
    nlp = spacy.load("en_core_web_sm")
except:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

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

# --- STEP 2: Fetch Live Jobs (SAFE + TIMEOUT) ---
def fetch_remotive_jobs():
    try:
        response = requests.get("https://remotive.io/api/remote-jobs", timeout=10)
        if response.status_code == 200:
            return pd.json_normalize(response.json()["jobs"])
    except Exception as e:
        st.warning(f"⚠️ Failed to fetch Remotive jobs: {e}")
    return pd.DataFrame()

# --- Skip Arbeitnow for now to avoid timeout ---
def fetch_arbeitnow_jobs():
    return pd.DataFrame()

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

# --- IMPROVEMENT SUGGESTIONS ---
def extract_top_keywords(texts, top_n=10):
    all_nouns = []
    for text in texts:
        doc = nlp(text)
        all_nouns.extend([token.lemma_.lower() for token in doc if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop])
    return [word for word, _ in Counter(all_nouns).most_common(top_n)]

def suggest_improvements(cv_text, job_texts):
    top_keywords = extract_top_keywords(job_texts, top_n=15)
    cv_doc = nlp(cv_text.lower())
    cv_words = set([token.lemma_.lower() for token in cv_doc if token.pos_ in ["NOUN", "PROPN"]])
    missing_keywords = [kw for kw in top_keywords if kw not in cv_words]
    return missing_keywords

# --- MAIN LOGIC ---
with st.spinner("🔍 Analyzing Hippolyte's CV and fetching jobs..."):
    remotive_df = fetch_remotive_jobs()
    arbeitnow_df = fetch_arbeitnow_jobs()

    remotive_df = remotive_df.rename(columns={"title": "job_title", "description": "job_description"})
    arbeitnow_df = arbeitnow_df.rename(columns={"title": "job_title", "description": "job_description"})

    all_jobs = pd.concat([remotive_df, arbeitnow_df], ignore_index=True)
    all_jobs = all_jobs.dropna(subset=['job_title', 'job_description'])

    if all_jobs.empty:
        st.error("❌ Could not fetch any job data. Please try again later.")
    else:
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

        # --- SUGGEST IMPROVEMENTS ---
        missing = suggest_improvements(cv_text, top_matches['job_description'].tolist())
        if missing:
            st.subheader("💡 Improve Your Profile to Match More Jobs")
            st.markdown("These keywords appeared in top job matches but are not present in your CV:")
            for word in missing:
                st.markdown(f"- {word}")
        else:
            st.markdown("✅ Your profile covers most key areas from matched jobs!")

