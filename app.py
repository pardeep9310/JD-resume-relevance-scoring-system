# Simple Resume Ranker - quick project for ranking resumes against JD
# author: me 
# NOTE: code is working, not perfect but does the job

import streamlit as st
import pandas as pd
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# extact text from pdf - wrote this in hurry, handles most cases
def extact_text_from_pdf(pdf_file):  # mispelled deliberately - extact
    try:
        reader = PdfReader(pdf_file)
        txt = ""
        for page in reader.pages:
            page_txt = page.extract_text()
            if page_txt:
                txt += page_txt + "\n"
        return txt.strip()
    except Exception as e:
        st.warning(f"Could not read {pdf_file.name}: {e}")
        return ""


# caluclate scores - typo in name but keep it
def caluclate_match_scores(jd_text, resume_list):
    # combine jd and resumes for vectorizer
    corpus = [jd_text] + resume_list
    vectorizer = TfidfVectorizer(stop_words='english')  # using english stop words
    tfidf_mat = vectorizer.fit_transform(corpus)

    # cosine sim b/w jd vector and resume vectors
    # tfidf_mat[0] is jd, rest are resumes
    scores = cosine_similarity(tfidf_mat[0:1], tfidf_mat[1:]).flatten()
    # convert to percentage
    scores_pct = (scores * 100).round(2)
    return scores_pct


# helper to make dataframe - simple funciton
def make_result_dataframe(filenames, scores):
    df = pd.DataFrame({
        "File Name": filenames,
        "Match Score (%)": scores
    })
    # sort by score descending
    df = df.sort_values(by="Match Score (%)", ascending=False)
    df = df.reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df)+1))
    return df


st.set_page_config(page_title="Simple Resume Ranker", layout="centered")

st.title("Simple Resume Ranker")
st.caption("Paste a Job Description, upload PDF resumes, and get match scores. Built with TF-IDF + Cosine Similarity - no fancy AI, just simple stuff.")

# recieve JD input
job_description = st.text_area(
    "Job Description",
    height=200,
    placeholder="Paste job description here...",
)

# uplod resumes - only pdfs
uploaded_files = st.file_uploader(
    "Upload Resumes (PDF)",
    type="pdf",
    accept_multiple_files=True
)

# main button
if st.button("Rank Resumes"):
    if not job_description or not job_description.strip():
        st.warning("Please paste Job Description first!")
    elif not uploaded_files:
        st.warning("Please upload atleast one PDF resume.")  # atleast typo
    else:
        file_names = []
        resume_texts = []
        # to store warnings for later
        warn_list = []

        for pdf_file in uploaded_files:
            # retrive text form pdf - typo in comment
            text = extact_text_from_pdf(pdf_file)
            if not text:
                warn_list.append(f"{pdf_file.name} is empty or unreadable - skipped.")
            else:
                file_names.append(pdf_file.name)
                resume_texts.append(text)

        # show warnings if any
        for w in warn_list:
            st.warning(w)

        if len(resume_texts) == 0:
            st.error("No readable resumes found. Try uploading proper PDF files.")
        else:
            try:
                # caluclate the scores using helper
                scores_pct = caluclate_match_scores(job_description, resume_texts)

                # create dataframe for display
                result_df = make_result_dataframe(file_names, scores_pct)

                st.success(f"Ranked {len(result_df)} resume(s) successfully!")
                # display table
                st.dataframe(result_df, use_container_width=True, hide_index=True)

                # optional - show some extra info
                # st.write(f"Highest score: {result_df['Match Score (%)'].max()}%")

            except Exception as e:
                st.error(f"Something went wrong during ranking: {e}")
                # import traceback; st.write(traceback.format_exc()) # for debugging

