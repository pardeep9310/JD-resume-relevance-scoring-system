# Simple Resume Ranker

Lightweight Streamlit app that ranks PDF resumes against a Job Description using TF-IDF + Cosine Similarity (scikit-learn). No LLM or external API.

## Setup & Run

```bash
# 1. (optional) create venv
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. install deps
pip install -r requirements.txt

# 3. run
streamlit run app.py
```

Open http://localhost:8501

## Usage
1. Paste Job Description in the text area
2. Upload one or more PDF resumes
3. Click **Rank Resumes** — table shows `Rank | File Name | Match Score (%)` sorted descending

## Requirements
- Python 3.9+
- streamlit, pypdf, scikit-learn, pandas (see `requirements.txt`)

## Notes
- Only text-based PDFs work (scanned images will be skipped with a warning)
- Scores are 0-100% cosine similarity
