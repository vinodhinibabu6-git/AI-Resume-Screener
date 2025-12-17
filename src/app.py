import streamlit as st
import nltk
import string
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
from nltk.corpus import stopwords

# Clean text function
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

st.title("📄 AI Resume Screener")

st.subheader("Upload Resume (PDF) or Paste Text")

uploaded_file = st.file_uploader("Upload Resume PDF", type=["pdf"])
resume_text = st.text_area("Or paste resume text here")

job_text = st.text_area("Paste Job Description Here")

if st.button("Check Match"):
    if (uploaded_file or resume_text) and job_text:
        if uploaded_file:
            resume_content = extract_text_from_pdf(uploaded_file)
        else:
            resume_content = resume_text

        resume_clean = clean_text(resume_content)
        job_clean = clean_text(job_text)

        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([resume_clean, job_clean])

        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        match_percentage = round(similarity * 100, 2)

        st.success(f"✅ Resume Match Percentage: {match_percentage}%")
    else:
        st.warning("⚠️ Please upload/paste resume and job description.")
