from sklearn.metrics.pairwise import cosine_similarity
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
from nltk.corpus import stopwords

def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def clean_text(text):
    text = text.lower()  
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    words = text.split()
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)

resume_text = load_text('../data/resume.txt')
job_text = load_text('../data/job_description.txt')

resume_clean = clean_text(resume_text)
job_clean = clean_text(job_text)
vectorizer = TfidfVectorizer()

tfidf_vectors = vectorizer.fit_transform([resume_clean, job_clean])

print("TF-IDF Vector Shape:", tfidf_vectors.shape)

similarity_score = cosine_similarity(
    tfidf_vectors[0:1],
    tfidf_vectors[1:2]
)[0][0]

match_percentage = round(similarity_score * 100, 2)

print("\n✅ Resume Match Percentage:", match_percentage, "%")

print("CLEANED RESUME TEXT:\n", resume_clean)
print("\nCLEANED JOB DESCRIPTION TEXT:\n", job_clean)

with open('../outputs/result.txt', 'w') as file:
    file.write(f"Resume Match Percentage: {match_percentage}%")

print("\n📁 Result saved successfully in outputs/result.txt")
