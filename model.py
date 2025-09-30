import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Sample dataset ---
data = {
    "title": [
        "The Hobbit",
        "The Lord of the Rings",
        "Harry Potter and the Sorcerer's Stone",
        "Harry Potter and the Chamber of Secrets",
        "A Game of Thrones",
    ],
    "author": [
        "J.R.R. Tolkien",
        "J.R.R. Tolkien",
        "J.K. Rowling",
        "J.K. Rowling",
        "George R.R. Martin",
    ],
    "description": [
        "A hobbit goes on a journey with dwarves and a wizard.",
        "A group sets out to destroy a powerful ring.",
        "A boy discovers he is a wizard and attends a magical school.",
        "The young wizard faces a hidden chamber and a dark force.",
        "Noble families vie for the throne in a fantasy kingdom.",
    ],
}

books_df = pd.DataFrame(data)

# --- TF-IDF ---
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(books_df["description"])

# --- Similarity ---
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# --- Function ---
def recommend_books(title, top_n=3):
    if title not in books_df["title"].values:
        return ["Book not found in dataset."]
    
    idx = books_df[books_df["title"] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1 : top_n + 1]
    book_indices = [i[0] for i in sim_scores]
    return books_df["title"].iloc[book_indices].tolist()
