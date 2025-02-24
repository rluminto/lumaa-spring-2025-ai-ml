#Import all the libraries we need
import pandas as pd
import re
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Remove punctuation and covert to lowercase
def clean_text(text):
    """Remove punctuation and convert text to lowercase."""
    text = re.sub(r"[^\w\s]", "", text)  
    text = text.lower()                 
    return text

#Convert combined text (description + reviews) to TF-IDF vectors
def vectorize_text(df):
    """Transform text data into TF-IDF vectors."""
    tfidf = TfidfVectorizer(stop_words="english") 
    tfidf_matrix = tfidf.fit_transform(df["combined_text"]) 
    return tfidf, tfidf_matrix

#Recommend movies based on query similarity
def recommend_movies(query, df, tfidf, tfidf_matrix):
    #Clean and vectorize the input query
    cleaned_query = clean_text(query)
    query_vector = tfidf.transform([cleaned_query])  
    
    #Calculate similarity scores
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    df["Similarity"] = similarity_scores
    
    #Get indices of top 5 matches in descending order
    top_indices = similarity_scores.argsort()[-5:][::-1]
    
    return df.iloc[top_indices] 

def main():
    #Load and preprocess data
    df = pd.read_csv("movies.csv") 
    
    # Clean and combine description + reviews
    df["cleaned_description"] = df["Description"].fillna("").apply(clean_text)
    df["cleaned_review"] = df["Review"].fillna("").apply(clean_text)
    df["combined_text"] = df["cleaned_description"] + " " + df["cleaned_review"]
    
    # Vectorize the combined text
    tfidf, tfidf_matrix = vectorize_text(df)
    
    #Take in command-line input
    if len(sys.argv) < 2:
        sys.exit(1)
    query = sys.argv[1] 
    
    
    # Generate and display recommendations
    recommendations = recommend_movies(query, df, tfidf, tfidf_matrix)

    print("\nTop Recommendations:")
    print(recommendations[["Title", "Description", "Genre", "Rating", "Similarity"]]) 
    
if __name__ == "__main__":
    main()