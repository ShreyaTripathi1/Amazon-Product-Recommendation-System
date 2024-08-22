import pandas as pd
import numpy as np
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image

# Load the dataset
data = pd.read_csv('amazon_product.csv')

# Remove unnecessary columns
data = data.drop('id', axis=1)

# Define tokenizer and stemmer
stemmer = SnowballStemmer('english')
def tokenize_and_stem(text):
    tokens = nltk.word_tokenize(text.lower())
    stems = [stemmer.stem(t) for t in tokens]
    return stems

# Create stemmed tokens column
data['stemmed_tokens'] = data.apply(lambda row: tokenize_and_stem(row['Title'] + ' ' + row['Description']), axis=1)

# Define TF-IDF vectorizer and cosine similarity function
tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem)
def cosine_sim(text1, text2):
    # tfidf_matrix = tfidf_vectorizer.fit_transform([text1, text2])
    text1_concatenated = ' '.join(text1)
    text2_concatenated = ' '.join(text2)
    tfidf_matrix = tfidf_vectorizer.fit_transform([text1_concatenated, text2_concatenated])
    return cosine_similarity(tfidf_matrix)[0][1]

# Define search function
def search_products(query):
    query_stemmed = tokenize_and_stem(query)
    data['similarity'] = data['stemmed_tokens'].apply(lambda x: cosine_sim(query_stemmed, x))
    results = data.sort_values(by=['similarity'], ascending=False).head(10)[['Title', 'Description', 'Category']]
    return results

# web app

# Custom CSS for styling
st.markdown(
    """
    <style>
    /* Set background color of the entire page */
    .stApp {
        background-color: #232F3E;
    }

    /* Center the button and move it to the bottom */
    .stButton button {
        background-color: #ff9900;
        color: white;
        margin-top: 20px;
        margin-left: auto;
        margin-right: 0;
        display: block;
        transform: translateX(-50%);
    }

    /* Style the text input */
    input[type="text"] {
        background-color: #fff;
        color: #000;
    }

    /* Style for the title with a professional font */
    .title-text {
        color: white;
        font-family: 'Arial', sans-serif;
        font-size: 36px;
        font-weight: bold;
    }

    /* Style for the input label */
    .stTextInput label {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Create two columns
col1, col2 = st.columns([1, 2])  # Adjust the ratio of the columns as needed

# Column 1: Display the logo aligned to the left
with col1:

    img = Image.open('amazon_logo.png')
    st.image(img, width=250)

# Column 2: Display the title next to the logo
with col2:
    st.write("")
    st.write("")
    st.write("")
    st.markdown("<div class='title-text'>Search Engine & Product Recommendation System</div>", unsafe_allow_html=True)

    #st.title("Search Engine & Product Recommendation System")



# Input field 
query = st.text_input("Enter Product Name")
# Search button
submit = st.button('Search')

# Handle the search and display results
if submit:
    res = search_products(query)
    st.write(res)