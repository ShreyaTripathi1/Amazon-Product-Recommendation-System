# Search Engine and Recommendation System on Amazon Product

## Overview

This project is a Search Engine and Recommendation System built using a dataset of Amazon products.
This project implements a text-based product recommendation system using Natural Language Processing (NLP) techniques. The recommendation system is designed to suggest the most relevant products based on a user's query by analyzing and comparing the textual content (title and description) of products in the dataset.

## Prerequisites

To run this project, you'll need to have the following software installed on your system:
Python 3.7 or higher
Jupyter Notebook
Necessary Python libraries (listed below)
- pandas: For data manipulation and analysis.
- numpy: For numerical operations.
- nltk: For natural language processing, including tokenization and stemming.
- scikit-learn: For TF-IDF vectorization and cosine similarity calculation.
- streamlit: For creating the web app interface.
- Pillow: For image handling and display.


## Dataset: 
amazon_product.csv



## How It Works
### 1. Data Preprocessing
Null Values: First, we check for any null values in the dataset and remove them if necessary.
Text Normalization: The text in the product titles and descriptions is converted to lowercase to ensure uniformity.
Tokenization: The text is broken down into individual words or tokens.
Stemming: We use the Snowball Stemmer to reduce words to their base or root form. This helps in simplifying the text and reducing dimensionality, making the analysis more effective.
### 2. Cosine Similarity
TF-IDF Vectorization: The preprocessed text is transformed into numerical vectors using the TfidfVectorizer from the scikit-learn library. TF-IDF (Term Frequency-Inverse Document Frequency) is a statistical measure that evaluates how important a word is to a document relative to a collection of documents.
Cosine Similarity Calculation: Cosine similarity is used to measure the similarity between the user query and each product's text. It calculates the cosine of the angle between two vectors (representing the text data), with a value close to 1 indicating high similarity.
### 3. Recommendation Process
Query Processing: The user's query is tokenized and stemmed in the same way as the product data.
Similarity Scoring: The system calculates the cosine similarity between the processed query and the stemmed tokens of each product in the dataset.
Result Ranking: Products are ranked based on their similarity scores, and the top 10 most similar products are returned as recommendations.

# Model Explanation
The recommendation system is based on cosine similarity rather than a traditional machine learning model. This approach leverages vectorized representations of text (TF-IDF) to measure and compare the similarity between the user's query and product descriptions.

# Why Cosine Similarity?
Cosine similarity is particularly effective for text-based recommendations because it considers the orientation (angle) between vectors rather than their magnitude. This makes it well-suited for comparing text documents of varying lengths.

# Accuracy
The accuracy of the recommendations depends on the quality of the product data and how well the query matches the available descriptions. While this system is not trained on labeled data like a supervised machine learning model, it can provide highly relevant results based on textual similarity.

# Limitations
The system's effectiveness relies heavily on the textual content of the product data. If descriptions are sparse or uninformative, recommendations may be less accurate.
It does not account for user preferences or historical data, which might limit personalization.

# Usage
To use the recommendation system, simply input a query related to the product you're looking for, and the system will return the top 10 products that most closely match your query based on text similarity.
