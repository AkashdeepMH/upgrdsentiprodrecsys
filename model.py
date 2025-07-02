# model.py

# Importing Libraries
import pandas as pd
import re
import nltk
import spacy
import string
import pickle as pk

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer, WordNetLemmatizer

# ------------------------
# NLTK and spaCy Setup
# ------------------------

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

try:
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
except:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

# ------------------------
# Load Pretrained Objects
# ------------------------

count_vector = pk.load(open('pickle_file/count_vector.pkl', 'rb'))             # Count Vectorizer
tfidf_transformer = pk.load(open('pickle_file/tfidf_transformer.pkl', 'rb'))   # TFIDF Transformer
model = pk.load(open('pickle_file/model.pkl', 'rb'))                            # Sentiment Classification Model
recommend_matrix = pk.load(open('pickle_file/user_final_rating.pkl', 'rb'))    # Recommendation Matrix

# Load dataset
product_df = pd.read_csv('sample30.csv', sep=",").fillna("")

# Fallback: Fit TFIDF transformer if not already fitted (needed for production stability)
if not hasattr(tfidf_transformer, 'idf_'):
    X_counts = count_vector.transform(product_df['reviews_text'])
    tfidf_transformer.fit(X_counts)

# ------------------------
# Text Preprocessing Utils
# ------------------------

stopword_list = stopwords.words('english')

def remove_special_characters(text, remove_digits=True):
    pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
    return re.sub(pattern, '', text)

def to_lowercase(words):
    return [word.lower() for word in words]

def remove_punctuation_and_splchars(words):
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word:
            new_word = remove_special_characters(new_word, True)
            new_words.append(new_word)
    return new_words

def remove_stopwords(words):
    return [word for word in words if word not in stopword_list]

def stem_words(words):
    stemmer = LancasterStemmer()
    return [stemmer.stem(word) for word in words]

def lemmatize_verbs(words):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word, pos='v') for word in words]

def normalize(words):
    words = to_lowercase(words)
    words = remove_punctuation_and_splchars(words)
    words = remove_stopwords(words)
    return words

def lemmatize(words):
    return lemmatize_verbs(words)

def normalize_and_lemmaize(input_text):
    input_text = remove_special_characters(input_text)
    words = nltk.word_tokenize(input_text)
    words = normalize(words)
    lemmas = lemmatize(words)
    return ' '.join(lemmas)

# ------------------------
# Prediction Functions
# ------------------------

def model_predict(text_series):
    """Predict the sentiment of input text series using the loaded ML model."""
    word_vector = count_vector.transform(text_series)
    tfidf_vector = tfidf_transformer.transform(word_vector)
    return model.predict(tfidf_vector)

def recommend_products(user_name):
    """
    Recommend top 20 products to a user and predict sentiments of reviews.
    Returns DataFrame with name, review text and predicted sentiment.
    """
    if user_name not in recommend_matrix.index:
        return pd.DataFrame()  # Return empty if user is not found

    product_list = pd.DataFrame(recommend_matrix.loc[user_name].sort_values(ascending=False)[0:20])
    product_frame = product_df[product_df.name.isin(product_list.index.tolist())].copy()

    product_frame['lemmatized_text'] = product_frame['reviews_text'].map(normalize_and_lemmaize)
    product_frame['predicted_sentiment'] = model_predict(product_frame['lemmatized_text'])

    return product_frame[['name', 'reviews_text', 'predicted_sentiment']]

def top5_products(df):
    """
    From the sentiment-tagged product DataFrame, return top 5 products
    with highest percentage of positive sentiment reviews.
    """
    if df.empty:
        return pd.DataFrame(columns=["name"])

    total_product = df.groupby(['name']).agg('count')
    rec_df = df.groupby(['name', 'predicted_sentiment']).agg('count').reset_index()
    
    merge_df = pd.merge(rec_df, total_product['reviews_text'], on='name')
    merge_df['%percentage'] = (merge_df['reviews_text_x'] / merge_df['reviews_text_y']) * 100
    merge_df = merge_df.sort_values(ascending=False, by='%percentage')

    output_products = pd.DataFrame(merge_df['name'][merge_df['predicted_sentiment'] == 1][:5])
    return output_products
