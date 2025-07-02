# model.py

import pandas as pd
import re
import nltk
import spacy
import string
import pickle as pk
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer, WordNetLemmatizer

# --------------------
# One-time Setup
# --------------------

def setup_nltk():
    for pkg in ['punkt', 'stopwords', 'wordnet', 'omw-1.4']:
        try:
            nltk.data.find(f'tokenizers/{pkg}')
        except LookupError:
            nltk.download(pkg)

def load_spacy_model():
    try:
        return spacy.load('en_core_web_sm', disable=['ner', 'parser'])
    except:
        raise RuntimeError("spaCy model 'en_core_web_sm' not found. Download and install it manually.")

# Call setup only when running the app locally or in a safe environment
setup_nltk()
nlp = load_spacy_model()

# --------------------
# Load Pretrained Artifacts
# --------------------

pickle_dir = Path('pickle_file')
required_files = [
    'count_vector.pkl',
    'tfidf_transformer.pkl',
    'model.pkl',
    'user_final_rating.pkl'
]

# Check existence of files
for fname in required_files:
    if not (pickle_dir / fname).exists():
        raise FileNotFoundError(f"Missing file: {fname}")

count_vector = pk.load(open(pickle_dir / 'count_vector.pkl', 'rb'))
tfidf_transformer = pk.load(open(pickle_dir / 'tfidf_transformer.pkl', 'rb'))
model = pk.load(open(pickle_dir / 'model.pkl', 'rb'))
recommend_matrix = pk.load(open(pickle_dir / 'user_final_rating.pkl', 'rb'))

# CSV
if not Path('sample30.csv').exists():
    raise FileNotFoundError("Missing file: sample30.csv")

product_df = pd.read_csv('sample30.csv')

# ------------------------
# Text Preprocessing Utils
# ------------------------

stopword_list = stopwords.words('english')

def remove_special_characters(text, remove_digits=True):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
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
    word_vector = count_vector.transform(text_series)
    tfidf_vector = tfidf_transformer.transform(word_vector)
    return model.predict(tfidf_vector)

def recommend_products(user_name):
    if user_name not in recommend_matrix.index:
        return pd.DataFrame()

    product_list = pd.DataFrame(recommend_matrix.loc[user_name].sort_values(ascending=False)[0:20])
    product_frame = product_df[product_df.name.isin(product_list.index.tolist())].copy()

    product_frame['lemmatized_text'] = product_frame['reviews_text'].map(normalize_and_lemmaize)
    product_frame['predicted_sentiment'] = model_predict(product_frame['lemmatized_text'])

    return product_frame[['name', 'reviews_text', 'predicted_sentiment']]

def top5_products(df):
    total_product = df.groupby(['name']).agg('count')
    rec_df = df.groupby(['name', 'predicted_sentiment']).agg('count').reset_index()

    merge_df = pd.merge(rec_df, total_product['reviews_text'], on='name')
    merge_df['%percentage'] = (merge_df['reviews_text_x'] / merge_df['reviews_text_y']) * 100
    merge_df = merge_df.sort_values(ascending=False, by='%percentage')

    output_products = pd.DataFrame(merge_df['name'][merge_df['predicted_sentiment'] == 1][:5])
    return output_products
