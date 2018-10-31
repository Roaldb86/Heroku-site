<<<<<<< HEAD
from worldbankapp import app

import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sqlalchemy import create_engine
from nltk.corpus import stopwords

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_data', engine)

# load model
model = joblib.load("../models/classifier.pkl")
||||||| merged common ancestors
=======
from worldbankapp import app
>>>>>>> 0cbfbe8ce0552b3c7c50c3a3253ae3cede443cb3
