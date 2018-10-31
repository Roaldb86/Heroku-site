import json
import plotly
import pandas as pd
import nltk
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sqlalchemy import create_engine
from nltk.corpus import stopwords

from flask import render_template
from wrangling_scripts.wrangle_data import return_figures
nltk.download("stopwords")

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
__main__.tokenize = tokenize()
try:
    engine = create_engine('sqlite:///DisasterResponse.db')
    df = pd.read_sql_table('disaster_data', engine)
except:
    print("path error to sql db")
try:
    joblib.load('small_model.pickle','rb')
except:
    print("cant load model")


# # try:
# #
# except:
#     print("no model")


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/disaster')
def disaster():
        ## extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    category_counts = df.drop(['id','message','original','genre'], axis=1).sum().sort_values(ascending=False)
    category_names = list(category_counts.index)

    words = pd.Series(' '.join(df['message']).lower().split())
    top_10 = words[~words.isin(stopwords.words("english"))].value_counts()[:5]
    top_10_cat = list(top_10.index)

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Message category distrubutions',
                'yaxis': {
                    'title': "Count"
                },

            }
        },
        {
            'data': [
                Bar(
                    x=top_10_cat,
                    y=top_10
                )
            ],

            'layout': {
                'title': 'Most frequent words in the disaster messages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Words"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results)
