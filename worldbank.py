import json
import plotly
import pandas as pd
import nltk
import pickle
from nltk.stem import WordNetLemmatizer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sqlalchemy import create_engine
from nltk.corpus import stopwords

from flask import render_template
from wrangling_scripts.wrangle_data import return_figures
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from train_classifier_for_web import tokenize


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)


# def tokenize(text):
#     """Takes a text as input an returns a list of tokenized words"""
#     stop_words = stopwords.words("english")
#     text = re.sub(r"[^a-zA-Z0-9]", " ", text).lower().strip()
#     words = word_tokenize(text)
#     clean_words = [w for w in words if w not in stopwords.words("english")]
#     tokens = [WordNetLemmatizer().lemmatize(w) for w in words if w not in stop_words]
#     return [PorterStemmer().stem(w) for w in tokens]
#
#     return clean_tokens

def main():
    global df
    global model

    try:
        engine = create_engine('sqlite:///DisasterResponse.db')
        df = pd.read_sql_table('disaster_data', engine)
    except:
        print("path error to sql db")
    try:
        model = joblib.load('web_model.sav','rb')
    except Exception as e:
        print("cant load model", e)




@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/flowers')
def flowers():
    return render_template('flowers.html')

@app.route('/disaster')
def disaster():
    global df
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
    global model
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

if __name__ == '__main__':
    main()
