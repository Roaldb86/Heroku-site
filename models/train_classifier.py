import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import seaborn as sns
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV

import pickle

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql('disaster_data', engine)
    X = df.message.values
    Y = df.drop(['message', 'id', 'original', 'genre'], axis=1).values
    target_names = df.drop(['message', 'id', 'original', 'genre'], axis=1).columns

    return X, Y, target_names

def tokenize(text):
    """Takes a text as input an returns a list of tokenized words"""
    stop_words = stopwords.words("english")
    text = re.sub(r"[^a-zA-Z0-9]", " ", text).lower().strip()
    words = word_tokenize(text)
    clean_words = [w for w in words if w not in stopwords.words("english")]
    tokens = [WordNetLemmatizer().lemmatize(w) for w in words if w not in stop_words]
    clean_tokens = [PorterStemmer().stem(w) for w in tokens]

    return clean_tokens
    
def build_model():
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=200, min_samples_leaf=2, min_samples_split=2), n_jobs=-1)),  
        ])
    parameters = {'clf__estimator__max_depth': [20, 50],
                  'clf__estimator__min_samples_leaf': [1, 4],
                  'clf__estimator__min_samples_split': [2, 7],
                  'clf__estimator__n_estimators': [200]}
    
    return GridSearchCV(estimator=pipeline, param_grid=parameters, verbose=10, n_jobs=-1)
    
    
def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    print("Accuracy of the model :", (y_pred == Y_test).mean())
    for i in y_pred:
        print(classification_report(Y_test, y_pred, target_names=category_names))
        break


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()