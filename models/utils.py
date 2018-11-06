import nltk
from nltk.corpus import stopwords
import re
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def tokenize(text):
    """Takes a text as input an returns a list of tokenized words"""
    stop_words = stopwords.words("english")
    text = re.sub(r"[^a-zA-Z0-9]", " ", text).lower().strip()
    words = word_tokenize(text)
    clean_words = [w for w in words if w not in stopwords.words("english")]
    tokens = [WordNetLemmatizer().lemmatize(w) for w in words if w not in stop_words]
    return [PorterStemmer().stem(w) for w in tokens]

    return clean_tokens
