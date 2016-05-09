import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

# Preprocessing

stops = set(stopwords.words("english"))
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def get_reviews(path):
    return pd.read_csv(path, header=0, delimiter='\t', quoting=3 )

def remove_html(s):
    return BeautifulSoup(s, 'lxml').get_text()

def remove_punct(s):
    return re.sub(r'[^a-zA-Z]', ' ', s)

def get_words(s):
    return s.lower().split()

def remove_stop_words(words):
    return [w for w in words if not w in stops]

def review_to_words(review, keep_stop_words=False):
    words = get_words(remove_punct(remove_html(review)))
    if not keep_stop_words:
        words = remove_stop_words(words)
    return words

def clean_reviews(reviews, keep_stop_words=False, join_words=False):
    print('Cleaning and parsing the set of movie reviews...')
    clean_train_reviews = []
    for i, review in enumerate(reviews):
        if (i+1) % 1000 == 0:
            print('Review {}'.format(i+1))
        words = review_to_words(review, keep_stop_words)                                                             
        clean_train_reviews.append(' '.join(words) if join_words else words)
    return clean_train_reviews

# For word2vec

def review_to_sentences(review, keep_stop_words=True):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if raw_sentence:
            sentences.append(review_to_words(raw_sentence, keep_stop_words))
    return sentences

def get_sentences(reviews):
    print('Cleaning and get sentences from the set of movie reviews...')
    sentences = []
    for i, review in enumerate(reviews):
        if (i+1) % 1000 == 0:
            print('Review {}'.format(i+1))    
        sentences += review_to_sentences(review)
    return sentences

# Learning

def train_random_forest(data_features, train):
    print('Train random forest...')
    forest = RandomForestClassifier(n_estimators=100)
    forest.fit(data_features, train['sentiment'])
    return forest

def train_logistic_regression(data_features, train):
    print('Train logistic regression classifier...')
    classifier = LogisticRegression()
    classifier.fit(data_features, train['sentiment'])
    return classifier

def train_voting(data_features, train):
    print('Train voting classifier...')
    rf = RandomForestClassifier(n_estimators=100)
    lr = LogisticRegression()
    classifier = VotingClassifier(estimators=[('rf', rf), ('lr', lr)], voting='soft')
    classifier.fit(data_features, train['sentiment'])
    return classifier

def predict(data_features, test, classifier, path):
    print('Predict...')
    result = classifier.predict(data_features)
    output = pd.DataFrame(data={'id': test['id'], 'sentiment': result})
    output.to_csv(path, index=False, quoting=3)