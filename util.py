import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Loading dataset

def get_reviews(path):
    return pd.read_csv(path, header=0)

# Preprocessing

stops = set(stopwords.words("english"))
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

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

def get_clean_reviews(reviews, keep_stop_words=False, join_words=False):
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

def train_classifier(algorithm, data_features, train):
    print('Train classifier...')
    estimators = []
    if 'rf' in algorithm:
        estimators.append(('rf', RandomForestClassifier(n_estimators=100)))
    if 'lr' in algorithm:
        estimators.append(('lr', LogisticRegression()))
    if 'mb' in algorithm:
        estimators.append(('mb', MultinomialNB()))
    # Training
    classifier = VotingClassifier(estimators=estimators, voting='soft')
    classifier.fit(data_features, train['sentiment'])
    return classifier

# Outputs

def predict(data_features, test, classifier, path):
    print('Predict...')
    results = classifier.predict(data_features)
    output = pd.DataFrame(data={'id': test['id'], 'sentiment': results})
    output.to_csv(path, index=False, quoting=3)

def evaluate(data_features, test, classifier):
    print('Evaluate...')
    results = classifier.predict(data_features)
    print('Accuracy:', accuracy_score(test['sentiment'], results))
    print('Confusion matrix:')
    print(confusion_matrix(test['sentiment'], results))