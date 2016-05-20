import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from util import *

def train_vectorizer(reviews):
    print('Learning vocabulary...')
    vectorizer = CountVectorizer(analyzer='word', stop_words=None, max_features=5000)
    vectorizer.fit(reviews)
    return vectorizer

def get_features(reviews, vectorizer):
    print('Creating bag of words...')
    return vectorizer.transform(reviews).toarray()

def get_word_features(word, vectorizer):
    return vectorizer.transform([word]).toarray().flatten()

def get_bag_of_words(reviews, filename, vectorizer=None):
    if os.path.exists(filename):
        print('Creating bag of words skipped - ' + filename + ' exists')
    else:
        clean_reviews = get_clean_reviews(reviews['review'], join_words=True)
        if not vectorizer:
            vectorizer = train_vectorizer(clean_reviews)
        features = get_features(clean_reviews, vectorizer)
        with open(filename, 'wb') as f:
            pickle.dump(features, f)
    return pickle.load(open(filename, 'rb')), vectorizer


def predict(algorithm):
    train = get_reviews('data/imdb/train_data.csv')
    train_features, vectorizer = get_bag_of_words(train, 'data/imdb/train_data_bow.pickle')

    classifier = train_classifier(algorithm, train_features, train)

    print('Free memory...')
    del train
    del train_features

    test = get_reviews('data/imdb/test_data.csv')
    test_features, _ = get_bag_of_words(test, 'data/imdb/test_data_bow.pickle', vectorizer)

    evaluate(test_features, test, classifier)

def get_words_weights():
    filename = 'models/bow-lr.pickle'
    if not os.path.exists(filename):
        train = get_reviews('data/imdb/train_data.csv')
        clean_train_reviews = clean_reviews(train['review'], join_words=True)
        vectorizer = train_vectorizer(clean_train_reviews)
        train_features = get_features(clean_train_reviews, vectorizer)
        classifier = train_classifier('lr', train_features, train)
        with open(filename, 'wb') as f:
            pickle.dump((vectorizer, classifier), f)
        print('Free memory...')
        del train
        del clean_train_reviews
        del train_features
    else:
        print('Training skipped - ' + filename + ' exists')

    with open(filename, 'rb') as f:
        vectorizer, classifier = pickle.load(f)
        sentiments = classifier.coef_.flatten()
        words = vectorizer.get_feature_names()
        word2sentiment = dict(zip(words, sentiments))

        n = 10
        sorted_words = sorted(word2sentiment.items(), key=lambda x: x[1])
        positive = reversed(sorted_words[-n:])
        negative = sorted_words[:n]
        print('NEGATIVE')
        for i, (word, sentiment) in enumerate(negative):
            print('{}. {} ({})'.format(i+1, word, sentiment))
        print('\nPOSITIVE')
        for i, (word, sentiment) in enumerate(positive):
            print('{}. {} ({})'.format(i+1, word, sentiment))

def get_learning_curve(algorithm):
    train = get_reviews('data/imdb/train_data.csv')
    train_features, vectorizer = get_bag_of_words(train, 'data/imdb/train_data_bow.pickle')

    test = get_reviews('data/imdb/test_data.csv')
    test_features, _ = get_bag_of_words(test, 'data/imdb/test_data_bow.pickle', vectorizer)

    show_learning_curve(algorithm, train_features, train, test_features, test)

get_learning_curve('lr')