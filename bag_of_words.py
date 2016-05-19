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
    data_features = vectorizer.transform(reviews)
    return data_features.toarray()

def get_word_features(word, vectorizer):
    return vectorizer.transform([word]).toarray().flatten()

def predict(algorithm='rf'):
    train = get_reviews('data/labeledTrainData.tsv')
    clean_train_reviews = clean_reviews(train['review'], join_words=True)
    vectorizer = train_vectorizer(clean_train_reviews)
    train_data_features = get_features(clean_train_reviews, vectorizer)

    if algorithm == 'rf':
        classifier = train_random_forest(train_data_features, train)
    elif algorithm == 'lr':
        classifier = train_logistic_regression(train_data_features, train)
    elif algorithm == 'v':
        classifier = train_voting(train_data_features, train)

    # Free memory !
    print('Free memory...')
    del train
    del clean_train_reviews
    del train_data_features

    test = get_reviews('data/testData.tsv')
    clean_test_reviews = clean_reviews(test['review'], join_words=True)
    test_data_features = get_features(clean_test_reviews, vectorizer)

    predict(test_data_features, test, classifier, 'results/bag_of_words_model_' + algorithm + '.csv')

def get_words_weights():
    filename = 'models/bow-lr'
    if not os.path.exists(filename):
        train = get_reviews('data/labeledTrainData.tsv')
        clean_train_reviews = clean_reviews(train['review'], join_words=True)
        vectorizer = train_vectorizer(clean_train_reviews)
        train_data_features = get_features(clean_train_reviews, vectorizer)
        classifier = train_logistic_regression(train_data_features, train)
        with open(filename, 'wb') as f:
            pickle.dump((vectorizer, classifier), f)
        print('Free memory...')
        del train
        del clean_train_reviews
        del train_data_features
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


        tests = ['like', 'love', 'dislike', 'hate', 'amazing', 'awful']


get_words_weights()