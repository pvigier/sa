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

predict('rf')