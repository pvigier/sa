import numpy as np
from gensim.models import Doc2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from util import *

model_name = 'models/imdb.d2v'
vocabulary_size = 10000
vector_size = 300

def train_vectorizer(reviews, r=(1, 1)):
    print('Learning vocabulary {}...'.format(str(r)))
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=r, stop_words=None, max_features=vocabulary_size)
    vectorizer.fit(reviews)
    return vectorizer

def get_features(reviews, clean_reviews, model, vectorizer):
	print('Creating features...')
	features = np.zeros((len(reviews), vocabulary_size + vector_size))
	features[:, :vocabulary_size] = vectorizer.transform(clean_reviews).toarray()
	for i, (review_id) in enumerate(reviews['id']):
		features[i, vocabulary_size:] = model.docvecs[review_id]
	return features

def predict(algorithm, r=(1, 1)):
    train = get_reviews('data/imdb/train_data.csv')
    clean_train_reviews = get_clean_reviews(train['review'], keep_stop_words=True, join_words=True)

    vectorizer = train_vectorizer(clean_train_reviews, r)
    model = Doc2Vec.load(model_name)

    train_features = get_features(train, clean_train_reviews, model, vectorizer)

    classifier = train_classifier(algorithm, train_features, train)

    print('Free memory...')
    del train
    del clean_train_reviews
    del train_features

    test = get_reviews('data/imdb/test_data.csv')
    clean_test_reviews = get_clean_reviews(test['review'], keep_stop_words=True, join_words=True)
    test_features = get_features(test, clean_test_reviews, model, vectorizer)

    evaluate(test_features, test, classifier)

predict('lr', (1, 2))