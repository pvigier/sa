import logging
import os
import numpy as np
from gensim.models import Word2Vec
from util import *

# Set values for various parameters
num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

model_name = 'models/{}features_{}minwords_{}context.pickle'.format(num_features, min_word_count, context)

def train_word2vec(sentences):
    print('Training word2vec model...')
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = Word2Vec(sentences, workers=num_workers, size=num_features, min_count=min_word_count, window=context, sample=downsampling)
    model.init_sims(replace=True)
    model.save(model_name)

def review_to_features(words, model, index2word_set, num_features):
    features = np.zeros((1, num_features), dtype="float32")
    n = 0
    for word in words:
        if word in index2word_set: 
            n += 1
            features += model[word]
    return features / n


def get_features(reviews, model, num_features):
    print('Creating features...')
    index2word_set = set(model.index2word)
    features = np.zeros((len(reviews),  num_features), dtype="float32")
    for i, review in enumerate(reviews):
       if (i+1) % 1000 == 0:
           print('Review {}'.format(i+1))
       features[i,:] = review_to_features(review, model, index2word_set, num_features)
    return features

def predict(algorithm='rf'):
    train = get_reviews('data/imdb/train_data.csv')

    if not os.path.exists(model_name):
        #unlabeled_train = get_reviews('data/unlabeledTrainData.tsv')
        sentences = get_sentences(train['review'])# + get_sentences(unlabeled_train['review'])
        train_word2vec(sentences)

    model = Word2Vec.load(model_name)

    clean_train_reviews = get_clean_reviews(train['review'])
    train_data_features = get_features(clean_train_reviews, model, num_features)

    classifier = train_classifier(algorithm, train_data_features, train)

    # Free memory !
    del train
    del clean_train_reviews
    del train_data_features

    test = get_reviews('data/imdb/test_data.csv')
    clean_test_reviews = get_clean_reviews(test['review'])
    test_data_features = get_features(clean_test_reviews, model, num_features)

    evaluate(test_data_features, test, classifier)

predict('lr')