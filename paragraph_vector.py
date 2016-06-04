import logging
import os
import numpy as np
import random
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from util import *

# Set values for various parameters
num_features = 300    # Word vector dimensionality                      
min_word_count = 1    # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-4   # Downsample setting for frequent words

model_name = 'models/imdb.d2v'

class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources
    
    def __iter__(self):
        for source in self.sources:
            for review_id, review in zip(source['id'], source['review']):
                yield LabeledSentence(review_to_words(review), [review_id])
    
    def to_array(self):
        self.sentences = []
        for source in self.sources:
            for review_id, review in zip(source['id'], source['review']):
                self.sentences.append(LabeledSentence(review_to_words(review), [review_id]))
        return self.sentences
    
    def sentences_perm(self):
        random.shuffle(self.sentences)
        return self.sentences

def train_doc2vec(sentences):
    print('Training word2vec model...')
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = Doc2Vec(workers=num_workers, size=num_features, min_count=min_word_count, window=context, sample=downsampling)
    model.build_vocab(sentences.to_array())
    for epoch in range(10):
        print('EPOCH', epoch)
        model.train(sentences.sentences_perm())
    model.save(model_name)

def get_features(reviews, model):
    print('Creating features...')
    features = np.zeros((len(reviews),  num_features), dtype="float32")
    for i, review_id in enumerate(reviews['id']):
       if (i+1) % 1000 == 0:
           print('Review {}'.format(i+1))
       features[i,:] = model.docvecs[review_id]
    return features

def predict(algorithm='rf'):
    train = get_reviews('data/imdb/train_data.csv')
    test = get_reviews('data/imdb/test_data.csv')

    sentences = LabeledLineSentence([train, test])

    if not os.path.exists(model_name):
        train_doc2vec(sentences)

    model = Doc2Vec.load(model_name)
    print(model.most_similar('good'))

    train_features = get_features(train, model)

    classifier = train_classifier(algorithm, train_features, train)

    # Free memory !
    del train
    del train_features

    test_features = get_features(test, model)

    evaluate(test_features, test, classifier)

#test()
predict('lr')