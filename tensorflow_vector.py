import pickle
import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec
from util import get_reviews, get_clean_reviews

"""pickle_file = 'models/wordvect-128.pickle'
words, embeddings = pickle.load(open(pickle_file, 'rb'))
word_to_embedding = {label: i for i, label in enumerate(words)}"""

model = Word2Vec.load('models/300features_40minwords_10context.pickle')
index2word_set = set(model.index2word)

input_size = 300

def accuracy(predictions, labels):
    return (100.0 * np.sum((predictions >= 0.5) == labels) / predictions.shape[0])

def logistic_regression(train_dataset, train_labels, test_dataset, test_labels):
    # Metaparameters
    ALPHA = 1.0
    LAMBDA = 0.0

    graph = tf.Graph()
    with graph.as_default():
        # Inputs
        tf_train_dataset = tf.constant(train_dataset)
        tf_train_labels = tf.constant(train_labels)
        tf_test_dataset = tf.constant(test_dataset)

        # Parameters
        weights = tf.Variable(tf.truncated_normal([input_size, 1]))
        bias = tf.Variable(tf.zeros([1]))

        # Outputs
        logits = tf.matmul(tf_train_dataset, weights) + bias
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, tf_train_labels)) + LAMBDA * tf.nn.l2_loss(weights)

        # Optimizer
        optimizer = tf.train.GradientDescentOptimizer(ALPHA).minimize(loss)

        # Prediction
        train_prediction = tf.nn.sigmoid(logits)
        test_prediction = tf.nn.sigmoid(tf.matmul(tf_test_dataset, weights) + bias)

    num_steps = 10001

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print('Initialized')
        for step in range(num_steps):
            _, l, predictions = session.run([optimizer, loss, train_prediction])
            if (step % 100 == 0):
                print('Loss at step %d: %f' % (step, l))
                print('Training accuracy: %.1f%%' % accuracy(predictions, train_labels))
                print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

def review_to_features(review):
    features = np.zeros((1, input_size))
    for word in review.split():
        if word in index2word_set:
            features += model[word]
    return features

def reviews_to_dataset(reviews, sentiments):
    dataset = np.zeros((len(reviews), input_size), dtype=np.float32)
    labels = np.zeros((len(sentiments), 1), dtype=np.float32)
    for i, (review, label) in enumerate(zip(reviews, sentiments)):
        dataset[i,:] = review_to_features(review)
        labels[i,0] = label
    return dataset, labels

def fit():
    train_reviews = get_reviews('data/imdb/train_data.csv')
    train_dataset, train_labels = reviews_to_dataset(get_clean_reviews(train_reviews['review'], join_words=True), train_reviews['sentiment'])
    print('TRAIN:', train_dataset.shape, train_labels.shape)
    print(train_dataset, train_labels)
    
    test_reviews = get_reviews('data/imdb/test_data.csv')
    test_dataset, test_labels = reviews_to_dataset(get_clean_reviews(test_reviews['review'], join_words=True), test_reviews['sentiment'])
    print('TEST:', test_dataset.shape, test_labels.shape)
    print(test_dataset, test_labels)

    logistic_regression(train_dataset, train_labels, test_dataset, test_labels)

fit()