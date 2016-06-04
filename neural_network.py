import numpy as np
import tensorflow as tf
from gensim.models import Doc2Vec
from util import *

batch_size = 128
H = 1024
ETA = 0.5
LAMBDA = 0.001

model_name = 'models/imdb.d2v'
num_features = 300

def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

def accuracy(predictions, labels):
    return (100.0 * np.sum((predictions >= 0.5) == labels) / predictions.shape[0])

def neural_network(train_dataset, train_labels, test_dataset, test_labels):
    print('Learning...')
    graph = tf.Graph()
    with graph.as_default():
        # Input data.
        x = tf.placeholder(tf.float32, shape=(None, num_features))
        y_ = tf.placeholder(tf.float32, shape=(None, 1))

        # Variables.
        W1 = tf.Variable(tf.truncated_normal([num_features, H]))
        B1 = tf.Variable(tf.zeros([H]))
        W2 = tf.Variable(tf.truncated_normal([H, 1]))
        B2 = tf.Variable(tf.zeros([1]))

        # Training computation.
        a2 = tf.nn.relu(tf.matmul(x, W1) + B1)
        y = tf.matmul(a2, W2) + B2
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y, y_)) + LAMBDA * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2))

        # Optimizer.
        optimizer = tf.train.GradientDescentOptimizer(ETA).minimize(loss)

        # Predictions.
        predictions = tf.nn.sigmoid(y)

    num_steps = 10001

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        shuffle_in_unison(train_dataset, train_labels)
        print("Initialized")
        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {x: batch_data, y_: batch_labels}
            _, l, p = session.run([optimizer, loss, predictions], feed_dict=feed_dict)
            if (step % 500 == 0):
                print("Minibatch loss at step {}: {}".format(step, l))
                print("Minibatch accuracy: {}".format(accuracy(predictions.eval(feed_dict={x: train_dataset}), train_labels)))
                print("Test accuracy: {}".format(accuracy(predictions.eval(feed_dict={x: test_dataset}), test_labels)))

        print("Test accuracy: {}".format(accuracy(predictions.eval(feed_dict={x: test_dataset}), test_labels)))

def get_features(reviews, model):
    print('Creating features...')
    features = np.zeros((len(reviews),  num_features), dtype="float32")
    for i, review_id in enumerate(reviews['id']):
       if (i+1) % 1000 == 0:
           print('Review {}'.format(i+1))
       features[i,:] = model.docvecs[review_id]
    return features

def predict():
    train = get_reviews('data/imdb/train_data.csv')
    test = get_reviews('data/imdb/test_data.csv')

    model = Doc2Vec.load(model_name)

    train_features = get_features(train, model)
    train_labels = train['sentiment'].as_matrix().reshape((len(train), 1))
    test_features = get_features(test, model)
    test_labels = test['sentiment'].as_matrix().reshape((len(test), 1))

    neural_network(train_features, train_labels, test_features, test_labels)

predict()