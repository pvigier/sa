import collections
import math
import random
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from util import *

def build_dataset(words, vocabulary_size):
    count = [('UNK', -1)]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for i, (word, _) in enumerate(count):
        dictionary[word] = i
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count = unk_count + 1
        data.append(index)
    count[0] = ('UNK', unk_count)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

def generate_batch(data_index, data, batch_size, skip_window):
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1, 2*skip_window), dtype=np.int32)
    for i in range(batch_size):
        if data_index + 2 * skip_window - 1 > len(data):
            data_index = 0
        target = data_index + skip_window + i
        batch[i] = data[target]
        for j in range(skip_window):
            labels[i, 0, j] = data[target-j-1]
            labels[i, 0, skip_window+j] = data[target+j+1]
    return data_index + batch_size, batch, labels

def fit(data, vocabulary_size, batch_size, embedding_size, skip_window):
    num_sampled = 64 # Number of negative examples to sample.

    graph = tf.Graph()
    with graph.as_default(), tf.device('/cpu:0'):
        # Input data.
        train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1, 2*skip_window])
      
        # Variables.
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        softmax_weights = []
        softmax_biases = []
        for i in range(2*skip_window):
            softmax_weights.append(tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size))))
            softmax_biases.append(tf.Variable(tf.zeros([vocabulary_size])))
      
        # Model.
        # Look up embeddings for inputs.
        embed = tf.nn.embedding_lookup(embeddings, train_dataset)
        # Compute the softmax loss, using a sample of the negative labels each time.
        loss = tf.Variable(0, trainable=False, dtype=tf.float32)
        for i in range(2*skip_window):
            loss = tf.add(loss, tf.reduce_mean(tf.nn.sampled_softmax_loss(softmax_weights[i], softmax_biases[i], embed, train_labels[:, :, i], num_sampled, vocabulary_size)))

        # Optimizer.
        optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
      
        # Compute the similarity between minibatch examples and all embeddings.
        # We use the cosine distance:
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm

    num_steps = 100001
    data_index = 0

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print('Initialized')
        average_loss = 0
        for step in range(num_steps):
            data_index, batch_data, batch_labels = generate_batch(data_index, data, batch_size, skip_window)
            feed_dict = {train_dataset : batch_data, train_labels : batch_labels}
            _, l = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += l
            if step % 2000 == 0:
                if step > 0:
                    average_loss = average_loss / 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step %d: %f' % (step, average_loss / (2*skip_window)))
                average_loss = 0
        return normalized_embeddings.eval()

def word2vec(words, vocabulary_size, batch_size, embedding_size, skip_window, filename):
    data, count, dictionary, reverse_dictionary = build_dataset(words, vocabulary_size)
    embeddings = fit(data, vocabulary_size, batch_size, embedding_size, skip_window)
    pickle.dump(([reverse_dictionary[i] for i in range(1, len(reverse_dictionary))], embeddings), open(filename, 'wb'))

import zipfile

def read_data():
    """Extract the first file enclosed in a zip file as a list of words"""
    with zipfile.ZipFile('text8.zip') as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
        print(type(data), data[:1000])
    return data

if __name__ == '__main__': 
    filename = 'models/wordvect-128.pickle'
    words = read_data()
    word2vec(words, 50000, 100, 128, 1, filename)
    labels, embeddings = pickle.load(open(filename, 'rb'))
    plot_tsne(labels, embeddings)