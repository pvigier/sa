import re
import pandas as pd
import numpy as np
import nltk
import random
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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

def review_to_words(review, keep_stop_words=True):
    words = get_words(remove_punct(remove_html(review)))
    if not keep_stop_words:
        words = remove_stop_words(words)
    return words

def get_clean_reviews(reviews, keep_stop_words=True, join_words=False):
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

def train_classifier(algorithm, features, train):
    print('Train classifier ({})...'.format(algorithm))
    estimators = []
    if 'rf' in algorithm:
        estimators.append(('rf', RandomForestClassifier(n_estimators=100)))
    if 'lr' in algorithm:
        estimators.append(('lr', LogisticRegression()))
    if 'mb' in algorithm:
        estimators.append(('mb', MultinomialNB()))
    # Training
    classifier = VotingClassifier(estimators=estimators, voting='soft')
    classifier.fit(features, train['sentiment'])
    return classifier

# Outputs

def predict(features, test, classifier, path):
    print('Predict...')
    results = classifier.predict(features)
    output = pd.DataFrame(data={'id': test['id'], 'sentiment': results})
    output.to_csv(path, index=False, quoting=3)

def evaluate(features, test, classifier):
    print('Evaluate...')
    results = classifier.predict(features)
    print('Accuracy:', accuracy_score(test['sentiment'], results))
    print('Confusion matrix:')
    print(confusion_matrix(test['sentiment'], results))

def shuffle_both(features, df):
    index = list(df.index)
    random.shuffle(index)
    shuffled_df = df.reindex(index)
    shuffled_features = np.zeros(features.shape)
    for i, j in enumerate(index):
        shuffled_features[i,:] = features[j,:]
    return shuffled_features, shuffled_df

def show_learning_curve(algorithm, train_features, train, test_features, test):
    x, y = [], []
    train_features, train = shuffle_both(train_features, train)
    steps = [(10, 100), (100, 1000), (1000, 25001)]
    for step, lim in steps:
        for batch_size in range(step, lim, step):
            print('Batch size:', batch_size)
            classifier = train_classifier(algorithm, train_features[:batch_size], train[:batch_size])
            results = classifier.predict(test_features)
            x.append(batch_size)
            y.append(accuracy_score(test['sentiment'], results))
            print('Accuracy:', y[-1])
    plt.plot(x, y)
    plt.xlabel('training set size')
    plt.ylim(0, 1)
    plt.ylabel('accuracy')
    plt.show()

def plot_pca(X, y):
    colors = ['b', 'r']
    pca = PCA(n_components=2)
    X_r = pca.fit(X).transform(X)
    plt.figure()
    for i, c in enumerate(colors):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=str(i))
    plt.legend()
    plt.title('PCA')

def plot_lda(X, y):
    colors = ['b', 'r']
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_r = lda.fit(X, y).transform(X)
    plt.figure()
    for i, c in enumerate(colors):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=str(i))
    plt.legend()
    plt.title('PCA')

def plot_tsne(labels, embeddings):
    num_points = 400
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    two_d_embeddings = tsne.fit_transform(embeddings[1:num_points+1, :])
    #plt.figure(figsize=(10, 10))
    for i, label in enumerate(labels[1:num_points+1]):
        x, y = two_d_embeddings[i,:]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.show()