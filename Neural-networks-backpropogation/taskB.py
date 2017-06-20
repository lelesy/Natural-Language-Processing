# import pandas as pd
# import pymorphy2
import numpy as np
import tensorflow as tf
import tflearn
import re
import io
import json
import nltk
import csv

from collections import Counter
from sklearn.model_selection import train_test_split
from tflearn.data_utils import to_categorical
from nltk.stem.snowball import RussianStemmer
from nltk.tokenize import TweetTokenizer

negative_news = []
positive_news = []
VOCAB_SIZE = 1000
with io.open('train.json','r',encoding="utf-8") as data_file:
    news = json.load(data_file)
    news_collection = []
    for i in range(0,100):
        content = news[i]["text"]
        tokens = [j.lower() for j in nltk.word_tokenize(content)]
        new_content = ""
        for token in tokens:
            new_content=new_content+" "+token+" "
        if(news[i]["sentiment"]=="negative"):
            negative_news.append(new_content)
        elif(news[i]["sentiment"]=="positive"):
            positive_news.append(new_content)
f1 = open('positive.csv','w')
writer = csv.writer(f1)

stemer = RussianStemmer()
regex = re.compile('[\u0400-\u0500]+')
stem_cache = {}

def get_stem(token):
    stem = stem_cache.get(token, None)
    if stem:
        return stem
    token = regex.sub('', token).lower()
    stem = stemer.stem(token)
    stem_cache[token] = stem
    return stem
stem_count = Counter()
tokenizer = TweetTokenizer()

def count_unique_tokens_in_tweets(tweets):
    for tweet in tweets:
        tokens = tokenizer.tokenize(tweet)
        for token in tokens:
            stem = get_stem(token)
            stem_count[stem] +=1

count_unique_tokens_in_tweets(positive_news)
count_unique_tokens_in_tweets(negative_news)
vocab = sorted(stem_count, key=stem_count.get, reverse=True)
token_2_idx = {vocab[i] : i for i in range(VOCAB_SIZE)}

def tweet_to_vector(tweet, show_unknowns=False):
    vector = np.zeros(VOCAB_SIZE, dtype=np.int_)
    for token in tokenizer.tokenize(tweet):
        stem = get_stem(token)
        idx = token_2_idx.get(stem, None)
        if idx is not None:
            vector[idx] = 1
        elif show_unknowns:
            print("Unknown token: {}".format(token))
    return vector

tweet_vectors = np.zeros((len(negative_news) + len(positive_news), VOCAB_SIZE),dtype=np.int_)
tweets = []
for i in range(len(negative_news)):
    tweets.append(negative_news[i])
    tweet_vectors[i] = tweet_to_vector(negative_news[i])
for i in range(len(positive_news)):
    tweets.append(positive_news[i])
    tweet_vectors[i] = tweet_to_vector(positive_news[i])

labels = np.append(np.zeros(len(negative_news), dtype=np.int_),np.ones(len(positive_news), dtype=np.int_))


X = tweet_vectors
y = to_categorical(labels, 2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

def build_model(learning_rate=0.1):
    tf.reset_default_graph()

    net = tflearn.input_data([None, VOCAB_SIZE])
    net = tflearn.fully_connected(net, 125, activation='ReLU')
    net = tflearn.fully_connected(net, 25, activation='ReLU')
    net = tflearn.fully_connected(net, 2, activation='softmax')
    regression = tflearn.regression(
        net,
        optimizer='sgd',
        learning_rate=learning_rate,
        loss='categorical_crossentropy')

    model = tflearn.DNN(net)
    return model
model = build_model(learning_rate=0.75)
model.fit(
    X_train,
    y_train,
    validation_set=0.1,
    show_metric=True,
    batch_size=128,
    n_epoch=30)
predictions = (np.array(model.predict(X_test))[:,0] >= 0.5).astype(np.int_)
accuracy = np.mean(predictions == y_test[:,0], axis=0)
print("Accuracy: ", accuracy)

def test_tweet(tweet):
    tweet_vector = tweet_to_vector(tweet, True)
    positive_prob = model.predict([tweet_vector])[0][1]
    print('Original tweet: {}'.format(tweet))
    print('P(positive) = {:.5f}. Result: '.format(positive_prob),
          'Positive' if positive_prob > 0.5 else 'Negative')
def test_tweet_number(idx):
    test_tweet(tweets[idx])
