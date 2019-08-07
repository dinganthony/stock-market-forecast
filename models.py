'''models.py contains the classifier models called by the scripts.'''

import csv
import numpy as np
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras 
import matplotlib as mpl
from matplotlib import pyplot as plt

# Constants
DEBUG_MODE = True
NB_ALPHA = 1.0
TFIDF_FEATURES = 10000

'''The base SentimentClassifier class contains core logic 
for different models of stock classifiers.'''

class SentimentClassifier:

    def __init__(self, file, debug=DEBUG_MODE):
        self.debug = debug
        if self.debug:
            print("Reading from file:", file)
        stock_data = open(file)
        reader = csv.reader(stock_data)
        self.data = self.preprocess_data(reader)
        self.file = file
        #if self.debug:
        #   print("Preprocessed data:")
        #    print(self.data)
        self.create_model()

    def preprocess_data(self, reader):
        raise NotImplementedError("To be implemented in subclasses")
    
    def create_model(self):
        raise NotImplementedError("To be implemented in subclasses")
    
    def classify(self, text):
        raise NotImplementedError("To be implemented in subclasses")


''' The NaiveBayesClassifier uses a Bayes Net and the 'naive' assumption, 
that features in article headlines such as word frequencies are independent
of each other, to classify stock market movement.'''

class NaiveBayesClassifier(SentimentClassifier):

    def __init__(self, file, debug=DEBUG_MODE, alpha=NB_ALPHA):
        self.vocabulary = set()
        self.alpha = alpha
        SentimentClassifier.__init__(self, file, debug)
        
    # TODO: y values should be ints not strings when read from CSV
    def preprocess_data(self, reader):
        x_words = []
        y_values = []
        first_row_header = True
        for row in reader:
            # Remove table headers
            if first_row_header:
                first_row_header = False
                continue
            y_values.append(row[1])
            combined_headline = ''
            for headline in row[2:]:
                combined_headline += self.create_vocabulary(headline)
            x_words.append(combined_headline)
        return (x_words, y_values)
                
    def create_vocabulary(self, text):
        words = text.lower().replace(",", "").replace(".", "").replace("!", "")\
        .replace("?", "").replace(";", "").replace(":", "").replace("*", "")\
        .replace("(", "").replace(")", "").replace("/", "").split()
        # Filter out stopwords (e.g. "the", "an")
        stop = set(stopwords.words('english'))
        for word in words:
            if word not in stop:
                self.vocabulary.add(word)
        return text

    def create_model(self):
        # Split data into sets for training and testing
        x_train, x_test, y_train, y_test =\
            train_test_split(self.data[0], self.data[1], test_size=0.2)
        # Use tfidf to vectorize data
        tfidfv = TfidfVectorizer(tokenizer=None, max_features=TFIDF_FEATURES)
        tfidfv.fit(self.vocabulary)
        self.tfidf_vocab = tfidfv.vocabulary
        self.tfidfv = tfidfv
        x_train_tfidf = tfidfv.transform(x_train)
        x_test_tfidf = tfidfv.transform(x_test)
        # Fit MultinomialNB to training data
        self.nb = MultinomialNB(alpha=self.alpha)
        self.nb.fit(x_train_tfidf, y_train)
        # Test accuracy of NaiveBayesClassifier in predicting
        self.testing = self.nb.predict(x_test_tfidf)
        if DEBUG_MODE:
            print("Testing classifier accuracy:", accuracy_score(self.testing, y_test) * 100)
    
    def classify(self, text):
        text = self.tfidfv.transform([text])
        return self.nb.predict(text)


'''These classifiers are a work in progress and will be implemented after the NaiveBayesClassifier.'''

class SVMClassifier(NaiveBayesClassifier):

    def create_model(self):
        raise NotImplementedError()

    def classify(self, text):
        raise NotImplementedError()


class DeepLearningClassifier(SentimentClassifier):

    def preprocess_data(self, reader):
        raise NotImplementedError()

    def create_model(self):
        raise NotImplementedError()

    def classify(self, text):
        raise NotImplementedError()
