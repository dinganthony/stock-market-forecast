import csv
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras 
import matplotlib as mpl
from matplotlib import pyplot as plt

# Constants
DEBUG_MODE = True


'''The base SentimentClassifier class contains core logic 
for different models of stock classifiers.'''

class SentimentClassifier:

    def __init__(self, file, debug=DEBUG_MODE):
        self.debug = debug
        if self.debug:
            print("Reading from file:", file)
        stock_data = open(file)
        reader = csv.reader(stock_data)
        self.data = self.process_data(reader)
        self.file = file
        if self.debug:
            print("Processed data:")
            print(self.data)

    def process_data(self, reader):
        raise NotImplementedError("To be implemented in subclasses")
    
    def create_model(self):
        raise NotImplementedError("To be implemented in subclasses")
    
    def classify(self):
        raise NotImplementedError("To be implemented in subclasses")

class NaiveBayesClassifier(SentimentClassifier):

    def __init__(self, file, debug=DEBUG_MODE):
        super.__init__(file, debug)
        self.vocabulary = set()

    def process_data(self, reader):
        x_words = []
        y_values = []
        first_row_header = True
        for row in reader:
            # Remove table headers
            if first_row_header:
                first_row_header = False
                continue
            y_values.append(row[1])
            words_in_row = []
            for headline in row[2:]:
                words = self.createVocabulary(headline)
                words_in_row.extend(words)
            x_words.append(words_in_row)
        data = zip(x_words, y_values)
        np.random.shuffle(data)
        return data
                
    def createVocabulary(self, text):
        words = text.lower().replace(",", "").replace(".", "").replace("!", "")\
        .replace("?", "").replace(";", "").replace(":", "").replace("*", "")\
        .replace("(", "").replace(")", "").replace("/", "").split()
        for word in words:
            self.vocabulary.add(word)
        return words

    def create_model(self):
        x_train, y_train, x_test, y_test = train_test_split(self.data, test_size=0.2)

    def _train_classify(self):
        raise NotImplementedError()

    def _validate_classify(self):
        raise NotImplementedError()

    def _test_classify(self):
        raise NotImplementedError()
    
    def classify(self):
        raise NotImplementedError()

class DeepLearningClassifier(SentimentClassifier):

    def process_data(self, reader):
        raise NotImplementedError()

    def create_model(self):
        raise NotImplementedError()

    def classify(self):
        raise NotImplementedError()