'''script.py contains the main running block for sentiment analysis'''

import models

# Globals / soon-to-be-args
DATA_FILE = 'Combined_News_DJIA.csv'
TYPE = 'NaiveBayes'

def main(clf_type):
    test_headline = "Oil prices fall in Kuwait."
    print("Headline: ", test_headline)
    if clf_type == 'NaiveBayes':
        classifier = models.NaiveBayesClassifier(DATA_FILE, alpha=0.5)
    elif clf_type == 'SVM':
        classifier = models.SVMClassifier(DATA_FILE, alpha=1.0)
    elif clf_type == 'DeepLearning':
        classifier = models.DeepLearningClassifier(DATA_FILE)
    else:
        raise ValueError("Not a valid classifier type.")
    label = classifier.classify(test_headline)
    if label:
        print("Prediction: market will fall")
    else:
        print("Prediction: market will rise")
    
if __name__ == "__main__":
    main(TYPE)