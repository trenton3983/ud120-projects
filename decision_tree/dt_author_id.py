#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""

from time import time
from email_preprocess import preprocess
from sklearn import tree
from sklearn.metrics import accuracy_score
from pprint import pprint as pp
import collections


# features_train and features_test are the features for the training and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels

features_train, features_test, labels_train, labels_test = preprocess()

# Reduce the size of the training/label set to 1%
# features_train = features_train[:int(len(features_train) / 100)]
# labels_train = labels_train[:int(len(labels_train) / 100)]

print(f'Number of features: {len(features_train[0])}')

# your code goes here

# min_samples = [2, 10, 20, 30, 40, 50]
min_samples = [40]
acc_samples = {}

for sample in min_samples:

    clf = tree.DecisionTreeClassifier(min_samples_split=sample)

    t0 = time()
    clf = clf.fit(features_train, labels_train)
    print("Training time:", round(time()-t0, 3), "s")

    t1 = time()
    pred = clf.predict(features_test)
    print("Testing time:", round(time() - t1, 3), "s")

    accuracy = accuracy_score(pred, labels_test)

    acc_samples[f'acc_min_samples_split_{sample}'] = accuracy

    print(f'Accuracy for min_samples_split = {sample}: {accuracy}\n')

    author_count = collections.Counter(pred)
    print(f'Number of emails predicted to be written by Sara(0) & Chris(1): {author_count}\n')


def submit_accuracies():
    """
    Only useful if testing more than one min_samples
    Return a dict with accuracies for each min_samples
    """
    return acc_samples


if __name__ == "__main__":

    pp(submit_accuracies())



