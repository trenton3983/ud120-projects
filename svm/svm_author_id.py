#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
from time import time
from email_preprocess import preprocess
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import collections


# features_train and features_test are the features for the training and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# Reduce the size of the training/label set to 1%
# features_train = features_train[:int(len(features_train) / 100)]
# labels_train = labels_train[:int(len(labels_train) / 100)]

# set the kernel type
k_type1 = 'linear'
k_type2 = 'rbf'

# c_param = [1.0, 10.0, 100.0, 1000.0, 10000.0]
c_param = [10000.0]

pred_element = [10, 26, 50]

for c in c_param:
    print(f'Testing penalty parameter C = {c}')
    clf = SVC(C=c, kernel=k_type2)
    t0 = time()

    # Train the dataset
    clf.fit(features_train, labels_train)
    print("Training time:", round(time()-t0, 3), "s")

    t1 = time()
    pred = clf.predict(features_test)

    print("Testing time:", round(time()-t1, 3), "s")
    for elem in pred_element:
        print(f'Element {elem} Prediction: {pred[elem]}')

    acc = accuracy_score(pred, labels_test)


    def submitAccuracy():
        return acc

    author_count = collections.Counter(pred)
    print(f'Number of emails predicted to be written by Sara(0) & Chris(1): {author_count}')


if __name__ == "__main__":

    print(f'Accuracy: {submitAccuracy()}\n')




