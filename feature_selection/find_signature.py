#!/usr/bin/python

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
from time import time
from sklearn import tree
from sklearn.metrics import accuracy_score
from pprint import pprint as pp

np.random.seed(42)


# The words (features) and authors (labels), already largely processed.
# These files should have been created from the previous (Lesson 10)
# mini-project.

files_path = Path(__file__).parents[1].joinpath('text_learning')

words_file = files_path.joinpath('your_word_data.pkl')
authors_file = files_path.joinpath('your_email_authors.pkl')
word_data = pickle.load(open(words_file, "rb"))
authors = pickle.load(open(authors_file, "rb"))


# test_size is the percentage of events assigned to the test set (the
# remainder go into training)
# feature matrices changed to dense representations for compatibility with
# classifier functions in versions 0.15.2 and earlier

features_train, features_test, labels_train, labels_test = train_test_split(word_data, authors, test_size=0.1,
                                                                            random_state=42)


vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test = vectorizer.transform(features_test).toarray()


# a classic way to overfit is to use a small number
# of data points and a large number of features;
# train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train = labels_train[:150]

# your code goes here

clf = tree.DecisionTreeClassifier()
t0 = time()
clf = clf.fit(features_train, labels_train)
print("Training time:", round(time() - t0, 3), "s")

t1 = time()
pred = clf.predict(features_test)
print("Testing time:", round(time() - t1, 3), "s")

accuracy = accuracy_score(pred, labels_test)

print('Accuracy: ', accuracy)

# Identify the most important feature
important_features = clf.feature_importances_

import_feature_and_number = [(i, item) for i, item in enumerate(important_features) if item >= 0.2]
pp(import_feature_and_number)

# Use TfIdf to Get the Most Important Word
feature_word = vectorizer.get_feature_names()

indices = np.argsort(important_features)[::-1]
print('Feature Ranking:')
for i in range(10):
    print(f'{i+1}: feature no.: {indices[i]}, importance: {important_features[indices[i]]}, '
          f'word: {feature_word[indices[i]]}')


