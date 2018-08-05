#!/usr/bin/python

import pickle
from feature_format import featureFormat, targetFeatureSplit
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import itertools

"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

data_dict_file = Path(__file__).parents[1].joinpath('final_project/final_project_dataset_unix.pkl')
data_dict = pickle.load(open(data_dict_file, 'rb'))

# add more features to features_list!
features_list = ["poi", "salary"]

sort_keys = Path(__file__).parents[1].joinpath('tools/python2_lesson14_keys_unix.pkl')

data = featureFormat(data_dict, features_list, sort_keys=sort_keys)
labels, features = targetFeatureSplit(data)

# your code goes here
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3,
                                                                            random_state=42)

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
score_ = clf.score(features_test, labels_test)
print('Score: ', score_)  # 0.7241379310344828


# Lesson 15.28: Number of POIs in Test Set
print('\nLesson 15.28: Number of POIs in Test Set')
prediction_ = clf.predict(features_test)
unique, counts = np.unique(prediction_, return_counts=True)
pois_in_test_set = dict(zip(unique, counts))
print('Test Set: ', pois_in_test_set)
print('POIs in Test Set: ', pois_in_test_set[1])

# Lesson 15.29: Number of People in Test Set
print('\nLesson 15.29: Number of People in Test Set')
people_count_in_test_set = len(prediction_)  # 29
print('Number of People in the Test Set: ', people_count_in_test_set)

# Lesson 15.30: Accuracy of Biased Identifier
print('\nLesson 15.30: Accuracy of Biased Identifier')
accuracy_ = accuracy_score(labels_test, prediction_)
accuracy_if_0 = pois_in_test_set[0]/people_count_in_test_set  # 0.8620689655172413
print('Accuracy: ', accuracy_)
print('If the identifier predicted 0 for everyone in the test set, what would the accuracy be?: ', accuracy_if_0)

# Lesson 15.31: Number of True Positives
print('\nLesson 15.31: Number of True Positives')
poi_confusion_matrix = confusion_matrix(labels_test, prediction_, labels=[0, 1])
print('Confusion Matrix:')
print(poi_confusion_matrix)
print('Number of True Positives: ', poi_confusion_matrix[1][1])


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    From:
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-
        examples-model-selection-plot-confusion-matrix-py
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


np.set_printoptions(precision=2)
class_names = ['Not POI', 'POI']

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(poi_confusion_matrix, classes=class_names, title='Confusion matrix 15.31, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(poi_confusion_matrix, classes=class_names, normalize=True,
                      title='Confusion matrix 15.31, normalized')

plt.show()

# Lesson 15.32: Unpacking Into Precision and Recall
print('\nLesson 15.32: Unpacking Into Precision and Recall')
precision_ = precision_score(labels_test, prediction_)
print('Precision: ', precision_)

# Lesson 15.33: Recall of Your POI Identifier
print('\nLesson 15.33: Recall of Your POI Identifier')
recall_ = recall_score(labels_test, prediction_)
print('Recall: ', recall_)

# Lesson 15.34: How Many True Positives?
print('\nLesson 15.34: How Many True Positives?')
predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

conf_mat_15_34 = confusion_matrix(true_labels, predictions)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(conf_mat_15_34, classes=class_names, title='Confusion matrix 15.34, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(conf_mat_15_34, classes=class_names, normalize=True,
                      title='Confusion matrix 15.34, normalized')

plt.show()

print('How many true positives?: ', conf_mat_15_34[1][1])

# Lesson 15.35: How Many True Negatives?
print('\nLesson 15.35: How Many True Negatives?')
print('How many true negatives?: ', conf_mat_15_34[0][0])

# Lesson 15.36: How Many False Positives?
print('\nLesson 15.36: How Many False Positives?')
print('How many true negatives?: ', conf_mat_15_34[0][1])

# Lesson 15.37: How Many False Negatives?
print('\nLesson 15.37: How Many False Negatives?')
print('How many true negatives?: ', conf_mat_15_34[1][0])

# Lesson 15.38: Precision
print('\nLesson 15.38: Precision')
print('Precision: ', precision_score(true_labels, predictions))

# Lesson 15.39: Recall
print('\nLesson 15.39: Recall')
print('Precision: ', recall_score(true_labels, predictions))
