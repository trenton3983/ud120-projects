#!/usr/bin/python

import pickle
from feature_format import featureFormat, targetFeatureSplit
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

"""
Starter code for the validation mini-project.
The first step toward building your POI identifier!

Start by loading/formatting the data

After that, it's not our code anymore--it's yours!
"""

data_dict_file = Path(__file__).parents[1].joinpath('final_project/final_project_dataset_unix.pkl')

data_dict = pickle.load(open(data_dict_file, 'rb'))

# first element is our labels, any added elements are predictor
# features. Keep this the same for the mini-project, but you'll
# have a different feature list when you do the final project.
features_list = ["poi", "salary"]

sort_keys = Path(__file__).parents[1].joinpath('tools/python2_lesson13_keys_unix.pkl')

data = featureFormat(data_dict, features_list, sort_keys=sort_keys)
labels, features = targetFeatureSplit(data)

"""
Implement Decision Tree Classifier trained on all the data
"""
clf = DecisionTreeClassifier()
clf.fit(features, labels)
dtc_score = clf.score(features, labels)
print(dtc_score)  # 0.9894736842105263

"""
Add Split and Training
"""
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3,
                                                                            random_state=42)

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
dtc_score = clf.score(features_test, labels_test)
print(dtc_score)  # 0.7241379310344828
