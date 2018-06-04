#!/usr/bin/python

# Starter code for the regression mini-project.
#
# Loads up/formats a modified version of the dataset
# (why modified?  we've removed some trouble points
# that you'll find yourself in the outliers mini-project).
#
# Draws a little scatterplot of the training/testing data
#
# You fill in the regression code where indicated:


import pickle
from pathlib import Path
from feature_format import featureFormat, targetFeatureSplit
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


sort_keys_dir = Path(__file__).parents[1].joinpath('tools')
sort_keys_file_name = 'python2_lesson06_keys_unix.pkl'
sort_keys_file_name = sort_keys_dir.joinpath(sort_keys_file_name)
print(f'Sort Keys: {sort_keys_file_name}')
print(f'Sort Type: {type(sort_keys_file_name)}')

location_dir = Path(__file__).parents[1].joinpath('final_project')
file_name = 'final_project_dataset_modified_unix.pkl'
file_name = location_dir.joinpath(file_name)
print(f'Data Path: {file_name}')

dictionary = pickle.load(open(file_name, "rb"))

# list the features you want to look at--first item in the
# list will be the "target" feature
features_list = ["bonus", "salary"]
data = featureFormat(dictionary, features_list, remove_any_zeroes=True, sort_keys=sort_keys_file_name, _print=True)
target, features = targetFeatureSplit(data)

# training-testing split needed in regression, just like classification

feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5,
                                                                          random_state=42)
train_color = "b"
test_color = "r"

# Your regression goes here!
# Please name it reg, so that the plotting code below picks it up and
# plots it correctly. Don't forget to change the test_color above from "b" to
# "r" to differentiate training points from test points.

reg = LinearRegression()
reg.fit(feature_train, target_train)

# draw the scatterplot, with color-coded training and testing points

for feature, target in zip(feature_test, target_test):
    plt.scatter(feature, target, color=test_color)
for feature, target in zip(feature_train, target_train):
    plt.scatter(feature, target, color=train_color)

# labels for the legend
plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
plt.scatter(feature_test[0], target_test[0], color=train_color, label="train")


print(f'Slope Train: {reg.coef_}')
print(f'Intercept Train: {reg.intercept_}')

print('\n########## Stats on Test Dataset ##########')
print(f'r-squared score: {reg.score(feature_test, target_test)}')
print('\n########## Stats on Training Dataset ##########')
print(f'r-squared score: {reg.score(feature_train, target_train)}\n')

# draw the regression line, once it's coded
try:
    plt.plot(feature_test, reg.predict(feature_test), color='r')
except NameError:
    pass

reg.fit(feature_test, target_test)
print(f'Slope Test: {reg.coef_}')
print(f'Intercept Test: {reg.intercept_}')
plt.plot(feature_train, reg.predict(feature_train), color='b')

plt.xlabel(features_list[1])
plt.ylabel(features_list[0])
plt.legend()
plt.savefig('finance.png')
plt.show()
