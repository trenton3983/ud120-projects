#!/usr/bin/python

import pickle
import matplotlib.pyplot as plt
from feature_format import featureFormat, targetFeatureSplit
from pathlib import Path


# read in data dictionary, convert to numpy array
data_dict_dir = Path(__file__).parents[1].joinpath('final_project')
data_dict_file = 'final_project_dataset_unix.pkl'
data_dict_file = data_dict_dir.joinpath(data_dict_file)
data_dict = pickle.load(open(data_dict_file, "rb"))
features = ["salary", "bonus"]
data = featureFormat(data_dict, features, _print=True)


# your code below

plt.scatter(data[:-1, 0], data[:-1, 1])
plt.xlabel('Salary')
plt.ylabel('Bonus')
plt.savefig('enron_outliers.png')
plt.show()



