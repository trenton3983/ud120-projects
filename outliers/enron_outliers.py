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
del(data_dict['TOTAL'])  # TOTAL is the sum off all the people and skews the data if included
features = ["salary", "bonus"]
data = featureFormat(data_dict, features, _print=False)

# key_anno = list(data_dict.keys())

key_anno = [k for k, v in data_dict.items() if (v['salary'] or v['bonus']) != 'NaN']

fig, ax = plt.subplots()
ax.scatter(data[:, 0], data[:, 1])

salary = list(data[:, 0])
bonus = list(data[:, 1])

for i, txt in enumerate(key_anno):
    ax.annotate(txt, (salary[i], bonus[i]))

plt.xlabel('Salary')
plt.ylabel('Bonus')
plt.savefig('enron_outliers.png')
plt.show()



