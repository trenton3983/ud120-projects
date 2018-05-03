#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
from pathlib import Path
from pprint import pprint as pp


location_dir = Path(__file__).parents[1].joinpath('final_project')

file_name = 'final_project_dataset_unix.pkl'

file_name = location_dir.joinpath(file_name)
print(file_name)

enron_data = pickle.load(open(file_name, "rb"))


if __name__ == '__main__':

    pp(enron_data)
