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
    print('\n')

    names = [key for key, val in enron_data.items()]
    pp(names)
    print('\n')

    print(f'Number of dataset features: {len(enron_data)}')

    features_in_name = set([len(val) for key, val in enron_data.items()])
    print(f'Features for each name: {features_in_name} \n')

    poi_names = [key for key, val in enron_data.items() if val['poi'] is True]
    print('People who are a POI:')
    pp(poi_names)
    print('\n')
    print(f'There are {len(poi_names)} people in the dataset who are a POI.\n')

    name_snip = 'pren'.lower()
    find_name = [key for key, val in enron_data.items() if name_snip in key.lower()]
    for name in find_name:
        print(f"{name} - total stock value: ${enron_data[name]['total_stock_value']}")
        pp(enron_data[name])
        print('\n')

    has_salary = len([key for key, val in enron_data.items() if val['salary'] != 'NaN'])
    print(f'Number of people in dataset with a salary: {has_salary}\n')
    has_email = len([key for key, val in enron_data.items() if val['email_address'] != 'NaN'])
    print(f'Number of people in dataset with a email address: {has_email}\n')