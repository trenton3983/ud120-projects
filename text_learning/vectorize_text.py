#!/usr/bin/python

import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from parse_out_email_text import parseOutText

"""
    Starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.

    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""


from_sara = open("from_sara.txt", "r")
from_chris = open("from_chris.txt", "r")

from_data = []
word_data = []

# temp_counter is a way to speed up the development--there are
# thousands of emails from Sara and Chris, so running over all of them
# can take a long time
# temp_counter helps you only look at the first 200 emails in the list so you
# can iterate your modifications quicker
temp_counter = 0


for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
        path = path.strip()
        path = path[:-1] + '_'
        print(path)
        # only look at first 200 emails when developing
        # once everything is working, remove this line to run over full dataset
        # temp_counter += 1
        if temp_counter < 200:
            local_dir = r'E:\PythonProjects\UDACITY_DATA\07_Machine_Learning'
            # local_dir = 'C:/Data/'
            path = os.path.join(local_dir, path)
            print(path)
            email = open(path, "r")

            # use parseOutText to extract the text from the opened email
            text = parseOutText(email)

            # use str.replace() to remove any instances of the words
            remove = ["sara", "shackleton", "chris", "germani",
                      "sshacklensf", "cgermannsf", "cgerman",
                      "shackletonhouectect", "shackletonhouect"]

            text_cleaned = [item for item in text if item not in remove]
            text_cleaned = ' '.join(text_cleaned)

            # append the text to word_data
            word_data.append(text_cleaned)

            # append a 0 to from_data if email is from Sara, and 1 if email is from Chris
            from_data.append(0 if name == 'sara' else 1)

            email.close()

print("emails processed")
from_sara.close()
from_chris.close()

pickle.dump(word_data, open("your_word_data.pkl", "wb"))
pickle.dump(from_data, open("your_email_authors.pkl", "wb"))

print(word_data)
print(from_data)

print(word_data[152])


# in Part 4, do TfIdf vectorization here
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(word_data)

transformer = TfidfTransformer()
tfidf = transformer.fit_transform(X)

stop_words = vectorizer.get_stop_words()
print(tfidf)

feature_names = vectorizer.get_feature_names()
print(feature_names)
print(len(feature_names))
print(feature_names[34597])

print('\nStop Words:\n', stop_words)
print('Number of Stop Words:\n', len(stop_words))

