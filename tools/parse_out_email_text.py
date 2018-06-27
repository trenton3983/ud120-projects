#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
from string import punctuation
from pathlib import Path


def parseOutText(f):
    """
    given an opened email file f, parse out all text below the
    metadata block at the top
    (in Part 2, you will also add stemming capabilities)
    and return a string that contains all the words
    in the email (space-separated)

    example use case:
    f = open("email_file_name.txt", "r")
    text = parseOutText(f)
    """

    f.seek(0)  # go back to beginning of file (annoying)
    all_text = f.read()

    # split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
        # remove punctuation
        text_string = content[1].translate(str.maketrans("", "", punctuation))

        # project part 2: comment out the line below
        # words = text_string

        # split the text string into individual words, stem each word,
        text_string_list = text_string.split()
        print(text_string_list)

        stemmer = SnowballStemmer('english')

        # text_stem_list = [stemmer.stem(word) for word in text_string_list]  # this line and the next -> same results
        text_stem_list = list(map(stemmer.stem, text_string_list))
        print(text_stem_list)
        # and append the stemmed word to words (make sure there's a single space between each stemmed word)

        # words = ' '.join(text_stem_list)  # return a string of joined words
        words = text_stem_list  # return a list of words

    return words


def main():
    location_dir = Path(__file__).parents[1].joinpath('text_learning')
    file_name = 'test_email.txt'
    test_email = location_dir.joinpath(file_name)

    ff = open(test_email, "r")
    text = parseOutText(ff)
    print(text)


if __name__ == '__main__':
    main()

