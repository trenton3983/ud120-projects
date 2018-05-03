"""
convert dos linefeeds (crlf) to unix (lf)
usage: dos2unix.py

The pickle file has to be using Unix new lines otherwise at least Python 3.4's C pickle parser fails with exception:
pickle.UnpicklingError: the STRING opcode argument must be quoted
I think that some git versions may be changing the Unix new lines ('\n') to DOS lines ('\r\n').

You may use this code to change "word_data.pkl" to "word_data_unix.pkl" and then use
the new .pkl file on the script "nb_author_id.py":

https://github.com/udacity/ud120-projects/issues/46
https://stackoverflow.com/questions/45368255/error-in-loading-pickle
https://stackoverflow.com/questions/2613800/how-to-convert-dos-windows-newline-crlf-to-unix-newline-lf-in-a-bash-script/19702943#19702943
"""
from pathlib import Path

location_dir = Path(__file__).parents[1].joinpath('final_project')

file_name = 'final_project_dataset.pkl'

original = location_dir.joinpath(file_name)
destination = location_dir.joinpath(f"{file_name.split('.')[0]}_unix.pkl")

content = ''
outsize = 0
with open(original, 'rb') as infile:
    content = infile.read()
with open(destination, 'wb') as output:
    for line in content.splitlines():
        outsize += len(line) + 1
        output.write(line + str.encode('\n'))

print("Done. Saved %s bytes." % (len(content)-outsize))
