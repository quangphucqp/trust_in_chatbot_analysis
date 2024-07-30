# Directory structure
`input` contains the input files for the script. Data are sorted into `X` and `O` directories, corresponding to the treated and control sessions. The chatlog files are processed so that they always hold a shape of 7 rows corresponding to the column header, practice question, and the 5 main questions. The static recommendation has 6 rows, omitting the practice question.  

Beliefs are collected from the experiment. Chatlog are generated in the X sessions, and are processed and labeled. Rec are randomly generated based on the distribution of the chatlog, and must go together with the beliefs to interpret the results.


# Script
- `clean_chatlog`: unshuffle the chatlog files into question order 1-5, then turn them into wide format, and add session + computer_number columns.
- `clean_staticrec`: unshuffle the static rec files, add session + computer_number columns. 
- `clean_beliefs_and_survey`: unshuffle the beliefs files + rename the column for intperpretability.
- `join_data`: combine all rec, chatlog, and beliefs into one df
- `analysis`: run the analysis on the joined data