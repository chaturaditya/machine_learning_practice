Implements a decision tree algorithm that takes as arguments a matrix of examples, where each row is
one example and each column is one attribute, a row vector of attributes, and the target vector
which contains the binary targets. The target vector will split the training data (examples) into
positive examples for a given target and negative examples (all the other labels).

(Assignment from Course 395 Machine Learning - Imperial College London)

## Required packages
This program is written for Python 3+

Visualisation requires pydot and graphviz
```
sudo apt install python-pydot python-pydot-ng graphviz
```
Other packages: NumPy, SciPy, Pickle

## What to run
Run test.py to:
 * load the data into numpy arrays
 * import all decision trees from pickle files
 * get predictions using the testTrees function
 * print the evaluation metrics

Run main.py to:
 * load the data into numpy arrays
 * train decision trees on all 6 emotions
 * test the accuracy of the decision trees on the clean and noisy data
 * get results for 10-folds cross validation
 * visualize the trees (the results will be saved in /results)
 * save the trees as .p files (the results will be saved in /results)
