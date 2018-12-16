import scipy.io

from decision_tree import *

# Read in data
clean_data = scipy.io.loadmat("./Data/cleandata_students.mat")
x_clean = clean_data.get("x")
y_clean = clean_data.get("y")

# Read trees from pickle
dec_tree_anger = load_tree('./results/dec_tree_anger.p')
dec_tree_disgust = load_tree('./results/dec_tree_disgust.p')
dec_tree_fear = load_tree('./results/dec_tree_fear.p')
dec_tree_happiness = load_tree('./results/dec_tree_happiness.p')
dec_tree_sadness = load_tree('./results/dec_tree_sadness.p')
dec_tree_surprise = load_tree('./results/dec_tree_surprise.p')

trees = [dec_tree_anger, dec_tree_disgust, dec_tree_fear, dec_tree_happiness, dec_tree_sadness, dec_tree_surprise]

# Test trees function creates predictions
#prediction_all_emotions = testTrees(trees, x_clean)
#prediction_all_emotions = testTrees2(trees, x_clean)
prediction_all_emotions = testTrees3(trees, x_clean) # best approach

# Print all evaluation metrics
evaluate_predictions(prediction_all_emotions, y_clean)
