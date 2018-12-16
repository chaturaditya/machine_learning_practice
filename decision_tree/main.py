# coding: utf-8

#### CW1 395 Machine Learning

import scipy.io
from decision_tree import *

#######################################                  Initialise               ######################################
clean_data = scipy.io.loadmat("./Data/cleandata_students.mat")
noisy_data = scipy.io.loadmat("./Data/noisydata_students.mat")

x_clean = clean_data.get("x")
print("clean dataset n_examples = " + str(x_clean.shape[0]) + " action_units = " + str(x_clean.shape[1]))
x_noisy = noisy_data.get("x")
print("noisy dataset n_examples = " + str(x_noisy.shape[0]) + " action_units = " + str(x_noisy.shape[1]))

y_clean = clean_data.get("y")
y_noisy = noisy_data.get("y")


###################################     Define binary emotions for clean data set   ####################################
print("\nDefine binary arrays for CLEAN data set ... ")
bin_anger =     subset(x_clean, y_clean, 1)
bin_disgust =   subset(x_clean, y_clean, 2)
bin_fear =      subset(x_clean, y_clean, 3)
bin_happiness = subset(x_clean, y_clean, 4)
bin_sadness =   subset(x_clean, y_clean, 5)
bin_surprise =  subset(x_clean, y_clean, 6)


##################################   Define binary emotions for noisy data set    ######################################
print("\nDefine binary arrays for NOISY data set ... ")
bin_anger_n =     subset(x_noisy, y_noisy, 1)
bin_disgust_n =   subset(x_noisy, y_noisy, 2)
bin_fear_n =      subset(x_noisy, y_noisy, 3)
bin_happiness_n = subset(x_noisy, y_noisy, 4)
bin_sadness_n =   subset(x_noisy, y_noisy, 5)
bin_surprise_n =  subset(x_noisy, y_noisy, 6)


#######################################         Learning on clean data set        ######################################
print("\nLearning on CLEAN data set ...")
dec_tree_anger =     decision_tree_learning(x_clean, range(45), bin_anger)
dec_tree_disgust =   decision_tree_learning(x_clean, range(45), bin_disgust)
dec_tree_fear =      decision_tree_learning(x_clean, range(45), bin_fear)
dec_tree_happiness = decision_tree_learning(x_clean, range(45), bin_happiness)
dec_tree_sadness =   decision_tree_learning(x_clean, range(45), bin_sadness)
dec_tree_surprise =  decision_tree_learning(x_clean, range(45), bin_surprise)


#######################################         Learning on noisy data set        ######################################
print("\nLearning on NOISY data set ...")
dec_tree_anger_n =     decision_tree_learning(x_noisy, range(45), bin_anger_n)
dec_tree_disgust_n =   decision_tree_learning(x_noisy, range(45), bin_disgust_n)
dec_tree_fear_n =      decision_tree_learning(x_noisy, range(45), bin_fear_n)
dec_tree_happiness_n = decision_tree_learning(x_noisy, range(45), bin_happiness_n)
dec_tree_sadness_n =   decision_tree_learning(x_noisy, range(45), bin_sadness_n)
dec_tree_surprise_n =  decision_tree_learning(x_noisy, range(45), bin_surprise_n)


#######################################           Test on clean data set          ######################################
print("\nTest perfect decision trees on the whole CLEAN data set ...")

print("test accuracy for perfect decision tree (anger)     = " + str(np.round(
    test_accuracy(x_clean, bin_anger, dec_tree_anger) * 100,2)) + str("%%"))
print("test accuracy for perfect decision tree (disgust)   = " + str(np.round(
    test_accuracy(x_clean, bin_disgust, dec_tree_disgust) * 100,2)) + str("%%"))
print("test accuracy for perfect decision tree (fear)      = " + str(np.round(
    test_accuracy(x_clean, bin_fear, dec_tree_fear) * 100,2)) + str("%%"))
print("test accuracy for perfect decision tree (happiness) = " + str(np.round(
    test_accuracy(x_clean, bin_happiness, dec_tree_happiness) * 100,2)) + str("%%"))
print("test accuracy for perfect decision tree (sadness)   = " + str(np.round(
    test_accuracy(x_clean, bin_sadness, dec_tree_sadness) * 100,2)) + str("%%"))
print("test accuracy for perfect decision tree (surprise)  = " + str(np.round(
    test_accuracy(x_clean, bin_surprise, dec_tree_surprise) * 100,2)) + str("%%"))


#######################################           Test on clean data set          ######################################
print("\nTest perfect decision trees on the whole NOISY data set ...")

print("test accuracy for perfect decision tree (anger)     = " + str(np.round(
    test_accuracy(x_noisy, bin_anger_n, dec_tree_anger_n) * 100,2)) + str("%%"))
print("test accuracy for perfect decision tree (disgust)   = " + str(np.round(
    test_accuracy(x_noisy, bin_disgust_n, dec_tree_disgust_n) * 100,2)) + str("%%"))
print("test accuracy for perfect decision tree (fear)      = " + str(np.round(
    test_accuracy(x_noisy, bin_fear_n, dec_tree_fear_n) * 100,2)) + str("%%"))
print("test accuracy for perfect decision tree (happiness) = " + str(np.round(
    test_accuracy(x_noisy, bin_happiness_n, dec_tree_happiness_n) * 100,2)) + str("%%"))
print("test accuracy for perfect decision tree (sadness)   = " + str(np.round(
    test_accuracy(x_noisy, bin_sadness_n, dec_tree_sadness_n) * 100,2)) + str("%%"))
print("test accuracy for perfect decision tree (surprise)  = " + str(np.round(
    test_accuracy(x_noisy, bin_surprise_n, dec_tree_surprise_n) * 100,2)) + str("%%"))


###################################          Test all emotions on clean data set         ###############################
print("\nTest all emotions on CLEAN data set...")
trees = [dec_tree_anger, dec_tree_disgust, dec_tree_fear, dec_tree_happiness, dec_tree_sadness, dec_tree_surprise]

prediction_all_emotions = testTrees3(trees, x_clean)

print("test accuracy for perfect decision tree (all emotions) = " + str(
    np.round(np.sum(prediction_all_emotions == y_clean) / len(y_clean), 2) * 100) + str("%%"))


###################################          Test all emotions on noisy data set         ###############################
print("\nTest all emotions on NOISY data set...")
trees_n = [dec_tree_anger_n, dec_tree_disgust_n, dec_tree_fear_n,
           dec_tree_happiness_n, dec_tree_sadness_n, dec_tree_surprise_n]

prediction_all_emotions_noisy = testTrees3(trees_n, x_noisy)

print("test accuracy for perfect decision tree (all emotions) = " + str(
    np.round(np.sum(prediction_all_emotions_noisy == y_noisy) / len(y_noisy), 2) * 100) + str("%%"))


############################     k-folds cross validation on clean and noisy data sets        ##########################
print("\n10-folds cross validation for CLEAN data set:\n")
k_fold_cross_validation(10, x_clean, y_clean)

print("\n10-folds cross validation for NOISY data set:\n")
k_fold_cross_validation(10, x_noisy, y_noisy)


#######################################         Visualisation         #######################################
if pydot_installed:
    draw_tree(dec_tree_anger, 'dec_tree_anger')
    draw_tree(dec_tree_disgust, 'dec_tree_disgust')
    draw_tree(dec_tree_fear, 'dec_tree_fear')
    draw_tree(dec_tree_happiness, 'dec_tree_happiness')
    draw_tree(dec_tree_sadness, 'dec_tree_sadness')
    draw_tree(dec_tree_surprise, 'dec_tree_surprise')
else:
    print('No visualisation because pydot not installed (see report).')


#######################################         save trees         #######################################
save_tree(dec_tree_anger, './results/dec_tree_anger.p')
save_tree(dec_tree_disgust, './results/dec_tree_disgust.p')
save_tree(dec_tree_fear, './results/dec_tree_fear.p')
save_tree(dec_tree_happiness, './results/dec_tree_happiness.p')
save_tree(dec_tree_sadness, './results/dec_tree_sadness.p')
save_tree(dec_tree_surprise, './results/dec_tree_surprise.p')


#######################################         load trees         #######################################
dec_tree_anger = load_tree('./results/dec_tree_anger.p')
print("")
print("test accuracy for pickled decision tree (anger)     = " + str(np.round(
    test_accuracy(x_clean, bin_anger, dec_tree_anger) * 100,2)) + str("%%"))
