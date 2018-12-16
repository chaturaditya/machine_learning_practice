import numpy as np


#######################################         conf matrix         #######################################
def confusion_matrix(y_pred, y_true, num_classes=7):
    con_mat = np.zeros((num_classes,num_classes))
    for i in range(len(y_pred)):
        con_mat[y_pred[i]][y_true[i]] +=1
    return con_mat

#######################################         conf matrix percentage        #######################################
def confusion_matrix_percentage(con_mat):
    con_mat_percentage = np.transpose(np.copy(con_mat))
    for i in range(con_mat_percentage.shape[0]):
        row_sum = np.sum(con_mat_percentage[i,:])
        for j in range(con_mat_percentage.shape[1]):
            con_mat_percentage[i][j] /= row_sum

    return np.transpose(np.round(con_mat_percentage*100,2))

#######################################         F1 mesure        #######################################
def f1_measure(precision_rate, recall_rate):
    # initialize the Fa measure array
    rate = np.zeros((7))

    # compute the Fa measure for each class
    for i in range(7):
        if precision_rate[i] == 0 or recall_rate[i] == 0:
            rate[i] = 0
        else:
            rate[i] = 2 * ((precision_rate[i]/100 * recall_rate[i]/100) / (precision_rate[i]/100 + recall_rate[i]/100))*100

    # return the Fa measure
    return rate


#######################################         precision        #######################################
def precision(conf_matrix):
    # initialize the precision rate array
    rate = np.zeros((7))

    # compute the precision rate for each class
    for i in range(7):
        if sum(conf_matrix[:, i]) == 0:
            rate[i] = conf_matrix[i, i] * 100
        else:
            rate[i] = conf_matrix[i, i] * 100 / (sum(conf_matrix[:, i]))

    # return the precision rate
    return rate

#######################################         recall        #######################################
def recall(conf_mnatrix):
    # initialize the recall rate array
    rate = np.zeros((7))

    # compute the recall rate for each class
    for i in range(7):
        if sum(conf_mnatrix[i, :]) == 0:
            rate[i] = conf_mnatrix[i, i] * 100
        else:
            rate[i] = conf_mnatrix[i, i] * 100 / (sum(conf_mnatrix[i, :]))

    # return the recall rate
    return rate


#######################################         classification rate        #######################################
def classification_rate(conf_matrix):
    # compute the classification rate

    rate = sum(conf_matrix.diagonal()) / conf_matrix.sum()

    # return the classification rate
    return rate