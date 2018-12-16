import glob
import pandas as pd
import numpy as np
import pickle
from keras.preprocessing.image import img_to_array, load_img

def load_fer_data(fer_folder, num_train=28708, num_test=3588, subtract_mean=True):
    if not fer_folder.endswith('/'):
        fer_folder += '/'

    path_labels = fer_folder + "/labels_public.txt"
    labels = pd.read_table(path_labels, delimiter=",").sort_values('img')
    train_df = labels[labels['img'].apply(lambda x: x[:5] == 'Train')].reset_index()
    test_df = labels[labels['img'].apply(lambda x: x[:4] == 'Test')].reset_index()

    print('Loading X_train and y_train...')
    X_train = np.ndarray(shape=(num_train, 48, 48, 3), dtype=np.float32)
    y_train = np.zeros(num_train, int)
    for i in range(num_train):
        if i % 100 == 0:
            print(i, 'of', num_train, 'training images loaded')
        X_train[i] = img_to_array(load_img(fer_folder + train_df['img'][i]))
        y_train[i] = train_df['emotion'][i]

    print('Loading X_test and y_test...')
    X_test = np.ndarray(shape=(num_test, 48, 48, 3), dtype=np.float32)
    y_test = np.zeros(num_test, int)
    for i in range(num_test):
        if i % 100 == 0:
            print(i, 'of', num_test, 'testing images loaded')
        X_test[i] = img_to_array(load_img(fer_folder + test_df['img'][i]))
        y_test[i] = test_df["emotion"][i]

    if subtract_mean:
        # get mean image of train data and pickle for test later
        mean_image = X_train.mean(axis=0)
        pickle.dump(mean_image, open('mean_image.pickle', 'wb'))
        X_train = X_train - mean_image
        X_test = X_test - mean_image

    print("X_train shape: " + str(X_train.shape))
    print("y_train shape: " + str(y_train.shape))
    print("X_test shape:  " + str(X_test.shape))
    print("y_test shape:  " + str(y_test.shape))

    return X_train, y_train, X_test, y_test
