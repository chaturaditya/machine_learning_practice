import keras
import glob
from PIL import Image
import numpy as np
import skimage
from skimage.color import rgb2gray
import pandas as pd
import pickle
from keras.preprocessing.image import img_to_array, load_img

from assignment2_advanced.src.utils.load_fer import load_fer_data
from assignment2_advanced.src.utils.evaluate_cnn import confusion_matrix, confusion_matrix_percentage, \
    f1_measure, precision, recall, classification_rate


def test_deep_fer_model(img_folder = "data/", model="data/model/model.h5"):
    """
    Given a folder with images, load the images and your best model to predict the facial expressions of each image.
    Args:
    - img_folder: Path to the images to be tested
    Returns:
    - preds: A numpy vector of size N with N being the number of images in img_folder.
    """
    keras_model = keras.models.load_model(model)

    X_test, y_test = load_and_preprocess_imgs(img_folder)

    X_test = X_test.reshape(X_test.shape[0], 48,48,1)

    score = keras_model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    preds = keras_model.predict_classes(X_test)
    con_mat = confusion_matrix(preds,np.argmax(y_test, axis=1))
    print("confusion matrix: \n")
    print(con_mat)
    assert(np.sum(con_mat) == X_test.shape[0])
    print("confusion matrix percentage: \n")
    con_mat_perc = confusion_matrix_percentage(con_mat)
    print(con_mat_perc)

    print("\n classification rate:")
    print(classification_rate(con_mat))

    prec = precision(con_mat)
    print("\n precision:")
    print(np.round(prec,2))
    rec = recall(con_mat)
    print("\n recall:")
    print(np.round(rec,2))
    f1 = f1_measure(prec, rec)
    print("\n f1 measure:")
    print(np.round(f1,2))

    return preds


def load_and_preprocess_imgs(path_data):
    path_labels = path_data + "labels/labels_public.txt"
    path_x_mean = path_data + "model/x_mean.npy"
    path_x_std = path_data + "model/x_std.npy"

    x_mean = np.load(path_x_mean)
    x_std = np.load(path_x_std)

    image_list = sorted(glob.glob(path_data+"/Test/*.jpg"))
    labels = pd.read_table(path_labels, delimiter=",").sort_values('img')

    x = np.ndarray.astype(np.array([np.array(Image.open(fname)) for fname in image_list]), "float64")    # only use 1 channel. images are grayscale
    x = skimage.color.rgb2gray(x)
    x -= x_mean
    x /= x_std

    print("x_mean: "+ str(x.mean()))

    test_df = labels[labels['img'].apply(lambda x: x[:4] == 'Test')].reset_index()
    num_test = x.shape[0]
    y = np.zeros(num_test, int)
    for i in range(num_test):
        y[i] = test_df["emotion"][i]

    x = x.reshape(x.shape[0], 48,48,1)

    y = keras.utils.to_categorical(y,7)

    print("x shape:  " + str(x.shape))
    print("y shape:  " + str(y.shape))
    return x,y
    

def test_fer_model(img_folder, model="fer_model.pickle", mean_image_file='mean_image.pickle'):
    """
    Given a folder with images, load the images and your best model to predict the facial expressions of each image.
    Args:
    - img_folder: Path to the images to be tested
    Returns:
    - preds: A numpy vector of size N with N being the number of images in img_folder.
    """
    if not img_folder.endswith('/'):
        img_folder += '/'

    preds = None

    # load images
    #X_train, y_train, X_val, y_val = load_fer_data(img_folder, 0, 100)
    test_filelist = sorted(glob.glob(img_folder + '*.jpg'))
    print('test_filelist:', test_filelist[:5])
    num_test_images = len(test_filelist)
    X_test = np.ndarray(shape=(num_test_images, 48, 48, 3), dtype=np.float32)
    for i in range(num_test_images):
        if i % 100 == 0:
            print(i, 'of', num_test_images, 'testing images loaded')
        X_test[i] = img_to_array(load_img(test_filelist[i]))

    # subtract mean image
    mean_image = pickle.load(open(mean_image_file, 'rb'))
    X_test -= mean_image

    # make grayscale
    X_test = X_test.mean(axis=3)

    # load model
    final_model = pickle.load(open(model, 'rb'))

    # predict expressions
    scores = final_model.loss(X_test)
    preds = np.argmax(scores, axis=1)

    return preds

if __name__ == '__main__':
    # test fer_model - question 5
    img_folder = 'datasets/FER2013/Test/'
    predictions = test_fer_model(img_folder)
    print('preds:', predictions[:5])

    labels = pd.read_table('datasets/FER2013/labels_public.txt', delimiter=',').sort_values('img')
    test_df = labels[labels['img'].apply(lambda x: x[:4] == 'Test')].sort_values('img').reset_index()
    print('test_df:', test_df.head(5))

    con_mat = confusion_matrix(predictions, test_df['emotion'])
    print("confusion matrix: \n")
    print(con_mat)

    print("confusion matrix percentage: \n")
    con_mat_perc = confusion_matrix_percentage(con_mat)
    print(con_mat_perc)

    print("\n classification rate:")
    print(classification_rate(con_mat))

    prec = precision(con_mat)
    print("\n precision:")
    print(np.round(prec,2))
    rec = recall(con_mat)
    print("\n recall:")
    print(np.round(rec,2))
    f1 = f1_measure(prec, rec)
    print("\n f1 measure:")
    print(np.round(f1,2))


    # test deep_fer_model - question 6
    preds = test_deep_fer_model()

