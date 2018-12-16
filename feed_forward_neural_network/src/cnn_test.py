import keras
import glob
from PIL import Image
import numpy as np
import skimage
from skimage.color import rgb2gray
import pandas as pd


from assignment2_advanced.src.utils.evaluate_cnn import confusion_matrix, confusion_matrix_percentage, \
    f1_measure, precision, recall, classification_rate


def prediction(img_folder = "../data/", model="../data/model/model.h5"):
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

if __name__ == '__main__':
    preds = prediction()

