from __future__ import print_function

import argparse
import h5py
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
from datetime import datetime
from tensorflow.python.lib.io import file_io
from keras import backend as K

K.set_image_dim_ordering('tf')
import sys
import io
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

print(keras.__version__)


def train_model(train_file="",
                job_dir='./tmp/co395', **args):
    logs_path = job_dir + '/logs/' + datetime.now().isoformat()
    print('Using logs_path located at {}'.format(logs_path))

    init_g = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_g)
        sess.run(init_l)

        f = io.BytesIO(file_io.read_file_to_string(train_file + 'x_train.npy'))

        print("loading x_train ...")
        X_train = tf.Variable(initial_value=np.load(f), name='X_train')
        sess.run(X_train.initializer)
        X_train = X_train.eval(session=sess)

        print("loading x_test ...")
        f = io.BytesIO(file_io.read_file_to_string(train_file + 'x_test.npy'))
        X_test = tf.Variable(initial_value=np.load(f), name='X_test')
        sess.run(X_test.initializer)
        X_test = X_test.eval(session=sess)

        print("loading y_train ...")
        f = io.BytesIO(file_io.read_file_to_string(train_file + 'y_train.npy'))
        y_train = tf.Variable(initial_value=np.load(f), name='y_train')
        sess.run(y_train.initializer)
        y_train = y_train.eval(session=sess)

        print("loading y_test ...")
        f = io.BytesIO(file_io.read_file_to_string(train_file + 'y_test.npy'))
        y_test = tf.Variable(initial_value=np.load(f), name='y_test')
        sess.run(y_test.initializer)
        y_test = y_test.eval(session=sess)

        X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)
        X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)
        num_classes = 7

        print("building CNN model ...")
        model = Sequential()

        # input size: Nx48x48x1
        model.add(Conv2D(64, (3, 3), padding='same', activation="relu", input_shape=X_train.shape[1:]))
        model.add(Conv2D(64, (3, 3), padding='same', activation="relu", input_shape=X_train.shape[1:]))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), padding='same', activation="relu", input_shape=X_train.shape[1:]))
        model.add(Conv2D(128, (3, 3), padding='same', activation="relu", input_shape=X_train.shape[1:]))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, (3, 3), padding='same', activation="relu", input_shape=X_train.shape[1:]))
        model.add(Conv2D(256, (3, 3), padding='same', activation="relu", input_shape=X_train.shape[1:]))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, (3, 3), padding='same', activation="relu", input_shape=X_train.shape[1:]))
        model.add(Conv2D(256, (3, 3), padding='same', activation="relu", input_shape=X_train.shape[1:]))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1024))
        model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
        model.add(Dropout(0.25))
        model.add(Dense(1024))
        model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
        model.add(Dropout(0.25))

        model.add(Dense(7))

        model.add(Activation('softmax'))

        ada = keras.optimizers.Adadelta(lr=0.1, epsilon=1e-08)
        model.compile(loss='categorical_crossentropy',
                      optimizer=ada,
                      metrics=['accuracy'])

        print("model summary: \n")
        print(model.summary())

        batch_size = 128
        epochs = 30
        augmentation = False

        tb_logs = keras.callbacks.TensorBoard(
            log_dir=logs_path,
            histogram_freq=0,
            write_graph=True,
            write_images=True,
            embeddings_freq=0)

        checkpoint = keras.callbacks.ModelCheckpoint("best_model_" +
                                                     str(datetime.now().isoformat()) + ".h5",
                                                     monitor='accuracy', verbose=1, save_best_only=True, mode='max')

        if augmentation:
            datagen = keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                vertical_flip=False)

            datagen.fit(X_train)
            datagen = datagen.flow(X_train, y_train, batch_size=batch_size)

            datagen_val = keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                vertical_flip=False)

            datagen_val.fit(X_test)
            datagen_val = datagen_val.flow(X_test, y_test, batch_size=batch_size)

            model.fit_generator(datagen, callbacks=[checkpoint, tb_logs], epochs=epochs,
                                steps_per_epoch=int(X_train.shape[0] / batch_size),
                                validation_data=datagen_val,
                                verbose=2)

        else:
            model.fit(X_train, y_train, batch_size=batch_size,
                      callbacks=[checkpoint, tb_logs],
                      epochs=epochs,
                      verbose=2,
                      shuffle=True,
                      validation_split=0.1)

        score = model.evaluate(X_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        model.save("model_acc=" + str(np.round(score[1], 2)) + "_time=" + str(datetime.now().isoformat()) + ".h5")

        # Save the model to the Cloud Storage bucket's jobs directory
        with file_io.FileIO('model.h5', mode='r') as input_f:
            with file_io.FileIO(job_dir + '/model.h5', mode='w+') as output_f:
                output_f.write(input_f.read())


if __name__ == '__main__':
    # for gcloud:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train-file',
        help='Cloud Storage bucket or local path to training data')
    parser.add_argument(
        '--job-dir',
        help='Cloud storage bucket to export the model and store temp files')
    args = parser.parse_args()
    arguments = args.__dict__
    train_model(**arguments)