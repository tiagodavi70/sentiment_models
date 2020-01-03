import keras
import tensorflow as tf
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from keras import regularizers
import argparse
import numpy as np
import os
import sklearn
import warnings
warnings.filterwarnings('ignore') # sklearn warns about bad std on verbo dataset

# import model_utils as utils


def load_dataset(path, test=False):
    train = np.genfromtxt(path + '/train.csv', delimiter=',')
    val = np.genfromtxt(path + '/val.csv', delimiter=',')

    x_train = train[:,:-1]
    y_train = train[:,-1:].astype(np.int8)

    x_val = val[:,:-1]
    y_val = val[:,-1:].astype(np.int8)

    x_train =np.expand_dims(x_train, axis=2)
    x_val = np.expand_dims(x_val, axis=2)

    y_train = keras.utils.np_utils.to_categorical(y_train, num_classes=None)
    y_val = keras.utils.np_utils.to_categorical(y_val, num_classes=None)


    return (x_train, y_train), (x_val, y_val)

####### model_utils
def createModel(input_shape,num_classes,name=""):

    model = Sequential()

    model.add(Conv1D(64, 30, padding='same',
                     input_shape=input_shape))
    model.add(Activation('relu'))

    model.add(Conv1D(128, 25, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Conv1D(128, 25, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Conv1D(128, 25, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Conv1D(128, 25, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Conv1D(128, 12, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Conv1D(128, 12, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Conv1D(128, 12, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Conv1D(128, 9, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Conv1D(128, 9, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Conv1D(128, 9, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Conv1D(128, 9, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Conv1D(128, 9, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Conv1D(256, 5, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Conv1D(256, 5,padding='same'))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model


import pickle
import copy
def saveModel(dir_path, model, scores, historic, saveH5=True):
    os.makedirs(dir_path)

    if saveH5:
        model.save(os.path.join(dir_path, 'model.h5'))

    pickle.dump(scores, open(os.path.join(dir_path, 'scores.p'), "wb"))

    hist_nomodel = copy.copy(historic)
    del hist_nomodel.model
    del hist_nomodel.validation_data
    pickle.dump(hist_nomodel, open(os.path.join(dir_path, 'hist.p'), "wb"))

    print('Saved trained model, scores and historic at %s ' % dir_path)

def saveModel(dir_path, model, scores, historic, saveH5=True):
    os.makedirs(dir_path)

    if saveH5:
        model.save(os.path.join(dir_path, 'model.h5'))

    pickle.dump(scores, open(os.path.join(dir_path, 'scores.p'), "wb"))

    hist_nomodel = copy.copy(historic)
    del hist_nomodel.model
    del hist_nomodel.validation_data
    pickle.dump(hist_nomodel, open(os.path.join(dir_path, 'hist.p'), "wb"))

    print('Saved trained model, scores and historic at %s ' % dir_path)

####### finish model_utils

### TODO: move this to a separate file

def train_single(model, opt, x_train, y_train, x_val, y_val, batch_size, epochs, exec_path):
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['acc'])
    hist = model.fit(x_train,y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_val,y_val),
                    shuffle=True, verbose=1)
    scores = model.evaluate(x_val, y_val, verbose=1)

    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    saveModel(exec_path, model, scores, hist, saveH5=False)

def train_all():

    alphas = [1e-4, 1e-5, 1e-6, 1e-7]
    lambdas = [1e-4, 1e-5, 1e-6, 1e-7]

    batch_size = 8
    num_classes = 6
    epochs = 100

    input_path = "processed_audio_datasets/"
    dir_path = "audio_results/" + "gitarchitecture/"
    # model = 0

    for alpha in alphas:
        for lamb in lambdas:

            prefix = 'AudioSentiment_'+'{:.0e}'.format(alpha)+'_decay_'+'{:.0e}'.format(lamb)
            exec_path = os.path.join(dir_path, prefix)

            opt = keras.optimizers.rmsprop(lr=alpha, decay=lamb)
            model = 0
            if not os.path.isdir(exec_path):
                print("####################################")
                print("Starting ravdess: " + prefix)
                (x_train, y_train), (x_val, y_val) = load_dataset(input_path + "ravdess")
                model = createModel(x_train[0].shape, num_classes)
                train_single(model, opt, x_train, y_train, x_val, y_val, batch_size, epochs, exec_path)

            exec_path = os.path.join(dir_path, "final_" + prefix)

            if not os.path.isdir(exec_path):
                print("####################################")
                print("Starting verbo")
                (x_train, y_train), (x_val, y_val) = load_dataset(input_path + "verbo")
                train_single(model, opt, x_train, y_train, x_val, y_val, batch_size, epochs//2, exec_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--regime', required=True, help="type of traning (ravdess,verbo,both)")
    # parser.add_argument('-o', '--output', required=True, help="path of the output directory")
    args = parser.parse_args()

    # for dataset, dirname in zip([ravdess, verbo],["ravdess","verbo"]):
    if args.regime == "both":
        model = train_all()
        # train(input_path + "verbo")
    else:
        pass


# def train_all(path, out_path):
#     (x_train, y_train), (x_val, y_val) = load_dataset(path)
#
#     alphas = [1e-4, 1e-5, 1e-6, 1e-7]
#     lambdas = [1e-4, 1e-5, 1e-6, 1e-7]
#
#     batch_size = 32
#     num_classes = 6
#     epochs = 200
#     dir_path = "audio_results/" + "gitarchitecture/"
#     # model = 0
#
#     for alpha in alphas:
#         for lamb in lambdas:
#
#             prefix += 'debug_AudioSentiment_'+'{:.0e}'.format(alpha)+'_decay_'+'{:.0e}'.format(lamb)
#             exec_path = os.path.join(dir_path, prefix)
#
#             if not os.path.isdir(exec_path):
#                 print("####################################")
#                 print("Starting: " + prefix)
#                 if model == None:
#                     model = createModel(x_train[0].shape, num_classes)
#                 else:
#                     epochs //= 2
#                 opt = keras.optimizers.rmsprop(lr=alpha, decay=lamb)
#
#                 model.compile(loss='categorical_crossentropy',
#                               optimizer=opt,
#                               metrics=['acc'])
#                 hist = model.fit(x_train,y_train,
#                                 batch_size=batch_size,
#                                 epochs=epochs,
#                                 validation_data=(x_val,y_val),
#                                 shuffle=True, verbose=1)
#                 scores = model.evaluate(x_val, y_val, verbose=1)
#
#                 print('Test loss:', scores[0])
#                 print('Test accuracy:', scores[1])
#
#                 saveModel(exec_path, model, scores, hist, saveH5=False)
#
#     return model
