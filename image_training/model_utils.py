import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Flatten, Input, GlobalAveragePooling2D

import os
import copy
import pickle
import numpy as np
from keras.applications.inception_v3 import InceptionV3


def createModel(name, imshape, num_classes):
    input_tensor = Input(shape=imshape)

    if name == "InceptionV3":
        base_model = InceptionV3(input_tensor=input_tensor, include_top=False)
    elif name == "VGG16":
        pass

    # add a global spatial average pooling layer
    x = base_model.output
    image_out = GlobalAveragePooling2D()(x)
    concat = image_out
    inputs = [base_model.input]

    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(concat)
    # and a logistic layer -- let's say we have num_classes classes
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=predictions)

    return model


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
