import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Flatten, Input, GlobalAveragePooling2D

import os
import copy
import pickle
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.nasnet import NASNetMobile
from keras.applications.resnet_v2 import ResNet152V2


def createModel(name, imshape, num_classes):
    input_tensor = Input(shape=imshape)

    if name == "xception":
        base_model = Xception(input_tensor=input_tensor, include_top=False)
    elif name == "vgg19":
        base_model = VGG19(input_tensor=input_tensor, include_top=False)
    elif name == "mobilenet":
        base_model = MobileNetV2(input_tensor=input_tensor, include_top=False)
    elif name == "resnet152":
        base_model = ResNet152V2(input_tensor=input_tensor, include_top=False)

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
