from keras import *
import keras
import keras.preprocessing.image as im
import cv2 as cv
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import model_utils as utils
import argparse
from keras.applications.imagenet_utils import preprocess_input

### Plot charts or images, wrapper for matplotlib axes and figs
def get_ax(rows=1, cols=1,figsize=(4,4), imgmode=False, returnfig=False):
    fig, axes = plt.subplots(figsize=figsize, dpi = 100, nrows=rows, ncols=cols)
    if imgmode:
        if rows == 1 and cols == 1:
            axes.clear()
            axes.get_xaxis().set_visible(False)
            axes.get_yaxis().set_visible(False)
        else:
            for ax in axes:
                if (isinstance(ax,np.ndarray)):
                    for a in ax:
                        a.clear()
                        a.get_xaxis().set_visible(False)
                        a.get_yaxis().set_visible(False)
                else:
                    ax.clear()
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
    return (fig, axes) if returnfig else axes

def datafromDir(imshape, batch_size=32):
    train_datagen = im.ImageDataGenerator(
        preprocessing_function=preprocess_input,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    val_datagen = im.ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
            'dataset/Training',
            target_size=(imshape[0],imshape[1]), # 4h:15 pra chegar nessa solucao
            batch_size=batch_size,
            class_mode="categorical")
    validation_generator = val_datagen.flow_from_directory(
        'dataset/PublicTest',
        target_size=(imshape[0],imshape[1]),
        class_mode="categorical")

    return {"training": train_generator, "validation": validation_generator}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-f', '--file', required=True, help="path of the csv file")
    # parser.add_argument('-o', '--output', required=True, help="path of the output directory")
    args = parser.parse_args()

    alphas = [1e-4, 1e-5, 1e-6, 1e-7]
    lambdas = [1e-4, 1e-5, 1e-6]
    batch_size = 32
    num_classes = 6
    epochs = 2
    dir_path = "training_results"
    imshape = (48*2, 48*2, 3)

    for alpha in alphas:
        for lamb in lambdas:

            prefix = 'FaceSentiment_'+'{:.0e}'.format(alpha)+'_decay_'+'{:.0e}'.format(lamb)
            exec_path = os.path.join(dir_path, prefix)

            if not os.path.isdir(exec_path):
                opt = keras.optimizers.rmsprop(lr=alpha, decay=lamb)

                print("####################################")
                print("Starting: " + prefix)

                data = datafromDir(imshape, 2)
                # class indices
                # {'angry': 0, 'fear': 1, 'happy': 2, 'neutral': 3, 'sad': 4, 'surprise': 5}
                model = utils.createModel("InceptionV3", imshape, num_classes)

                model.compile(loss='categorical_crossentropy',
                              optimizer=opt,
                              metrics=['acc'])
                hist = model.fit_generator(data["training"],
                                steps_per_epoch=len(data["training"]), # generator was crated with batch of 32 images
                                epochs=epochs,
                                validation_data=data["validation"],
                                shuffle=True, verbose=2)
                scores = model.evaluate_generator(data["validation"], verbose=1)

                print('Test loss:', scores[0])
                print('Test accuracy:', scores[1])

                utils.saveModel(exec_path, model, scores, hist,saveH5=False)
            else:
                print(prefix + ' exists')
