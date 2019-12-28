

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import keras
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
import os
import pandas as pd
import glob
import scipy.io.wavfile
import sys

# data, sampling_rate = librosa.load('output10.wav')
#
# plt.figure(figsize=(15, 5))
# librosa.display.waveplot(data, sr=sampling_rate)
# plt.savefig('foo.png')
# #plt.show()

# classes:
# {'angry': 0, 'fear': 1, 'happy': 2, 'neutral': 3, 'sad': 4, 'surprise': 5}
# calm == neutral
# remove disgust
# {'raiva': 0, 'medo': 1, 'alegria': 2, 'neutro': 3, 'tristeza': 4, 'surpresa': 5}
def generate_mccs_ravdess():

    df = pd.DataFrame(columns=['feature'])
    bookmark=0

    datasetPath = 'ravdess/'
    classes = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    for dir in classes:
        base_path = datasetPath + dir + "/"
        file_paths = [x for x in os.listdir(base_path) if x[:2] != "._"]
        print(file_paths)
        for index, r_path in enumerate(file_paths):

            X, sample_rate = librosa.load(base_path + r_path, res_type='kaiser_fast',
                                duration=10, sr=22050*2, offset=0)
            sample_rate = np.array(sample_rate)
            mfccs = np.mean(librosa.feature.mfcc(y=X,
                                                sr=sample_rate,
                                                n_mfcc=13),
                            axis=0)
            feature = mfccs

            df.loc[bookmark] = [feature]
            bookmark += 1
            print(feature.shape, X.shape)
            if bookmark % 100 == 0:
                print(bookmark)

def read_verbo_dataset():
    df = pd.read_csv("Dt_alegria.arff", header=None)
    print(df)

generate_mccs_ravdess()
#read_verbo_dataset()
