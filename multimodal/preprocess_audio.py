

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram

import os
import pandas as pd
import glob
import scipy.io.wavfile
import sys
import shutil

import sklearn
import warnings
warnings.filterwarnings('ignore') # sklearn warns about bad std on verbo dataset

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

def pad_audio_wave(a, size):
    l = a.shape[0]
    pad_l = size - l
    p1 = np.zeros(pad_l//2)
    p2 = np.zeros(size - l - pad_l//2)
    return np.hstack((p1,a,p2))

def gen_mfcc_features(path, duration=6):
    sr_ratio = 22050*2
    X, sample_rate = librosa.load(path, res_type='kaiser_fast',
                        duration=duration, sr=sr_ratio, offset=0)
    X = pad_audio_wave(X, duration * sr_ratio)
    sample_rate = np.array(sample_rate)
    features = librosa.feature.mfcc(y=X,
                                        sr=sample_rate,
                                        n_mfcc=13)
    features = sklearn.preprocessing.scale(features, axis=1)
    # return librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)/-100
    return np.mean(features,axis=0)

def generate_mfccs_ravdess():

    features, labels = np.array([]), []
    bookmark=0

    datasetPath = 'ravdess/'
    classes = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    print("start of ravdess mfcc feature extraction")
    for class_index, dir in enumerate(classes):
        base_path = datasetPath + dir + "/"
        file_paths = [d for d in os.listdir(base_path) if d[:2] != "._"]
        for index, r_path in enumerate(file_paths):

            feature = gen_mfcc_features(base_path + r_path)
            if bookmark == 0:
                features = feature
            else:
                features = np.vstack((features, feature))
            labels.append(class_index)

            bookmark += 1
            if bookmark % 200 == 0:
                print(bookmark)
    print(str(bookmark) + "\nend of ravdess mfcc feature extraction")

    return {'features':features,
            "classes": np.array(labels, dtype=np.int8)}

def generate_mfccs_verbo():
    features, labels = np.array([]), []
    bookmark=0

    datasetPath = 'verbo/'
    classes_prefix = ['rai', 'med', 'ale', 'neu', 'tri', 'sur']

    print("start of verbo mfcc feature extraction")
    dirs = [d for d in os.listdir(datasetPath) if os.path.isdir(datasetPath + d)]

    for dir_index, dir in enumerate(dirs):
        base_path = datasetPath + dir + "/"
        file_paths = [d for d in os.listdir(base_path) if d[:2] != "._" and d[:3] != 'des']

        for index, r_path in enumerate(file_paths):
            prefix = r_path[:3]
            class_index = classes_prefix.index(prefix)

            feature = gen_mfcc_features(base_path + r_path)
            if bookmark == 0:
                features = feature
            else:
                features = np.vstack((features, feature))
            labels.append(class_index)

            bookmark += 1
            if bookmark % 200 == 0:
                print(bookmark)
    print("total: " + str(bookmark) + "\nend of verbo mfcc feature extraction")

    return {'features':features,
            "classes": np.array(labels, dtype=np.int8)}

def split_dataset(d):
    # join features and classes
    data = np.hstack((d["features"],np.reshape(d["classes"], (d["classes"].shape[0],1))))
    np.random.shuffle(data)

    split = np.random.rand(len(data)) < 0.75
    train = data[split]

    val_test = data[~split]
    split = np.random.rand(len(val_test)) < 0.5
    val = val_test[split]
    test = val_test[~split]

    return train, val, test


def write(dir, tr,vl,tt):
    shutil.rmtree(dir)
    os.makedirs(dir)
    np.savetxt(dir + "/train.csv", tr, delimiter=",")
    np.savetxt(dir + "/val.csv", vl, delimiter=",")
    np.savetxt(dir + "/test.csv", tt, delimiter=",")
    print("saved: " + dir)

if __name__ == "__main__":
    ravdess = generate_mfccs_ravdess()
    verbo = generate_mfccs_verbo()

    for dataset, dirname in zip([ravdess, verbo],["ravdess","verbo"]):
        splits = split_dataset(dataset)
        write("processed_audio_datasets/" + dirname, splits[0], splits[1], splits[2])

# path = 'ravdess/Angry/' + '01-01-05-01-01-01-01.wav'
# f = gen_mfcc_features(path)
# f = sklearn.preprocessing.scale(f, axis=1)
# plt.figure(figsize=(15, 5))
# librosa.display.specshow(f, sr=22050*2, x_axis='time')
# plt.savefig('foo.png')
