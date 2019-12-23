# coding: utf-8
'''
This script creates 3-channel gray images from FER 2013 dataset.
It has been done so that the CNNs designed for RGB images can
be used without modifying the input shape.
This script requires two command line parameters:
1. The path to the CSV file
2. The output directory
It generates the images and saves them in three directories inside
the output directory - Training, PublicTest, and PrivateTest.
These are the three original splits in the dataset.
3. delete disgusting (Change for Roberto project)
'''
import os
import csv
import argparse
import numpy as np
import scipy.misc
import skimage

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', required=True, help="path of the csv file")
parser.add_argument('-o', '--output', required=True, help="path of the output directory")
args = parser.parse_args()
w, h = 48, 48
image = np.zeros((h, w), dtype=np.uint8)
id = 1
labels_csv = "id,emotion,usage\n"
#0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
# delete disgust after generating
emotions_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
with open(args.file, 'rb') as csvfile:
	datareader = csv.reader(csvfile, delimiter =',')
	headers = datareader.next()
	print(headers)

	for row in datareader:
		emotion = row[0]
		pixels = map(int, row[1].split())
		usage = row[2]
		#print emotion, type(pixels[0]), usage
		pixels_array = np.asarray(pixels)

		image = pixels_array.reshape(w, h)
		#print image.shape

		stacked_image = np.dstack((image,) * 3)
		#print stacked_image.shape

		image_folder = os.path.join(args.output, usage, emotions_labels[int(emotion)])
		if not os.path.exists(image_folder):
			os.makedirs(image_folder)
		image_path =  os.path.join(image_folder , str(id) + '.jpg')
		scipy.misc.imsave(image_path, skimage.transform.rescale(stacked_image,2))
		labels_csv += str(id) + "," + str(emotion) + "," + usage + "\n"
		id += 1
		if id % 100 == 0:
			print('Processed {} images'.format(id))

print("Finished processing {} images".format(id))
csv_labels_file = open("image_labels.csv","w")
csv_labels_file.write(labels_csv)
csv_labels_file.close()
