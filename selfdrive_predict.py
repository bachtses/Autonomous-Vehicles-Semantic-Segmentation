import tensorflow as tf
import os
import cv2
import sys
import random
from PIL import Image
from resizeimage import resizeimage
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import matplotlib
from itertools import chain
from skimage.io import imread, imshow
from keras.models import model_from_json
import json

tf.enable_eager_execution()


IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

# _______________  LOAD  _____________
model = tf.keras.models.load_model("my_model.h5")
# load weights into new model
# model.load_weights("model_weights.h5")


# _______________  DEMONSTRATE PREDICTION  _____________
test_PATH = 'test/'
for item in os.listdir(test_PATH):  # iterate over each image
    path = os.path.join(test_PATH, item)
    img = cv2.imread(path)  # convert to array
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img = np.expand_dims(img, axis=0)
    print(np.shape(img))

predict = model.predict(img, verbose=1)
prediction_matrix = np.squeeze(predict[0])


img_synthesis = np.zeros((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
most_common_colors_matrix = np.zeros(14)


# LABELING THE CLASSES
for i in range(IMG_HEIGHT):
    for j in range(IMG_WIDTH):
        if np.argmax(predict[0][i][j]) == 0:
            img_synthesis[i][j] = np.array([128, 64, 128])
            most_common_colors_matrix[0] += 1
        elif np.argmax(predict[0][i][j]) == 1:
            img_synthesis[i][j] = np.array([107, 142, 35])
            most_common_colors_matrix[1] += 1
        elif np.argmax(predict[0][i][j]) == 2:
            img_synthesis[i][j] = np.array([0, 0, 142])
            most_common_colors_matrix[2] += 1
        elif np.argmax(predict[0][i][j]) == 3:
            img_synthesis[i][j] = np.array([244, 35, 232])
            most_common_colors_matrix[3] += 1
        elif np.argmax(predict[0][i][j]) == 4:
            img_synthesis[i][j] = np.array([153, 153, 153])
            most_common_colors_matrix[4] += 1
        elif np.argmax(predict[0][i][j]) == 5:
            img_synthesis[i][j] = np.array([70, 130, 180])
            most_common_colors_matrix[5] += 1
        elif np.argmax(predict[0][i][j]) == 6:
            img_synthesis[i][j] = np.array([70, 70, 70])
            most_common_colors_matrix[6] += 1
        elif np.argmax(predict[0][i][j]) == 7:
            img_synthesis[i][j] = np.array([152, 251, 152])
            most_common_colors_matrix[7] += 1
        elif np.argmax(predict[0][i][j]) == 8:
            img_synthesis[i][j] = np.array([220, 20, 60])
            most_common_colors_matrix[8] += 1
        elif np.argmax(predict[0][i][j]) == 9:
            img_synthesis[i][j] = np.array([255, 0, 0])
            most_common_colors_matrix[9] += 1
        elif np.argmax(predict[0][i][j]) == 10:
            img_synthesis[i][j] = np.array([119, 11, 32])
            most_common_colors_matrix[10] += 1
        elif np.argmax(predict[0][i][j]) == 11:
            img_synthesis[i][j] = np.array([102, 102, 156])
            most_common_colors_matrix[11] += 1
        elif np.argmax(predict[0][i][j]) == 12:
            img_synthesis[i][j] = np.array([190, 153, 153])
            most_common_colors_matrix[12] += 1
        elif np.argmax(predict[0][i][j]) == 13:
            img_synthesis[i][j] = np.array([0, 0, 0])
            most_common_colors_matrix[13] += 1


print(most_common_colors_matrix)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.set_title('Real Image of car camera')
ax1.imshow(img[0])
ax2.set_title('Neural Network Prediction')
ax2.imshow(img_synthesis)
plt.show()

imshow(img_synthesis)
plt.savefig('NN_prediction.png')



