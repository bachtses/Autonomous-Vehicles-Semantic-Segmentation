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


'''
_______________  CLASSES / LABELS  _____________
Unlabeled     0,0,0         #black
Road          128,64,128    #purple
Sidewalk      244,35,232    #magenta
Building      70,70,70      #gray
Wall          102,102,156   #lila
Fence         190,153,153   #pale pink
Pole          153,153,153   #dark gray
TrafficLight
TrafficSign
Vegetation    107,142,35    #green
Terrain       152,251,152   #pale green
Sky           70,130,180    #light blue
Person        220,20,60     #light red
Rider         255,0,0       #red
Car           0,0,142       #blue electrique
Truck
Bus
Train
Motorcycle
Bicycle       119,11,32     #burgundy
'''


batch = 10
epochs = 10

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
n_classes = 14

IMAGES_PATH = 'images/'
MASKS_PATH = 'masks/'

seed = 42
random.seed = seed
np.random.seed = seed

path, dirs, files = next(os.walk(IMAGES_PATH))
images_number = len(files)
print("\n")
print("FOUND: ", images_number, "images")
print("\n")

X = np.zeros((images_number, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y = np.zeros((images_number, IMG_HEIGHT, IMG_WIDTH, n_classes), dtype=np.bool)


# _______________  IMAGES READ  _____________
k = 0
for item in os.listdir(IMAGES_PATH):
    img = cv2.imread(os.path.join(IMAGES_PATH, item))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    # plt.imshow(img)
    # plt.show()
    X[k] = img
    k = k + 1

# _______________  MASKS READ  _____________
k = 0
mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
for item in os.listdir(MASKS_PATH):
    mask = cv2.imread(os.path.join(MASKS_PATH, item))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask = cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH))
    # print(mask)
    # plt.imshow(mask)
    # plt.show()
    # LABELING THE CLASSES
    mask_targets = np.zeros((IMG_HEIGHT, IMG_WIDTH, n_classes), dtype=np.bool)
    for i in range(IMG_HEIGHT):
        for j in range(IMG_WIDTH):
            if np.array_equal(mask[i][j], np.array([128, 64, 128])):
                mask_targets[i][j] = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            elif np.array_equal(mask[i][j], np.array([107, 142, 35])):
                mask_targets[i][j] = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            elif np.array_equal(mask[i][j], np.array([0, 0, 142])):
                mask_targets[i][j] = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            elif np.array_equal(mask[i][j], np.array([244, 35, 232])):
                mask_targets[i][j] = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            elif np.array_equal(mask[i][j], np.array([153, 153, 153])):
                mask_targets[i][j] = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            elif np.array_equal(mask[i][j], np.array([70, 130, 180])):
                mask_targets[i][j] = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
            elif np.array_equal(mask[i][j], np.array([70, 70, 70])):
                mask_targets[i][j] = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
            elif np.array_equal(mask[i][j], np.array([152, 251, 152])):
                mask_targets[i][j] = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
            elif np.array_equal(mask[i][j], np.array([220, 20, 60])):
                mask_targets[i][j] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
            elif np.array_equal(mask[i][j], np.array([255, 0, 0])):
                mask_targets[i][j] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
            elif np.array_equal(mask[i][j], np.array([119, 11, 32])):
                mask_targets[i][j] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
            elif np.array_equal(mask[i][j], np.array([102, 102, 156])):
                mask_targets[i][j] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
            elif np.array_equal(mask[i][j], np.array([190, 153, 153])):
                mask_targets[i][j] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
            # UNLABELED
            else:
                mask_targets[i][j] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    Y[k] = mask_targets
    k = k + 1


# images
x_train = X
print('x_train shape', np.shape(x_train))
# masks
y_train = Y
print('y_train shape', np.shape(y_train))


# _________________  BUILD U-Net MODEL  __________________
inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

c1 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

c3 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

c4 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

c5 = tf.keras.layers.Conv2D(256, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(c5)

u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(c8)

u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal', padding='same')(c9)

outputs = tf.keras.layers.Conv2D(n_classes, (1, 1), activation='softmax')(c9)

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


history = model.fit(x_train, y_train, validation_split=0.1, batch_size=batch, epochs=epochs)

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# _______________  SAVE  _____________
# model_json = model.to_json()
# with open("model_in_json.json", "w") as json_file:
#    json_file.write(model_json)
# serialize weights to HDF5
# model.save_weights("model_weights.h5")

model.save('my_model.h5')
