#!/usr/bin/env python
# %%
import matplotlib
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from sklearn.model_selection import train_test_split
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import glob

from keras import backend as KerasBackend

# %%
epochs = 100
lr = 1e-3
batch_size = 64
image_dimensions = (96, 96, 3)


data = []
labels = []
cwd = os.path.abspath(os.getcwd())
path_to_model_file = os.path.join(cwd, 'model.h5')
path_to_dataset_directory = os.path.join(cwd, 'dataset')

# %%
image_files_path = [f for f in glob.glob(
    path_to_dataset_directory + "/**/*", recursive=True) if not os.path.isdir(f)]
random.seed(42)
random.shuffle(image_files_path)

print("Total " + str(len(image_files_path)) + " images found")

# %%
for img_path in image_files_path:
    image = cv2.imread(img_path)
    image = cv2.resize(image, (image_dimensions[0], image_dimensions[1]))
    image = img_to_array(image)
    data.append(image)

    label = img_path.split(os.path.sep)[-2]
    if label == "men":
        label = 0
    else:
        label = 1
    labels.append([label])

if len(labels) != len(data):
    exit()

# %%
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.2, random_state=42
)
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

# %%
aug = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

input_shape = [
    image_dimensions[0],
    image_dimensions[1],
    image_dimensions[2]
]

model = Sequential([
    Conv2D(32, (3, 3), padding="same",
           input_shape=input_shape, activation='relu'),
    MaxPooling2D(pool_size=(3, 3)),
    Dropout(0.25),

    Conv2D(64, (3, 3), padding="same", activation='relu'),
    Conv2D(64, (3, 3), padding="same", activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), padding="same", activation='relu'),
    Conv2D(128, (3, 3), padding="same", activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),

    Dense(2),
    Activation("sigmoid"),
])

model.summary()

# %%
opt = Adam(lr=lr, decay=lr/epochs)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

model_history = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=batch_size),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // batch_size,
    epochs=epochs,
    verbose=1
)
# %%
model.save(path_to_model_file)

# %%
print(model_history.history.keys())
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs),
         model_history.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs),
         model_history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), model_history.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, epochs),
         model_history.history["val_accuracy"], label="val_accuracy")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")

# save plot to disk
# plt.savefig(args.plot)
