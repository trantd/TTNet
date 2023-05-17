from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.python.lib.io import file_io

import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

# from keras.utils import plot_model
from sklearn.metrics import *
# from keras.engine import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Activation, Conv2D, MaxPool2D, BatchNormalization, Dropout, \
    MaxPooling2D, ReLU, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import skimage
from skimage.transform import rescale, resize

import pydot
import multiprocessing

from tensorflow.keras.optimizers import SGD
from keras_vggface.vggface import VGGFace


def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1 / 0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()
        if p_1 > p:
            return input_img
        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)
            if left + w <= img_w and top + h <= img_h:
                break
        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)
        input_img[top:top + h, left:left + w, :] = c
        return input_img
    return eraser


def get_datagen(dataset, aug=False):
    if aug:
        datagen = ImageDataGenerator(
            rescale=1. / 255,
            featurewise_center=False,
            featurewise_std_normalization=False,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            preprocessing_function=get_random_eraser(v_l=0, v_h=1, pixel_level=True), )
    else:
        datagen = ImageDataGenerator(rescale=1. / 255)
    return datagen.flow_from_directory(
        dataset,
        target_size=(197, 197),
        color_mode='rgb',
        shuffle=True,
        class_mode='categorical',
        batch_size=32)


train_generator = get_datagen('/Users/phungthetai/Documents/NCKH/data/fer-2013/train', True)
test_generator = get_datagen('/Users/phungthetai/Documents/NCKH/data/fer-2013/test')

lr = 0.01
batch_size = 32
epochs = 128

vgg_notop = VGGFace(model='resnet50', include_top=False, input_shape=(197, 197, 3), pooling='avg')
last_layer = vgg_notop.get_layer('avg_pool').output
x = Flatten(name='flatten')(last_layer)
x = Dropout(0.5)(x)
x = Dense(4096, activation='relu', name='fc6')(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu', name='fc7')(x)
x = Dropout(0.5)(x)

batch_norm_indices = [2, 6, 9, 13, 14, 18, 21, 24, 28, 31, 34, 38, 41, 45, 46, 53, 56, 60, 63, 66, 70, 73, 76, 80, 83,
                      87, 88, 92, 95, 98, 102, 105, 108, 112, 115, 118, 122, 125, 128, 132, 135, 138, 142, 145, 149,
                      150, 154, 157, 160, 164, 167, 170]
for i in range(170):
    if i not in batch_norm_indices:
        vgg_notop.layers[i].trainable = False

out = Dense(7, activation='softmax', name='classifier')(x)

model = tf.keras.Model(vgg_notop.input, out)

sgd = SGD(learning_rate=lr, momentum=0.9, decay=0.0001, nesterov=True)

model.compile(optimizer=sgd, loss=['categorical_crossentropy'], metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20)
rlrop =tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy",
                                          patience=10,
                                           verbose=1,
                                           factor=0.5,
                                          min_lr=0.0001)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='./model',
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)


history = model.fit_generator(
    generator=train_generator,
    validation_data=test_generator,
    steps_per_epoch=28709 // batch_size,
    validation_steps=3509 // batch_size,
    shuffle=True,
    epochs=epochs,
    callbacks=[rlrop, early_stopping, model_checkpoint_callback],
)

model.load_weights('./model')
model.save('model.h5')


plt.figure(figsize=(20, 6))
plt.subplot(121)
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Training and validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

plt.subplot(122)
plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Training and validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
