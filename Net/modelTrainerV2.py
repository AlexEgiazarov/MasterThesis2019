from keras.preprocessing.image import ImageDataGenerator, load_img, array_to_img, img_to_array #Imports data generator and image functions
from PIL import Image #image processing
import numpy as np #numpy

#Keras model imports

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Input, Reshape, Dropout, Dense, Flatten, BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam


#Datagenerator
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    #rescale=1./255, #rescale test disabled
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')


inputShape = (200, 200)
model = Sequential()

model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=(200, 200, 3), padding="same"))
#Leaky relu is similar to usual relu. If x < 0 then f(x) = x * alpha, otherwise f(x) = x.
model.add(LeakyReLU(alpha=0.2))

#Dropout blocks some connections randomly. This help the model to generalize better.
#0.25 means that every connection has a 25% chance of being blocked.
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))

#Zero padding adds additional rows and columns to the image. Those rows and columns are made of zeros.
model.add(ZeroPadding2D(padding=((0,1),(0,1))))
model.add(BatchNormalization(momentum=0.8))
model.add(LeakyReLU(alpha=0.2))

model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
model.add(BatchNormalization(momentum=0.8))
model.add(LeakyReLU(alpha=0.2))

model.add(Dropout(0.25))
model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
model.add(BatchNormalization(momentum=0.8))
model.add(LeakyReLU(alpha=0.2))

model.add(Dropout(0.25))
model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
model.add(BatchNormalization(momentum=0.8))
model.add(LeakyReLU(alpha=0.2))

model.add(Dropout(0.25))
#Flatten layer flattens the output of the previous layer to a single dimension.
model.add(Flatten())
#Outputs a value between 0 and 1 that predicts whether image is real or generated. 0 = generated, 1 = real.
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

'''PREPARING THE DATA IN BATCHES'''

batch_size = 16

#augmentation config for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

#augmentation config for testing
test_datagen = ImageDataGenerator(rescale=1./255)

#generator that will take the images from train directory and augment them
train_generator = train_datagen.flow_from_directory(
    'data/stocks/train', #target directory
    target_size = (200, 200), #resizing images
    batch_size=batch_size,
    class_mode='binary') #binary labels

#Similar generator for validation data
validation_generator = test_datagen.flow_from_directory(
    'data/stocks/validation',
    target_size=(200,200),
    batch_size=batch_size,
    class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch= 16000 // batch_size,
    epochs=5,
    validation_data = validation_generator,
    validation_steps = 3000 // batch_size)

model.save('/home/theinfernal/MasterProject/Net/NetModel/models/saved-model-stocks-big-set-big-model.h5')

print('DONE')
