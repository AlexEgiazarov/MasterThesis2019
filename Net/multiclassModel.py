'''
THIS IS THE TRAINING MODULE FOR SEPARATE CREATING AND TRAINING OF CNN FOR CLASSIFICATION
WEIGHTS AND MODEL ITSELF IS SAVED IN THE SEPARATE FOLDER
'''

from keras.preprocessing.image import ImageDataGenerator, load_img, array_to_img, img_to_array #Imports data generator and image functions
from PIL import Image #image processing
import numpy as np #numpy

#Keras model imports

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

from keras.utils import plot_model

#Datagenerator
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.05,
    height_shift_range=0.05,
    #rescale=1./255, #rescale test disabled
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode='nearest')

'''MODEL'''

'''THIS MODEL IS YET MOST OPTIMAL FOR TRAINING: SUBJECT TO CHANGE'''

model = Sequential()

#COVNET LAYERS
model.add(Conv2D(32, (3, 3), input_shape=(200, 200, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#DENSE LAYERS
model.add(Flatten()) #Converts 3D feature maps to 1D vectors

model.add(Dense(64))
model.add(Activation('relu'))

#model.add(Dropout(0.5)) #Dropout level

model.add(Dense(32)) #Not original part
model.add(Activation('relu'))

model.add(Dropout(0.5)) #Dropout level

#model.add(Dense(1, activation='sigmoid'))
#model.add(Activation('sigmoid'))
model.add(Dense(2, activation = 'softmax'))

#model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#plot_model(model, to_file='model.png')
print(model.summary())

'''PREPARING THE DATA IN BATCHES'''

batch_size = 8

#augmentation config for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True)

#augmentation config for testing
test_datagen = ImageDataGenerator(rescale=1./255)

#generator that will take the images from train directory and augment them
train_generator = train_datagen.flow_from_directory(
    'binaryData/train', #target directory
    target_size = (200, 200), #resizing images
    batch_size=batch_size,
    #class_mode='binary') #binary labels
    class_mode='categorical')

#Similar generator for validation data
validation_generator = test_datagen.flow_from_directory(
    'binaryData/validation',
    target_size=(200,200),
    batch_size=batch_size,
    #class_mode='binary')
    class_mode='categorical')

#print(train_labels)

model.fit_generator(
    train_generator,
    steps_per_epoch= 16000 // batch_size,
    epochs=5,
    validation_data = validation_generator,
    validation_steps = 3000 // batch_size)

model.save('/home/theinfernal/MasterProject/Net/NetModel/models/model-stocks(5epochs-binary).h5')

'''
OLD WAY
model.save_weights('models/model.h5') #Saving weights!
model_json = model.to_json() #Saving model to json file
with open("/models/model.json", "w") as json_file:
    json_file.write(model_json)
'''

print('DONE')
