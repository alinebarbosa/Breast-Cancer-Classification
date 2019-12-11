#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 22:31:36 2019

Breast Cancer Classification
Deep Learning

@author: Aline Barbosa Alves
"""

# Import packages and set the seed
from imutils import paths
import random, shutil, os
random.seed(123)
import matplotlib
matplotlib.use("Agg")
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adagrad
from keras.utils import np_utils
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import SeparableConv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K

os.getcwd()

# Input data
input_dataset = "Documentos/Ubiqum/Final Project/BreastCancerClassification/data/IDC_regular_ps50_idx5"
base_path = "Documentos/Ubiqum/Final Project/BreastCancerClassification/data"
train_path = os.path.sep.join([base_path, "training"])
val_path = os.path.sep.join([base_path, "validation"])
test_path = os.path.sep.join([base_path, "testing"])
train_split = 0.8
val_split = 0.2

originalPaths = list(paths.list_images(input_dataset))
random.shuffle(originalPaths)
index_train = int(len(originalPaths)*train_split)
trainPaths = originalPaths[:index_train]
testPaths = originalPaths[index_train:]
index_val = int(len(trainPaths)*val_split)
valPaths = trainPaths[:index_val]
trainPaths = trainPaths[index_val:]
datasets = [("training", trainPaths, train_path), # 127309 / 50307 - 2.5306
            ("validation", valPaths, val_path), # 31707 / 12696 - 2.4974
            ("testing", testPaths, test_path)] # 39722 / 15783 - 2.5168

# Build data sets (Run once!)
for (setType, originalPaths, basePath) in datasets:
        print(f'Building {setType} set')
        if not os.path.exists(basePath):
                print(f'Building directory {basePath}')
                os.makedirs(basePath)
        for path in originalPaths:
                # Pick name of the image file
                image = path.split(os.path.sep)[-1]
                # Pick the class of the image
                class_image = image[-5:-4]
                classPath = os.path.sep.join([basePath, class_image])
                if not os.path.exists(classPath):
                        print(f'Building directory {classPath}')
                        os.makedirs(classPath)
                newPath = os.path.sep.join([classPath, image])
                shutil.copy2(path, newPath)
        
# Constants        
num_epochs = 40
init_lr = 0.01
batch_size = 32

# Create data with list of paths
trainPaths = list(paths.list_images(train_path))
lenTrain = len(trainPaths)
valPaths = list(paths.list_images(val_path))
lenVal = len(valPaths)
testPaths = list(paths.list_images(test_path))
lenTest = len(testPaths)

trainClass = [int(p.split(os.path.sep)[-2]) for p in trainPaths]
trainClass = np_utils.to_categorical(trainClass)
classTotals = trainClass.sum(axis=0)
classWeight = classTotals.max() / classTotals

# Create reduced data sets to create a model
reduced_split = 0.01
random.shuffle(trainPaths)
index_train_reduced = int(len(trainPaths)*reduced_split)
trainReduced = trainPaths[:index_train_reduced]
random.shuffle(valPaths)
index_val_reduced = int(len(valPaths)*reduced_split)
valReduced = valPaths[:index_val_reduced]

# Create a function to define the model
def model(width,height,depth,classes):
    if K.image_data_format() != 'channels_first':
        shape = (height,width,depth)
        channelDim = -1
    else:
        shape = (depth,height,width)
        channelDim = 1
    
    model = Sequential()
    model.add(SeparableConv2D(32, (3,3), padding="same",input_shape=shape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channelDim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(classes))
    model.add(Activation("softmax"))
    
    return model

# Create and compile model
model = model(width=50, height=50, depth=3, classes=len(classTotals))
opt = Adagrad(lr=init_lr, decay=init_lr/num_epochs)
model.compile(loss="binary_crossentropy",optimizer=opt,metrics=["accuracy"])


# -----------------------------------------------------------------------------
trainAug = ImageDataGenerator(
            rescale=1/255.0,
            rotation_range=20,
            zoom_range=0.05,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.05,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode="nearest")

generator = ImageDataGenerator(rescale=1/255.0)

trainGen = generator.flow_from_directory(train_path,
                                        class_mode="categorical",
                                        target_size=(50,50),
                                        color_mode="rgb",
                                        shuffle=True,
                                        batch_size=batch_size)

valGen = generator.flow_from_directory(val_path,
                                        class_mode="categorical",
                                        target_size=(50,50),
                                        color_mode="rgb",
                                        shuffle=True,
                                        batch_size=batch_size)

M = model.fit_generator(
        trainGen,
        steps_per_epoch=(lenTrain/batch_size)*reduced_split,
        validation_data=valGen,
        validation_steps=(lenVal/batch_size)*reduced_split,
        class_weight=classWeight,
        epochs=1)

predictions = model.predict_generator(valGen, 
                                      steps=lenVal/batch_size)
# Get most likely class
predicted_classes = np.argmax(predictions, axis=1)
predicted_classes.sum()

true_classes = valGen.classes
class_labels = list(valGen.class_indices.keys())

report = classification_report(true_classes, 
                                       predicted_classes, 
                                       target_names=class_labels)

matrix = confusion_matrix(true_classes,
                          predicted_classes,
                          labels=[0, 1])
