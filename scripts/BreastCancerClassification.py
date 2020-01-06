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
matplotlib.use('Agg')
import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
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
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

os.getcwd()

"""Input data"""
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

"""Prepare data"""
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
train_reduced = os.path.sep.join([base_path, "training_reduced"])
val_reduced = os.path.sep.join([base_path, "validation_reduced"])
reduced_split = 0.01

random.shuffle(trainPaths)
index_train_reduced = int(len(trainPaths)*reduced_split)
trainReduced = trainPaths[:index_train_reduced]
random.shuffle(valPaths)
index_val_reduced = int(len(valPaths)*reduced_split)
valReduced = valPaths[:index_val_reduced]
datasets_reduced = [("training_reduced", trainReduced, train_reduced),
                    ("validation_reduced", valReduced, val_reduced)]

for (setType, trainPaths, basePath) in datasets_reduced:
        print(f'Building {setType} set')
        if not os.path.exists(basePath):
                print(f'Building directory {basePath}')
                os.makedirs(basePath)
        for path in trainReduced:
                # Pick name of the image file
                image = path.split(os.path.sep)[-1]
                # Pick the class of the image
                class_image = image[-5:-4]
                classPath = os.path.sep.join([train_reduced, class_image])
                if not os.path.exists(classPath):
                        print(f'Building directory {classPath}')
                        os.makedirs(classPath)
                newPath = os.path.sep.join([classPath, image])
                shutil.copy2(path, newPath)

trainReduced = list(paths.list_images(train_reduced))
valReduced = list(paths.list_images(val_reduced))

"""Models"""        
# Constants        
num_epochs = 40
init_lr = 0.01
batch_size = 32

# Create a function to define the model - Keras
def model(width,height,depth,classes):
    if K.image_data_format() != 'channels_first':
        shape = (height,width,depth)
        channelDim = -1
    else:
        shape = (depth,height,width)
        channelDim = 1
    
    model = Sequential()
    model.add(SeparableConv2D(32, (3,3), 
                              padding = "same", 
                              input_shape = shape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis = channelDim))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.25))
    
    model.add(SeparableConv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channelDim))
    model.add(SeparableConv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channelDim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(SeparableConv2D(128, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channelDim))
    model.add(SeparableConv2D(128, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channelDim))
    model.add(SeparableConv2D(128, (3,3), padding="same"))
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

# Create functions to calculate metrics
def sensitivity(y_true, y_pred):
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
  return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
  true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
  possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
  return true_negatives / (possible_negatives + K.epsilon())

def fmed(y_true, y_pred):
  spec = specificity(y_true, y_pred)
  sens = sensitivity(y_true, y_pred)
  fmed = 2 * (spec * sens)/(spec+sens+K.epsilon())
  return fmed

def f1(y_true, y_pred):
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
  predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
  precision = true_positives / (predicted_positives + K.epsilon())
  recall = true_positives / (possible_positives + K.epsilon())
  f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
  return f1_val

# Create and compile model
model = model(width = 48, height = 48, depth = 3, classes = len(classTotals))
opt = Adagrad(lr = init_lr, decay = (init_lr/num_epochs))
model.compile(loss = "binary_crossentropy", 
              optimizer = opt,
              metrics = ["accuracy", 
                         sensitivity,
                         specificity,
                         fmed,
                         f1])

# Augmentation
trainAug = ImageDataGenerator(
            rescale = 1/255.0,
            rotation_range = 20,
            zoom_range = 0.05,
            width_shift_range = 0.1,
            height_shift_range = 0.1,
            shear_range = 0.05,
            horizontal_flip = True,
            vertical_flip = True,
            fill_mode = "nearest")

generator = ImageDataGenerator(rescale=1/255.0)

trainGen = trainAug.flow_from_directory(train_path,
                                        class_mode = "categorical",
                                        target_size = (48,48),
                                        color_mode = "rgb",
                                        shuffle = True,
                                        batch_size = batch_size)

valGen = generator.flow_from_directory(val_path,
                                        class_mode = "categorical",
                                        target_size = (48,48),
                                        color_mode = "rgb",
                                        shuffle = False,
                                        batch_size = batch_size)

testGen = generator.flow_from_directory(test_path,
                                        class_mode = "categorical",
                                        target_size = (48,48),
                                        color_mode = "rgb",
                                        shuffle = False,
                                        batch_size = batch_size)

# Create callback
learn_control = ReduceLROnPlateau(monitor = "val_sensitivity", 
                                  patience = 3,
                                  verbose = 1,
                                  factor = 0.2, 
                                  min_lr = 1e-5)

model_path = "Documentos/Ubiqum/Final Project/BreastCancerClassification/model/model.hdf5"
checkpoint = ModelCheckpoint(model_path, 
                             monitor = "val_sensitivity", 
                             verbose = 1, 
                             save_best_only = True, 
                             mode = "max")

early_stopping = EarlyStopping(monitor = "val_sensitivity", 
                               mode = "max", 
                               verbose = 1, 
                               patience = 5)

# Fit the model
model_history = model.fit_generator(
        trainGen,
        steps_per_epoch = (lenTrain/batch_size),
        validation_data = valGen,
        validation_steps = (lenVal/batch_size),
        class_weight = classWeight,
        epochs = num_epochs,
        callbacks=[early_stopping, checkpoint])#, learn_control])


predictions = model.predict_generator(testGen, 
                                      steps = (lenTest/batch_size))
# Get most likely class
predicted_classes = np.argmax(predictions, axis=1)
predicted_classes.sum()

true_classes = testGen.classes
class_labels = list(testGen.class_indices.keys())

report = classification_report(true_classes, 
                               predicted_classes, 
                               target_names = class_labels)
print(report)

matrix = confusion_matrix(true_classes,
                          predicted_classes,
                          labels = [0, 1])
print(matrix)

# Plot history
history = pd.DataFrame(model_history.history)
history[['loss', 'val_loss']].plot()
plt.savefig('Documentos/Ubiqum/Final Project/BreastCancerClassification/plots/loss.png')

history[['accuracy', 'val_accuracy']].plot()
plt.savefig('Documentos/Ubiqum/Final Project/BreastCancerClassification/plots/acc.png')

# Tensorflow (without Keras)
leaky_relu_alpha = 0.2
dropout_rate = 0.5

def conv2d( inputs , filters , stride_size ):
    out = tf.nn.conv2d( inputs , 
                       filters , 
                       strides=[ 1 , stride_size , stride_size , 1 ] , 
                       padding='SAME') 
    return tf.nn.leaky_relu( out , alpha=leaky_relu_alpha ) 

def maxpool( inputs , pool_size , stride_size ):
    return tf.nn.max_pool2d( inputs , 
                            ksize=[ 1 , pool_size , pool_size , 1 ] , 
                            padding='VALID' , 
                            strides=[ 1 , stride_size , stride_size , 1 ] )

def dense( inputs , weights ):
    x = tf.nn.leaky_relu( tf.matmul( inputs , weights ) , 
                         alpha=leaky_relu_alpha )
    return tf.nn.dropout( x , rate=dropout_rate )


initializer = tf.initializers.glorot_uniform()
def get_weight( shape , name ):
    return tf.Variable( initializer( shape ) , 
                       name=name , 
                       trainable=True , 
                       dtype=tf.float32 )

shapes = [
    [ 50 , 50 , 3 , 3 ] , 
    [ 50 , 50 , 3 , 3 ] , 
    [ 1875 , 32 ] , 
    [ 32 , 2 ] 
]

weights = []
for i in range( len( shapes ) ):
    weights.append( get_weight( shapes[ i ] , 'weight{}'.format( i ) ) )
    
def model_tf( x ) :
    x = tf.cast( x , dtype=tf.float32 )
    c1 = conv2d( x , weights[ 0 ] , stride_size=1 ) 
    c1 = conv2d( c1 , weights[ 1 ] , stride_size=1 ) 
    p1 = maxpool( c1 , pool_size=2 , stride_size=2 )
    
    flatten = tf.reshape( p1 , shape=( tf.shape( p1 )[0] , -1 ))

    d1 = dense( flatten , weights[ 2 ] )
    logits = tf.matmul( d1 , weights[ 3 ] )

    return tf.nn.softmax( logits )

def loss_tf( pred , target ):
    return tf.losses.categorical_crossentropy( target , pred )

optimizer = tf.optimizers.Adam(init_lr)

def train_step( model, inputs , outputs ):
    with tf.GradientTape() as tape:
        current_loss = loss_tf(model(inputs), outputs)
    grads = tape.gradient( current_loss , weights )
    optimizer.apply_gradients( zip( grads , weights ) )
    print( tf.reduce_mean( current_loss ) )
    
for features in trainGen:
    image , label = features[0] , features[1]
    train_step( model_tf , image , label)

"""Hyperparameters"""
# Model to tune - No parameters
def model_tune(learn_rate):
    model = Sequential()
    model.add(SeparableConv2D(32, (3,3), 
                              padding = "same", 
                              input_shape = (48, 48, 3)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis = -1))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.25))
        
    model.add(SeparableConv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis= -1))
    model.add(SeparableConv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis= -1))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(SeparableConv2D(128, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis= -1))
    model.add(SeparableConv2D(128, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis= -1))
    model.add(SeparableConv2D(128, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis= -1))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(2))
    model.add(Activation("softmax"))

    opt = Adagrad(lr = learn_rate)

    model.compile(loss = "binary_crossentropy",     
                #  optimizer = "adam",
                  optimizer = opt,
                  metrics = ["accuracy"])

    return model

# Grid Search to tune hyperparameters
model_grid = KerasClassifier(build_fn = model_tune, batch_size=32, verbose = 0)
epochs = [20, 30, 40, 50, 60]
batch = [16, 32, 48, 60, 72]
learn_rate = [0.0001, 0.001]#, 0.01, 0.1]
param_grid = dict(learn_rate = learn_rate) #batch_size = batch, epochs = epochs) 
grid = GridSearchCV(estimator = model_grid, 
                    param_grid = param_grid, 
                    n_jobs = 1, 
                    cv = 3)

trainGenNoBatch = generator.flow_from_directory(train_reduced,
                                        class_mode = "categorical",
                                        target_size = (48,48),
                                        color_mode = "rgb",
                                        shuffle = True,
                                        batch_size = 32)

#------------------------------------------------------------------------------
len(trainGenNoBatch[0])

trainGenNoBatch.dtype
trainGenNoBatch(1)
trainGenNoBatch # 1776 - Number of batchs
trainGenNoBatch[0] # 2 - Image / Label
trainGenNoBatch[0][0] # 1 - Batch size
trainGenNoBatch[0][0][0] # 50 - Pixels
trainGenNoBatch[0][0][0][0] # 50 - Pixels
trainGenNoBatch[0][0][0][0][0] # 3  - Channels

trainGenNoBatch[500]

t = trainGenNoBatch._get_batches_of_transformed_samples
#------------------------------------------------------------------------------

images = []
labels = []
c = 0
for i, o in trainGenNoBatch:
    if c < 3:#1776:
         images.append(i[0])
         labels.append(o[0])
         c = c + 1
    else:
        break
    
len(images[0])

images = np.array(images)

grid_result = grid.fit(images, labels)
# 18:11 - 18:12 / 13 - 
# 13:29 - 
# 15:47 - 15:59 - 3 images and 2 param

# Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))