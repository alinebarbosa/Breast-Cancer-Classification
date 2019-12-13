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


#------------------------------------------------------------------------------
# Tensorflow

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
#    [ 3 , 3 , 16 , 32 ] , 
#    [ 3 , 3 , 32 , 32 ] ,
#    [ 3 , 3 , 32 , 64 ] , 
#    [ 3 , 3 , 64 , 64 ] ,
#    [ 3 , 3 , 64 , 128 ] , 
#    [ 3 , 3 , 128 , 128 ] ,
#    [ 3 , 3 , 128 , 256 ] , 
#    [ 3 , 3 , 256 , 256 ] ,
#    [ 3 , 3 , 256 , 512 ] , 
#    [ 3 , 3 , 512 , 512 ] ,
    [ 1875 , 32 ] , 
#    [ 3600 , 2400 ] ,
#    [ 2400 , 1600 ] , 
#    [ 1600 , 800 ] ,
#    [ 800 , 64 ] ,
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
    
#    c2 = conv2d( p1 , weights[ 2 ] , stride_size=1 )
#    c2 = conv2d( c2 , weights[ 3 ] , stride_size=1 ) 
#    p2 = maxpool( c2 , pool_size=2 , stride_size=2 )
    
#    c3 = conv2d( p2 , weights[ 4 ] , stride_size=1 ) 
#    c3 = conv2d( c3 , weights[ 5 ] , stride_size=1 ) 
#    p3 = maxpool( c3 , pool_size=2 , stride_size=2 )
    
#    c4 = conv2d( p3 , weights[ 6 ] , stride_size=1 )
#    c4 = conv2d( c4 , weights[ 7 ] , stride_size=1 )
#    p4 = maxpool( c4 , pool_size=2 , stride_size=2 )

#    c5 = conv2d( p4 , weights[ 8 ] , stride_size=1 )
#    c5 = conv2d( c5 , weights[ 9 ] , stride_size=1 )
#    p5 = maxpool( c5 , pool_size=2 , stride_size=2 )

#    c6 = conv2d( p5 , weights[ 10 ] , stride_size=1 )
#    c6 = conv2d( c6 , weights[ 11 ] , stride_size=1 )
#    p6 = maxpool( c6 , pool_size=2 , stride_size=2 )

    flatten = tf.reshape( p1 , shape=( tf.shape( p1 )[0] , -1 ))

    d1 = dense( flatten , weights[ 2 ] )
#    d2 = dense( d1 , weights[ 13 ] )
#    d3 = dense( d2 , weights[ 14 ] )
#    d4 = dense( d3 , weights[ 15 ] )
#    d5 = dense( d4 , weights[ 16 ] )
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
    
#for e in range( num_epochs ):
    for features in trainGen:
#        print(features)
        image , label = features[0] , features[1]
#        label = tf.convert_to_tensor(label, tf.float32)
#        print(label.dtype)
#        train_step( model_tf , image , tf.one_hot( label , depth=3 ) )
        train_step( model_tf , image , label)
    
