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

os.getcwd()

# Input data
input_dataset = "Documentos/Ubiqum/Final Project/BreastCancerClassification/data/IDC_regular_ps50_idx5"
base_path = "Documentos/Ubiqum/Final Project/BreastCancerClassification/data"
train_path = os.path.sep.join([base_path, "training"])
val_path = os.path.sep.join([base_path, "validation"])
test_path = os.path.sep.join([base_path, "testing"])
train_split = 0.8
val_split = 0.2

originalPaths=list(paths.list_images(input_dataset))
random.shuffle(originalPaths)
index_train=int(len(originalPaths)*train_split)
trainPaths=originalPaths[:index_train]
testPaths=originalPaths[index_train:]
index_val=int(len(trainPaths)*val_split)
valPaths=trainPaths[:index_val]
trainPaths=trainPaths[index_val:]
datasets=[("training", trainPaths, train_path),
          ("validation", valPaths, val_path),
          ("testing", testPaths, test_path)]

# Build data sets
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
                
