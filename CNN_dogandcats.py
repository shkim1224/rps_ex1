# -*- coding: utf-8 -*-
"""
Created on Thu May 28 13:07:18 2020

@author: owner
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

DATADIR = "E:/Keras_CNN/Petimages"
CATEGORIES = ["Dog", "Cat"]   # [ ] -> list
IMG_SIZE = 50

training_data = [] # empty list
def create_training_data():
    for category in CATEGORIES:
        class_num = CATEGORIES.index(category)
        path = os.path.join(DATADIR, category) #path to cats or dogs dir
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)  # E:/Keras_CNN/Petimages/Dog/*.jpg 를 흑백으로 변환
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()
random.shuffle(training_data)

X=[]    # 입력데이터 리스트 
y=[]    # 출력 list

for features, label in training_data:
    X.append(features)
    y.append(label)
    
X=np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()
"""
with open("X.pickle", "wb") as f:
    pickle.dump(X,f)
"""    
pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()
"""
with open("y.pickle","wb") as f:
    pickle.dump(y, f)
pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)
"""
with open("X.pickle","rb") as f:
    X = pickle.load(f)

with open("y.pickle", "rb") as f:
    y = pickle.load(f)




        
