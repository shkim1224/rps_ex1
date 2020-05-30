# -*- coding: utf-8 -*-
"""
Created on Sat May 30 13:38:37 2020

@author: owner
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import matplotlib.pyplot as plt

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))


X = X/255.0

model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=X.shape[1:], activation='relu'))
model.add( MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss="binary_crossentropy",
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X,y,batch_size=32, validation_split=0.3)

pred = model.predict(X[:10])     # 10개의 입력에 대한 예측값
print(pred)                      # 10개의 예측값

plt.imshow(X[9].reshape(50,50))
plt.show()


# god=0, cat=1





              
              



