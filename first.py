import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from PIL import Image
from skimage import transform
import cv2
import numpy as np

local_zip = 'E:/PycharmProjects/rps_ex1/tmp/rps.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('E:/PycharmProjects/rps_ex1/tmp/')
zip_ref.close()

local_zip = 'E:/PycharmProjects/rps_ex1/tmp/rps-test-set.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('E:/PycharmProjects/rps_ex1/tmp/')
zip_ref.close()

TRAINING_DIR = "E:/PycharmProjects/rps_ex1/tmp/rps"
VALIDATION_DIR = "E:/PycharmProjects/rps_ex1/tmp/rps-test-set"

training_datagen = ImageDataGenerator(rescale=1./255)

training_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(150,150),
    class_mode='categorical'
    )

validation_generator = training_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(150,150),
    class_mode='categorical')

class_map = {
    0: 'paper',
    1: 'rock',
    2: 'scissor'}

for x, y in training_generator:
    print(x.shape, y.shape)
    print(y[0])

    fig, axes = plt.subplots(2, 5)
    fig.set_size_inches(15, 6)
    for i in range(10):
        axes[i//5, i%5].imshow(x[i])
        axes[i//5, i%5].set_title(class_map[np.argmax(y[i])], fontsize=15)
    plt.show()
    break

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64,(3,3),activation='relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
# Flatten the result to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
# 512 neuron hidden layer
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
    ])

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit_generator(training_generator, epochs=1,
                              validation_data = validation_generator,
                              verbose= 1)
plt.figure(figsize=(9, 6))
epochs=1
plt.plot(np.arange(1, epochs+1), history.history['loss'])
plt.plot(np.arange(1, epochs+1), history.history['accuracy'])
plt.title('Loss / accuracy', fontsize=20)
plt.xlabel('Epochs')
plt.ylabel('Loss/accuracy')
plt.legend(['loss', 'accuracy'], fontsize=15)
plt.show()


for x, y in validation_generator:
    print(x.shape, y.shape)
    print(y[0])
    classes = model.predict(x)
    plt.imshow(x[0])
    plt.show()
    print(classes[0])
    break