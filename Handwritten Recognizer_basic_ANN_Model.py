#!/usr/bin/env python
# coding: utf-8

#installing the required package

pip install opencv-python
pip install tensorflow

#import package
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os


#loading dataset
mnist=tf.keras.datasets.mnist


#splitting
(x_train,y_train),(x_test,y_test)=mnist.load_data()

#normalization
x_train=tf.keras.utils.normalize(x_train,axis=1)
y_test=tf.keras.utils.normalize(x_test,axis=1)


#creating basic model
model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(200,activation='relu'))
model.add(tf.keras.layers.Dense(200,activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))


#compiling the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#fitting the model
model.fit(x_train,y_train,epochs=3)

#saving the model
model.save('handwritten.model')

#using the saving model

model1=tf.keras.models.load_model('handwritten.model')


#evaluting the model
model1.evaluate(x_test,y_test)


#predicting the images with our saved model

image_no=1

while os.path.isfile(f"C:/Users/91936/Documents/Digit_recognizer/testingimg/digits{image_no}.png"):
    try:
        img=cv2.imread(f"C:/Users/91936/Documents/Digit_recognizer/testingimg/digits{image_no}.png")[:,:,0]
        img=np.invert(np.array([img]))
        pred=model.predict(img)
        print(f"The digit is {np.argmax(pred)}")
        plt.imshow(img[0],cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error")
    finally:
        image_no+=1




