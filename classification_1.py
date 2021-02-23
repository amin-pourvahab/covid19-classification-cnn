# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 13:31:04 2021

@author: Poorvahab
"""

import numpy as np
import glob
import cv2
from sklearn.model_selection import train_test_split


path_normal='G:/proje/covid-19/COVID-19 Radiography Database/NORMAL/'
path_covid='G:/proje/covid-19/COVID-19 Radiography Database/COVID/'
Normal=glob.glob(path_normal+'*.png')
Covid=glob.glob(path_covid+'*.png')

images_Normal=[]
images_Covid=[]

labels_Normal=[]
labels_Covid=[]

for x in Normal:
    img=cv2.imread(x)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.resize(img,(100,100))
    img=img.astype('float32')
    img=img/np.max(img)
    images_Normal.append(img)
    labels_Normal.append(0)
    
   
for x in Covid:
    img=cv2.imread(x)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.resize(img,(100,100))
    img=img.astype('float32')
    img=img/np.max(img)
    images_Covid.append(img)
    labels_Covid.append(1)
    
images_Normal.extend(images_Covid)
labels_Normal.extend(labels_Covid)

images=np.array(images_Normal)
labels=np.array(labels_Normal)



x_train,x_test,y_train,y_test=train_test_split(images,labels,test_size=0.2,random_state=None)


from keras.utils import np_utils
y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)    


from keras.models import Sequential
from keras.layers import Conv1D,Flatten,MaxPool1D,Dense,Dropout,BatchNormalization
from keras.losses import binary_crossentropy
from keras.optimizers import Adam


model=Sequential()
model.add(Conv1D(256,3,activation='relu',input_shape=(100,100)))

model.add(Flatten())



model.add(Dropout(0.2))
model.add(Dense(64,activation='relu'))

model.add(Dense(2,activation='sigmoid'))

model.compile(loss=binary_crossentropy,optimizer=Adam(),metrics=['accuracy'])
print('_____Training Machine_____')
print('_____Please Wait...!_____')


net=model.fit(x_train,y_train,batch_size=512,epochs=60,verbose=0,validation_split=0.2)


def PlotModel(net):
    import matplotlib.pyplot as plt 
    history=net.history
    Accuracy=history['accuracy']
    ValidationAccuracy=history['val_accuracy']
    Loss=history['loss']
    ValidatioLoss=history['val_loss']

    plt.figure('Accuracy Diagram')
    plt.plot(Accuracy)
    plt.plot(ValidationAccuracy)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train Data','Validation Data'])
    plt.title('Accuracy Diagram')
    plt.show()

    plt.figure('Loss Diagram')
    plt.plot(Loss)
    plt.plot(ValidatioLoss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train Data','Validation Data'])
    plt.title('Loss Diagram')
    plt.show()


PlotModel(net)




loss,acc=model.evaluate(x_test,y_test)
print(f'loss is : {loss} accuracy is: {acc}')   



























    