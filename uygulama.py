import pyautogui
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import numpy as np
import cv2
from keras.models import Sequential
import keras
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation,Dense,Conv2D,Dropout,MaxPooling2D,Dropout,Flatten 
import time 

girisverisi =np.load("girisverimiz_.npy")
cikisverisi = np.load("cikisverimiz_.npy")
splitverisi = girisverisi[0:50]
splitcikis = np.array([[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],
                       [0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],
                       [0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],
                       [0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],
                       [0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0]])

splitverisi = splitverisi.reshape(-1,300,300,1)

model = Sequential()
model.add(Conv2D(50,11,input_shape =(300,300,1)))
model.add(Conv2D(50,2))
model.add(Conv2D(50,2))
model.add(Conv2D(50,2))
model.add(Conv2D(50,2))
model.add(Conv2D(50,3))
model.add(Conv2D(50,2))
model.add(Conv2D(50,2))
model.add(Conv2D(50,2))
model.add(MaxPooling2D(5,5))
model.add(Conv2D(50,3))
model.add(Conv2D(50,3))
model.add(Conv2D(50,3))
model.add(Conv2D(50,3))
model.add(Conv2D(50,2))
model.add(Conv2D(50, 3))
model.add(Conv2D(50,2))
model.add(Conv2D(50,2))
model.add(Conv2D(50,2))
model.add(Conv2D(50,2))
model.add(Conv2D(50,2))
model.add(MaxPooling2D(3,3))
model.add(Conv2D(50,2))
model.add(Conv2D(50,3))
model.add(Conv2D(50,3))
model.add(Conv2D(50,3))
model.add(Flatten())

model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(1000, activation='relu'))

model.add(Dense(2))
model.add(Activation("softmax"))
model.compile(loss ="binary_crossentropy",optimizer= keras.optimizers.RMSprop(lr=0.00001),metrics=["accuracy"])
model.summary()
model.fit(girisverisi/255,cikisverisi,batch_size=1,epochs=10,validation_data=(splitverisi,splitcikis))

#model.load_weights("metinmodeli")

girisverisi2=np.array([])
def ResimTahmin(resim,metin,metindegil):
    girdi = []
    resim = np.array(resim)
    boyutlandilirmisresim = cv2.resize(resim,(300,300))
    girdi = np.append(girdi,boyutlandilirmisresim)
    girdi = np.reshape(girdi,(-1,300,300,1))
    tahmin =model.predict(girdi/255)
    metin =tahmin [0][0]
    metindegil = tahmin[0][1]
    
# Burada kesilecek resmin sol,sag,üst,alt konumunu belirliyoruz.

# Toplam 50 resim oluşacak

while True:
    sol =0
    sag =0
    ust =40
    alt =200
    metin =0
    metindegil =0
    for i in range(1,6):
        sol =40
        sag =200
        im_= pyautogui.screenshot()
        im__ =cv2.cvtColor(np.array(im_), cv2.COLOR_BGR2HSV)
        gray_im = cv2.cvtColor(im__, cv2.COLOR_BGR2GRAY)
        for j in range(1,9):
            sol = sol + 230
            sag = sag +230
            im = gray_im[ust:alt,sol:sag] # defines crop points
            sayi =0
            #img = cv2.cvtColor(np.float32(im), cv2.COLOR_RGB2GRAY) // Farklı dönüşümlere bak
            girdi = []
            boyutlandilirmisresim = cv2.resize(np.float32(im),(300,300))
            girdi = np.append(girdi,boyutlandilirmisresim)
            girdi = np.reshape(girdi,(-1,300,300,1))
            tahmin =model.predict(girdi/255)
            metin =tahmin [0][0]
            metindegil = tahmin[0][1]
            print(f"{metin},{metindegil}")
            if metin>0.7 and metin<=1:
                print("Metin bulundu !!! ")
                pyautogui.leftClick(sol+30,ust-30)
                pyautogui.leftClick(sol+30,ust-30)
                time.sleep(5)
                time.sleep(0.2)
        ust = ust+ 180
        alt = alt +180


























    
    