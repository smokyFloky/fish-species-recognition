import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Flatten, Dropout,Activation
from keras.layers.convolutional import MaxPooling2D, Conv2D


import os
from PIL import Image,ImageFile
import numpy as np
import array as arr
import matplotlib.pyplot as pl
from keras.utils import np_utils, plot_model
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import h5py


gpu_options =tf.compat.v1.GPUOptions(allow_growth=True)
sess =tf.compat.v1.Session()

ImageFile.LOAD_TRUNCATED_IMAGES = True



path3="D:/ai/res3"


# # listing = os.listdir(path2)

num_sample = 491

# On a utilisé ce bloc de code pour convertir les images en grayscale
# listing = os.listdir(path2)
#  for file in listing:
#    im = Image.open(path2+"\\"+file)
#    img = im.resize((200,200))
#    gray = img.convert("L")
#    gray.save(path3+"\\"+file, "JPEG")



imlist = os.listdir(path3)

imatrix = np.array([np.around(Image.open(path3+"\\"+im2)).flatten()  #1 chaque matrice est transofrmer a un tableau grace a la fonction flatten()
            for im2 in imlist],"f" )

label = np.ones((num_sample,),dtype=int)

label[0:168] = 0   #2 etiqueter chaque ensemble avec un numero
label[168:329] =1
label[329:] = 2


data,Label= shuffle(imatrix, label, random_state=3)  #3 shuffling les données aleatoirmenent vue que les images sont triées

train_data = [data,label]
print(train_data)


#setting the parameters

batch_size= 64 #le model entraine les images en 64 parties

nb_classe= 3

nb_epoch = 20 #nombre d'iteration

img_rows, img_cols = 100, 100
img_channels = 1 #grayscale

nb_filters = 32

nb_pool = 2

kernel_size= 3

(X, y)= (train_data[0], train_data[1])

#step 1 : split x and y into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

X_train = X_train.reshape(X_train.shape[0],img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0],img_rows, img_cols, 1)
X_train= X_train.astype("float32")

X_test = X_test.astype("float32")

X_train/=255
X_test/=255

print(X_train)

y_train= np_utils.to_categorical(y_train, nb_classe)
y_test = np_utils.to_categorical(y_test, nb_classe)

#architecture
model = Sequential()
#convolutional layer
model.add(Conv2D(nb_filters, kernel_size, kernel_size,
                 border_mode="valid",

                 input_shape=(img_cols, img_rows, 1)))

convOut1= Activation("relu")
model.add(convOut1)

#convolutiona layer
model.add(Conv2D(nb_filters, (kernel_size,kernel_size)))
convOut2= Activation("relu")
model.add(convOut2)

#maxpooling layer
model.add(MaxPooling2D(pool_size=(nb_pool,nb_pool)))

#flatten layer
model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))
model.add(Dropout(0.4))
model.add(Dense(nb_classe))
model.add(Activation("softmax"))

#compiling the model
model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=["accuracy"])

#training and printing the results

model.fit(X_train,y_train,batch_size=batch_size,nb_epoch=nb_epoch,verbose=1,validation_split=0.4,validation_data=(X_test,y_test))
score = model.evaluate(X_test, y_test, verbose=0)

model.save("my_model.h5") #sauvegarder le modele