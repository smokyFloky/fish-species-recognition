from keras.models import load_model,Model
from keras.preprocessing import image
import numpy as np
from skimage import transform
from PIL import Image


model = load_model('my_model5.h5') #un fichier qui sauvegarde tous les parametres de model (architectures, weights, optimizer)
model.compile(loss='binary_crossentropy',
                 optimizer='Adadelta',
                 metrics=['accuracy'])
def load(path): #function to convert image to grayscale
    img = Image.open(path)
    img= np.array(img).astype('float32')/255
    img = transform.resize(img, (100, 100, 1))
    img = np.expand_dims(img, axis=0)
    return img 

path4="D:/ai/rs1.jpg" #insert path of the image over here #change extension of the image
image = load(path4)
c=model.predict_classes(image)
print(c)
if(c==0):
    print("It's arctic char fish")
if(c==1):
    print("It's bass Fish")


if(c==2):
    print("It's redsnapper")