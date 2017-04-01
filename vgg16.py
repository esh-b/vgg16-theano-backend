import cv2
import h5py
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.optimizers import SGD

def VGG_16(weights_path=None, num_classes=None):

    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    if weights_path:
        f = h5py.File(weights_path)
        for k in range(f.attrs['nb_layers']):
            # Get weights for all but the output layer
            if k >= len(model.layers) - 1:⋅
                break
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            model.layers[k].set_weights(weights)

    return model

df = pd.read_csv('tw.csv')
s = df['Color'].dropna().index

#Read images from the directory named 'twimgs' and store it in a variable 'lst'
arr = []
count = 0
for i in s:
    img = cv2.imread("./twimgs/" + str(i) + ".jpg", 1)
    arr.append(img)
arr = np.asarray(arr)

#Reshape X_train to have channels_first
X_train = arr[:1000,:,:,:]
X_train = X_train.reshape(X_train.shape[0], 3, 224, 224).astype('float32')

#Reshape X_test to have channels_first
X_test = arr[1000:1200, :,:,:]
X_test = X_test.reshape(X_test.shape[0], 3, 224, 224).astype('float32')

#Normalize the values
X_train = X_train / 255
X_test = X_test / 255

#Convert categories to binary streams
x = df['Color'].loc[s].tolist()
Y = pd.get_dummies(x[:1200])

#Initialize Y_train and Y_test variables
Y_train = Y.iloc[:1000]
Y_train = np.asarray(Y_train)
Y_test = Y.iloc[1000:1200]
Y_test = np.asarray(Y_test)

#Number of classes is the number of categories
num_classes = Y.shape[1]


#Initialize the model
model = VGG_16('vgg16_weights.h5', num_classes)
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')
model.fit(X_train, Y_train, nb_epoch=10, batch_size=200)

im = cv2.resize(cv2.imread('bus.jpg'), (224, 224))
im = im.transpose((2,0,1))
im = im.reshape(1, 3, 224, 224)
out = model.predict(im)
print(out)
