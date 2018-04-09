from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras import backend as K
from keras.layers import Activation
from keras.layers import MaxPool2D
from keras.layers import Flatten,Dense

class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        if K.image_data_format() == 'channels_first':
            input_shape = (depth, height, width)
        else:
            input_shape = (height, width, depth)
        model = Sequential()
        model.add(Conv2D(10, (5, 5), padding='same', input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(40, (5, 5), padding='same'))
        model.add(Activation("relu"))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model