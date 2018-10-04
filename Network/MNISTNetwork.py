from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential

class MNISTNetwork:
    def __init__(self):
        # have hard coded it because we know the shape of the input
        self.dataShape = (28,28)
        self.noOfPixels = 784
        self.noOfClasses = 10

    def getNNModel(self):
        # building a linear stack of layers with the sequential model
        model = Sequential()
        model.add(Dense(512, input_shape=(self.noOfPixels,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(self.noOfClasses))
        model.add(Activation('softmax'))
        return model


    def getCNNModel(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=self.dataShape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.noOfClasses, activation='softmax'))
        return model