from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np

class MNIST_data:
    def __init__(self):
        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()
        self.noOfTrainingSamples = self.X_train.shape[0]
        self.noOfTestingSamples = self.X_test.shape[0]
        self.dataShape = self.X_train[0].shape
        self.noOfPixels = np.prod(self.dataShape)

        self.labels, self.dataDistribution = np.unique(self.y_train, return_counts=True)

    def getData(self):
        return (self.X_train, self.y_train), (self.X_test, self.y_test)

    def getTrainingData(self):
        return (self.X_train, self.y_train)

    def getTestingData(self):
        return (self.X_test, self.y_test)

    def getMNISTdata(self,  reshape:bool=True, Normalize:bool=True, oneHotEncoding:bool=True):
        X_train = self.X_train
        X_test = self.X_test
        Y_train = self.y_train
        Y_test = self.y_test

        if reshape:
            X_train = self.X_train.reshape(self.noOfTrainingSamples, self.noOfPixels)
            X_test = self.X_test.reshape(self.noOfTestingSamples, self.noOfPixels)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        # normalizing the data to help with the training
        if Normalize:
            X_train /= 255
            X_test /= 255

        # one hot encoding
        if oneHotEncoding:
            noOfclasses = len(self.labels)
            Y_train = np_utils.to_categorical(Y_train, noOfclasses)
            Y_test = np_utils.to_categorical(Y_test, noOfclasses)

        return (X_train, Y_train), (X_test, Y_test)