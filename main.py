from Data.MNISTdata import MNIST_data
from Network.MNISTNetwork import MNISTNetwork
from utils.drawSamples import displayImages
import numpy as np
import matplotlib
matplotlib.use('QT4agg')
import matplotlib.pyplot as plt
import configparser
import os

def parseConfigurations(configPath:str):
    print('parsing configfile...')
    cfg = configparser.ConfigParser()
    cfg.read(configPath)
    return cfg

class TrainigParams:
    def __init__(self, batch_size:int, epochs:int):
        self.batch_size = batch_size
        self.epochs = epochs

class Application:
    def __init__(self, model_path:str, trainingParams:TrainigParams):
        self.model_path = model_path
        #load MNIST data
        (self.X_train, self.Y_train), (self.X_test, self.Y_test) =  MNIST_data().getMNISTdata()
        # get the model
        self.model = MNISTNetwork().getNNModel()

        # compiling the sequential model
        self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

        # update training Parameters
        self.trainingParams = trainingParams

    def train(self):
        # training the model and saving metrics in history
        history = self.model.fit(self.X_train, self.Y_train,
                                 batch_size=self.trainingParams.batch_size,
                                 epochs=self.trainingParams.epochs,
                                 verbose=2,
                                 validation_data=(self.X_test, self.Y_test))

        # saving the model
        self.model.save(self.model_path)
        print('Saved trained model at %s ' % self.model_path)

        # plotting the metrics
        fig = plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='lower right')

        plt.subplot(2, 1, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')

        plt.tight_layout()

        fig.show()
        fig.savefig('TrainingParams.png')

    def inference(self):
        loss_and_metrics = self.model.evaluate(self.X_test, self.Y_test, verbose=2)

        print("Test Loss", loss_and_metrics[0])
        print("Test Accuracy", loss_and_metrics[1])

        # this is for prediction
        predicted_classes = self.model.predict_classes(self.X_test)


        # load original data
        (X_test, y_test) = MNIST_data().getTestingData()

        # see which we predicted correctly and which not
        correct_indices = np.nonzero(predicted_classes == y_test)[0]
        incorrect_indices = np.nonzero(predicted_classes != y_test)[0]
        print()
        print(len(correct_indices), " classified correctly")
        print(len(incorrect_indices), " classified incorrectly")

        # adapt figure size to accomodate 18 subplots
        plt.rcParams['figure.figsize'] = (7, 14)

        figure_evaluation = plt.figure()

        # plot 9 correct predictions
        for i, correct in enumerate(correct_indices[:9]):
            plt.subplot(6, 3, i + 1)
            plt.imshow(X_test[correct].reshape(28, 28), cmap='gray', interpolation='none')
            plt.title(
                "Predicted: {}, Truth: {}".format(predicted_classes[correct],
                                                  y_test[correct]))
            plt.xticks([])
            plt.yticks([])

        # plot 9 incorrect predictions
        for i, incorrect in enumerate(incorrect_indices[:9]):
            plt.subplot(6, 3, i + 10)
            plt.imshow(X_test[incorrect].reshape(28, 28), cmap='gray', interpolation='none')
            plt.title(
                "Predicted {}, Truth: {}".format(predicted_classes[incorrect],
                                                 y_test[incorrect]))
            plt.xticks([])
            plt.yticks([])

        figure_evaluation.show()
        figure_evaluation.savefig('result.png')

if __name__ == '__main__':
    cfg = parseConfigurations(r'Configuration.ini')
    TrainigParamsI = TrainigParams(batch_size=int(cfg['TrainingParams']['batch_size']),
                                   epochs= int(cfg['TrainingParams']['epochs']))
    ModelFilePath = cfg['Model']['ModelPath']
    ModelFileName = cfg['Model']['ModelName']
    ModelFile = ModelFileName
    if os.path.exists(ModelFilePath):
        ModelFile = os.path.join(ModelFilePath, ModelFileName)

    modelFilePath = ModelFile
    app = Application(model_path=modelFilePath, trainingParams=TrainigParamsI)
    
    app.train()
    app.inference()