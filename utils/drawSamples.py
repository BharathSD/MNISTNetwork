# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt

class displayImages:
    def __init__(self):
        self.cmap='gray'
        self.interpolation='none'

    def plotImages(self, ImageList:list, LabelList:list):
        fig = plt.figure()
        noOfSamples = len(ImageList)
        noOfCols = 3
        noOfRows = int(noOfSamples / noOfCols)
        noOfRows = noOfRows + 1 if noOfSamples % noOfCols is not 0 else noOfRows

        for i in range(noOfSamples):
            plt.subplot(noOfRows, noOfCols, i + 1)
            plt.tight_layout()
            plt.imshow(ImageList[i], cmap=self.cmap, interpolation=self.interpolation)
            plt.title("Digit: {}".format(LabelList[i]))
            plt.xticks([])
            plt.yticks([])
        fig.show()

    def plotPixelValueDistribution(self, image, index):
        fig = plt.figure()
        plt.subplot(2, 1, 1)
        plt.imshow(image, cmap='gray', interpolation='none')
        plt.title("Digit: {}".format(index))
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2, 1, 2)
        plt.hist(image.reshape(-1,1))
        plt.title("Pixel Value Distribution")
        fig