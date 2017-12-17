import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import *
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

import Src.Config as cfg

cfg.dir
warnings.filterwarnings("ignore")
pd.set_option('display.max_colwidth', 30000)
My_col = ['Retweets', 'Favorites', 'New_Feature', 'Class']


class KNNClassifier(object):

    def __init__(self, file_name):
        # Load traingin csv//
        self.Processed_file = pd.read_csv(file_name, sep=',', usecols=My_col, index_col=None)

        # Create arbitrary dataset for example
        df = pd.DataFrame({'x': self.Processed_file['Retweets'],
                           'y': self.Processed_file['Favorites'],
                           'Class': self.Processed_file['Class']}
                          )
        X = np.array(self.Processed_file.values[:, :3])
        Y = self.Processed_file['Class']
        # init the model
        self.KNN = KNeighborsClassifier(n_neighbors=101)
        self.KNN.fit(X, Y.values)

    def classify_testdata(self, filename):
        self.test_file = pd.read_csv(filename, sep=',', usecols=My_col, index_col=None)
        self.test = np.array([self.test_file['Retweets'], self.test_file['Favorites'], self.test_file['New_Feature']])
        self.test = np.array(self.test_file.values[:, :3])
        self.predicted_label = self.KNN.predict(self.test)
        return self.predicted_label

    def classify(self, x):
        output = self.KNN.predict(x)
        return output

    def plot(self):
        color = ['red' if l == 1 else 'green' for l in self.Processed_file['Class']]
        color_test = ['black' if l == 1 else 'blue' for l in self.predicted_label]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.Processed_file['Retweets'], self.Processed_file['Favorites'],
                   self.Processed_file['New_Feature'], zdir='z', s=20, depthshade=True, color=color, marker='^')
        ax.scatter(self.test_file['Retweets'], self.test_file['Favorites'], self.test_file['New_Feature'], zdir='z',
                   s=20, depthshade=True, color=color_test, marker='^')
        plt.title("KNN Classifier")
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.legend(loc=2)
        plt.show()

    def confusionMatrix(self, predict):
        Accuracy_Score = accuracy_score(self.test_file['Class'], predict)
        accuracy = Accuracy_Score * 100
        print("Accuracy for KNN")
        print(accuracy)
        print("Confusion Matrix for KNN")
        print(confusion_matrix(self.test_file['Class'], predict))
        return accuracy


if __name__ == "__main__":
    print("You are in main")
