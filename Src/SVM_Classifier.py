import Src.Config as cfg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix

cfg.dir
warnings.filterwarnings("ignore")
pd.set_option('display.max_colwidth', 30000)
My_col = ['Retweets', 'Favorites', 'New_Feature', 'Class']


class SVMClassifier(object):

    def __init__(self, file_name):
        # Load training csv
        self.Processed_file = pd.read_csv(file_name, sep=',', usecols=My_col, index_col=None)

        # Create arbitrary dataset for example
        df = pd.DataFrame({'x': self.Processed_file['Retweets'],
                           'y': self.Processed_file['Favorites'],
                           'Class': self.Processed_file['Class']}
                          )
        X = np.array(self.Processed_file.values[:, :3])
        Y = self.Processed_file['Class']
        # init the model
        self.SVM = svm.SVC(kernel='linear', C=1.0, gamma=2)
        self.SVM.fit(X, Y.values)

    def classify_testdata(self, filename):
        self.test_file = pd.read_csv(filename, sep=',', usecols=My_col, index_col=None)
        self.test = np.array([self.test_file['Retweets'], self.test_file['Favorites'], self.test_file['New_Feature']])
        self.test = np.array(self.test_file.values[:, :3])
        self.predicted_label = self.SVM.predict(self.test)
        return self.predicted_label

    def classify(self, x):
        output = self.SVM.predict(x)
        return output

    def confusionMatrix(self, predict):
        Accuracy_Score = accuracy_score(self.test_file['Class'], predict)
        print("Accuracy for SVM")
        accuracy = Accuracy_Score * 100
        print(accuracy)
        print("Confusion Matrix for SVM")
        print(confusion_matrix(self.test_file['Class'], predict))
        return accuracy

    def plot(self):
        color = ['red' if l == 1 else 'black' for l in self.Processed_file['Class']]
        color_test = ['green' if l == 1 else 'blue' for l in self.predicted_label]
        z = lambda x, y: (-self.SVM.intercept_[0] - self.SVM.coef_[0][0] * x - self.SVM.coef_[0][1]) / \
                         self.SVM.coef_[0][2]
        tmp = np.linspace(1, 140, 14)
        x, y = np.meshgrid(tmp, tmp)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(x, y, z(x, y))
        ax.scatter(self.Processed_file['Retweets'], self.Processed_file['Favorites'],
                   self.Processed_file['New_Feature'], zdir='z', s=20, depthshade=True, color=color, marker='*')
        ax.scatter(self.test_file['Retweets'], self.test_file['Favorites'], self.test_file['New_Feature'], zdir='z',
                   s=20, depthshade=True, color=color_test, marker='*')
        plt.title("SVM Classifier")
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.legend(loc=2)
        plt.show()


if __name__ == "__main__":
    print("You are in main")
