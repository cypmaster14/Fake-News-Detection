import matplotlib.pyplot as plt
import numpy as np

plt.rcdefaults()


class BarPlot(object):

    def __init__(self, svm, knn, nb, mv, cc):
        objects = ('SVM', 'KNN', 'NB', 'Max Voting', 'Cascaded')
        color = ('red', 'yellow', 'blue', 'aqua', "green")

        y_pos = np.arange(len(objects))
        performance = [svm, knn, nb, mv, cc]  # Accuracy scores for all three classifiers
        plt.bar(y_pos, performance, align='center', alpha=0.5, color=color)
        plt.xticks(y_pos, objects)
        plt.ylabel('Accuracy')
        plt.title('Comparison of Accuracies')
        plt.show()
