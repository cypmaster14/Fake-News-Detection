import math

import Src.Config as cfg
import Src.KNN_Classifier as knn
import Src.NB_Classfier as nb
import Src.SVM_Classifier as svm
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import *

cfg.dir

columns_header = ['Retweets', 'Favorites', 'New_Feature', 'Class']
training_file_location = "Training_feature_extracted.csv"
testing_file_location = "Test_feature_extracted.csv"

'''
Evaluating Cascade Classifier
The three algorithms are trained.
Classify the testing dataset.
First, Naive Bayes is used.
If the error rate is higher then .2 (0.2), that entry is classified using the other two algorithms.
    The output will be the classified label that is in majority.
Otherwise, the output will be the classified label provided by Naive Bayes 
'''


class CascadeClassifier(object):
    class_column_name = "Class"

    def __init__(self, training_file):
        self.nb_classifier = nb.NbClassifier(training_file)
        self.knn_classifier = knn.KNNClassifier(training_file)
        self.svm_classifier = svm.SVMClassifier(training_file)
        self.accuracy = 0

    def __init__(self, nb_classifier, knn_classifier, svm_classifier):
        self.nb_classifier = nb_classifier
        self.knn_classifier = knn_classifier
        self.svm_classifier = svm_classifier
        self.accuracy = 0

    def test(self, testing_file):
        test_file = pd.read_csv(testing_file, sep=",", usecols=columns_header, index_col=None)
        test_data = np.array(test_file.values[:, :3])
        test_data_class = test_file[self.class_column_name]

        outputs = []
        for test_entry in test_data:
            test_entry = test_entry.reshape(1, 3)
            nb_output, nb_probability = self.nb_classifier.classify(test_entry)
            pw1 = abs((nb_probability[0][0]) / (nb_probability[0][0] + nb_probability[0][1]))
            pw2 = abs((nb_probability[0][1]) / (nb_probability[0][0] + nb_probability[0][1]))
            pw1_normalized = math.ceil(pw1 * 100.0) / 100.0
            pw2_normalized = math.ceil(pw2 * 100.0) / 100.0
            error = min(pw1_normalized, pw2_normalized)
            error_rate = math.ceil(error * 100.0) / 100.0
            if error_rate > .2:
                # print(error_rate)
                svm_output = self.svm_classifier.classify(test_entry)
                knn_output = self.knn_classifier.classify(test_entry)
                algorithms_predicted_outputs = [svm_output[0], knn_output[0]]
                output = max(algorithms_predicted_outputs, key=algorithms_predicted_outputs.count)
                outputs.append(output)
                # print(" ")
            else:
                outputs.append(nb_output[0])

        cm = metrics.confusion_matrix(test_data_class, outputs)
        self.accuracy = accuracy_score(test_data_class, outputs)
        self.accuracy = self.accuracy * 100

        print("Accuracy")
        print(self.accuracy)
        print("Confusion Matrix")
        print(cm)


if __name__ == "__main__":
    cascade_classifier = CascadeClassifier(training_file_location)
    cascade_classifier.test(testing_file_location)
