import Src.Config as cfg
import Src.KNN_Classifier as knn
import Src.NB_Classfier as nb
import Src.SVM_Classifier as svm
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import *

cfg.dir

My_col = ['Retweets', 'Favorites', 'New_Feature', 'Class']

if __name__ == "__main__":
    nb_classifier = nb.NbClassifier('Training_feature_extracted.csv')
    KnnClassifier = knn.KNNClassifier('Training_feature_extracted.csv')
    SvmClassifier = svm.SVMClassifier('Training_feature_extracted.csv')

    test_file = pd.read_csv('Test_feature_extracted.csv', sep=',', usecols=My_col, index_col=None)
    test_data = np.array([test_file['Retweets'], test_file['Favorites'], test_file['New_Feature']])
    test_data = np.array(test_file.values[:, :3])
    test_data_class = test_file.Class
    # print(len(test_data))
    # print(test_data)
    finalized_outputs = []
    for test in test_data:
        test = test.reshape(1, 3)
        a, b = nb_classifier.classify(test)
        svm_output = SvmClassifier.classify(test)
        knn_output = KnnClassifier.classify(test)
        predicted_outputs = [svm_output[0], knn_output[0], a[0]]
        finalized_output = max(predicted_outputs, key=predicted_outputs.count)
        finalized_outputs.append(finalized_output)
    cm = metrics.confusion_matrix(test_data_class, finalized_outputs)
    accuracy = accuracy_score(test_data_class, finalized_outputs)
    print("Accuracy")
    print(accuracy * 100)
    print("Confusion Matrix")
    print(cm)
