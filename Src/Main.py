import Src.Bar_graph_plot as br
import Src.Config as cfg
import Src.KNN_Classifier as knn
import Src.NB_Classfier as nb
import Src.PreProcessing as prp
import Src.SVM_Classifier as svm
import numpy as np
from Src.cascaded_classifier import CascadeClassifier
from Src.max_voting_classifier import MaxVotingClassifier

cfg.dir

testing_file_location = 'Test_feature_extracted.csv'
training_file_location = 'Training_feature_extracted.csv'

if __name__ == "__main__":
    training_data = prp.PreProcessing('../Data/RawTrainingDataSet.csv', 'Training')
    test_data = prp.PreProcessing('../Data/RawTestDataSet.csv', 'Test')
    svm_classifier = svm.SVMClassifier(training_file_location)
    knn_classifier = knn.KNNClassifier(training_file_location)
    nb_classifier = nb.NbClassifier(training_file_location)

    test_point = np.array([5, 8, 3])
    test_point = test_point.reshape(1, 3)
    predicted_class = svm_classifier.classify(test_point)
    print(predicted_class)
    predict_test_data_class = svm_classifier.classify_testdata(testing_file_location)
    accuracy_svm = svm_classifier.confusion_matrix(predict_test_data_class)
    svm_classifier.plot()

    test_knn_point = np.array([57, 82, 3])
    test_knn_point = test_knn_point.reshape(1, 3)
    predicted_class = knn_classifier.classify(test_knn_point)
    print(predicted_class)
    predict_test_data_class = knn_classifier.classify_testdata(testing_file_location)
    accuracy_knn = knn_classifier.confusion_matrix(predict_test_data_class)
    knn_classifier.plot()

    predict_test_data_class = nb_classifier.classify_testdata(testing_file_location)
    accuracy_nb = nb_classifier.confusion_matrix(predict_test_data_class)
    nb_classifier.plot()

    cascade_classifier = CascadeClassifier(nb_classifier, knn_classifier, svm_classifier)
    cascade_classifier.test(testing_file_location)
    accuracy_cc = cascade_classifier.accuracy

    max_voting_classifier = MaxVotingClassifier(nb_classifier, knn_classifier, svm_classifier)
    max_voting_classifier.test(testing_file_location)
    accuracy_mv = max_voting_classifier.accuracy

    br.BarPlot(accuracy_svm, accuracy_knn, accuracy_nb, accuracy_mv, accuracy_cc)
