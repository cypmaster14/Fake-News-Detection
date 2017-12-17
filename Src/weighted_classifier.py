import math

import Code.Src.Config as cfg
import Code.Src.KNN_Classifier as knn
import Code.Src.NB_Classfier as nb
import Code.Src.SVM_Classifier as svm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    print(len(test_data))
    print(test_data)
    finalized_outputs = []
    fully_credible = []
    fully_spam = []
    partially_credible = []
    partially_spam = []
    for test in test_data:
        try:
            test = test.reshape(1, 3)
            a, b = nb_classifier.classify(test)
            pw1 = abs((b[0][0]) / (b[0][0] + b[0][1]))
            pw2 = abs((b[0][1]) / (b[0][0] + b[0][1]))
            pw1_normalized = math.ceil(pw1 * 100.0) / 100.0
            pw2_normalized = math.ceil(pw2 * 100.0) / 100.0
            error = min(pw1_normalized, pw2_normalized)
            error_rate = math.ceil(error * 100.0) / 100.0
            if error_rate > 0.2:
                if a[0] == 1:
                    partially_credible.append(a[0])
                else:
                    partially_spam.append(a[0])
                # print(" ")
            else:
                if a[0] == 1:
                    fully_credible.append(a[0])
                else:
                    fully_spam.append(a[0])
        except:
            print("Error")

    print("$$$$$$$$$$$$$$$$$$ Post Process: Analysis $$$$$$$$$$$$$$$$$$$$$$$")
    print("")
    print("Total Postprocessed Tweets: ", len(test_data))
    print("")
    print("Fully Credible Tweets: ", len(fully_credible))
    print("Partially Credible Tweets: ", len(partially_credible))
    print("Partially Spam Tweets: ", len(partially_spam))
    print("Fully Spam Tweets: ", len(fully_spam))
    print("")

    objects = ('Fully Credible', 'Partially Credible', 'Partially Spam', 'Fully Spam')
    color = ('red', 'yellow', 'blue', 'aqua')
    y_pos = np.arange(len(objects))
    performance = [len(fully_credible), len(partially_credible), len(partially_spam),
                   len(fully_spam)]  ## Accuracy scores for all three classifiers
    plt.bar(y_pos, performance, align='center', alpha=0.5, color=color)
    plt.xticks(y_pos, objects)
    plt.ylabel('Number of tweets')
    plt.title('Weighted Tweets')
    plt.show()
