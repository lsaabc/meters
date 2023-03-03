from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from data_preprocess import DataPreprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class ModelBuilder2(DataPreprocessing):
    def __init__(self, *args, **kwargs):
        super(ModelBuilder2, self).__init__(*args, **kwargs)

    def ANN(self, X_train, X_test, y_train, y_test):
        #Create DT model
        DT_classifier2 = MLPClassifier()

        #Train the model
        DT_classifier2.fit(X_train, y_train)

        #Test the model
        DT_predicted2 = DT_classifier2.predict(X_test)

        error2 = 0
        for i in range(len(y_test)):
            error2 += np.sum(DT_predicted2 != y_test)

        total_accuracy2 = 1 - error2 / len(y_test)

        #get performance
        self.accuracy2 = accuracy_score(y_test, DT_predicted2)

        return DT_classifier2