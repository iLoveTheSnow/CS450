from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


# S1
iris = datasets.load_iris()

# S2
target = iris.target

data_train, data_test, target_train, target_test = train_test_split(iris.data, target, test_size=0.2)

# S3
classifier = KNeighborsClassifier(n_neighbors=3)
model = classifier.fit(data_train, target_train)

# S4
target_predicted = model.predict(data_test)
print(target_predicted)

# S5
class HardCodedClassifier:

    k = 1
    data = []
    target = []

    def __init__(self, k, data=[], target=[]):
        self.k = k
        self.data = data
        self.target = target

    def fit(self, data, target):
        self.data = data
        self.target = target
        return HardCodedClassifier(self.k, self.data, self.target)

    def prediction(self, test_data):
        numInputs = np.shape(test_data)[0]
        closest =np.zeros(numInputs)

        for a in range(numInputs):
            distances = np.sum((self.data-test_data[a, :])**2, axis=1)
            indices = np.argsort(distances, axis=0)
            classes = np.unique(self.target[indices[:self.k]])

            if len(classes) == 1:
                closest[a] = np.unique(classes)
            else:
                 count = np.zeros(max(classes) + 1)
                 for i in range(self.k):
                    count[self.target[indices[i]]]
                 closest[a] = np.max(count)


        return closest

    def percentScore(self, data_test, target_test):
        total = 0
        correct = 0

        for x in data_test:
            total += 1
        y = 0
        while y < total:
            if data_test[y] == target_test[y]:
                correct += 1
            y += 1

        return float(correct / total)

BlankClassifier = HardCodedClassifier(2)

BlankModel = BlankClassifier.fit(data_train, target_train)
BlankPredicted = BlankModel.prediction(data_test)
print("Percentage of HardCoded Correct:")
print(100 * BlankModel.percentScore(BlankPredicted, target_test))


