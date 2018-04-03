def get_accuracy(predictions, actuals):

    count = 0

    for index in range(actuals.__len__()):
        if int(predictions[index]) is int(actuals[index]):
            count += 1

    return round((count / actuals.__len__()) * 100, 2)


class Classifier(object):


    def __init__(self):
        self.training_set   = None
        self.testing_set    = None
        self.validation_set = None

    def train(self, dataset):

        if self.training_set is None:
            self.training_set = dataset
        else:
            self.training_set.add_to_data(dataset.data, dataset.target, dataset.target_names)

    def predict(self, dataset):
        return 0
