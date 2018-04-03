from sklearn.preprocessing import normalize
from sknn.mlp import Classifier, Layer
import numpy as np

from dataset import Dataset
from neural_network import NeuralNetwork
from classifier import get_accuracy
from csv_complier import create_csv_file

def write_to_results_file(file, data):
    with open(file, mode='a') as results:
        results.write(data)


def main():
    while True:
        data_set_name = input("Please provide the name of the data set you want to work with: ")

        # Load, Randomize, Normalize, Discretize Dataset
        data_set = Dataset()
        data_set.read_file_into_dataset("\\Users\\nathanielfacuri\\Desktop\\" + data_set_name)
        data_set.randomize()
        data_set.data = normalize(data_set.data)
        #data_set.discretize()
        #print(data_set.data)

        data_set.set_missing_data()

        # Split Dataset
        split_percentage = 0.7
        data_sets    = data_set.split_dataset(split_percentage)
        training_set = data_sets[0]
        testing_set  = data_sets[1]

        # Create Custom Classifier, Train Dataset, Predict Target From Testing Set
        iterations = int(input("How many iterations do you want to do? "))
        layers = int(input("How many layers do you want in your neural network? "))
        num_nodes = []
        for i in range(layers):
            if i + 1 == layers:
                number = int(input("How many nodes on the output layer? "))
            else:
                number = int(input("How many nodes on the " + str(i) + " layer? "))
            num_nodes.append(number)

        neuralNetwork = NeuralNetwork(iterations)
        neuralNetwork.create_layered_network(num_nodes, training_set.feature_names.__len__())
        #neuralNetwork.display_network()
        neuralNetwork.train(training_set)
        predictions = neuralNetwork.predict(testing_set)

        # Check Results
        my_accuracy = get_accuracy(predictions, testing_set.target)
        print("Accuracy: " + str(my_accuracy) + "%")

        # Compare To Existing Implementations
        layers_objs = []
        for i in range(layers):
            if i + 1 == layers:
                layers_objs.append(Layer("Softmax", units=num_nodes[i]))
            else:
                layers_objs.append(Layer("Sigmoid", units=num_nodes[i]))

        mlp_nn = Classifier(layers=layers_objs, learning_rate=0.4, n_iter=iterations)
        mlp_nn.fit(np.array(training_set.data), np.array(training_set.target))
        predictions = mlp_nn.predict(np.array(testing_set.data))

        mlp_nn_accuracy = get_accuracy(predictions, testing_set.target)
        print("NN Accuracy: " + str(mlp_nn_accuracy) + "%")

        create_csv_file(neuralNetwork.accuracies, "\\Users\\nathanielfacuri\\Desktop\\" + data_set_name + ".csv")
        # Do another or not
        toContinue = False

        while True:
            another = input("Do you want to examine another dataset? (y / n) ")

            if another != 'y' and another != 'n':
                print("Please provide you answer in a 'y' or 'n' format.")
            elif another == 'y':
                toContinue = True
                break
            else:
                toContinue = False
                break

        if not toContinue:
            break

# Produce a textual view of your resulting tree ASK

if __name__ == '__main__':
    main()
