import numpy as np


class Node(object):

    def __init__(self, data, targets, cols, root=False):
        self.nodes = []
        self.feature = None
        self.value = None
        self.data = data
        self.targets = targets
        self.cols = cols
        self.entropy = self.calculate_entropy()

    def create_next_node(self):

        target_set = np.unique(self.targets)
        if len(target_set) == 1:
            self.value = target_set[0]
            return
        elif len(self.cols) == 0:
            tar_freq = []
            for tar in target_set:
                tar_freq.append(self.targets.tolist().count(tar))
            self.value = target_set[tar_freq.index(max(tar_freq))]
            return

        row_count = len(self.data[:, 0])
        information_gain = []
        category_nodes = []
        for feature in self.cols:
            gain = self.entropy
            temp_nodes = []
            for pos_values in np.unique(self.data[:, feature]):
                prob = self.data[:, feature].tolist().count(pos_values) / row_count

                temp_data = np.append(self.data, np.reshape(self.targets, (-1, 1)), 1).T
                temp_data = temp_data[:, temp_data[feature] == pos_values].T
                temp_cols = self.cols[:]
                temp_cols.remove(feature)

                temp_node = Node(temp_data[:, :len(temp_data[0]) - 1], temp_data[:, len(temp_data[0]) - 1:].astype(np.float), temp_cols)

                gain -= prob * temp_node.entropy
                temp_nodes.append(temp_node)
            category_nodes.append(temp_nodes)
            information_gain.append(gain)

        index_of_next_feature = information_gain.index(max(information_gain))
        self.feature = self.cols[index_of_next_feature]
        self.nodes = category_nodes[index_of_next_feature]

        # Recurse
        for node in self.nodes:
            node.create_next_node()

        return

    def calculate_entropy(self):
        sum = 0
        for x in np.unique(self.targets):
            prob = self.targets.tolist().count(x) / len(self.targets)
            sum += -1 * prob * np.log2(prob)
        return sum

class Leaf(Node):

    def __init__(self):
        self.nodes = []
        self.value = 0

class DecisionTree(object):

    def __init__(self, dataset):
        self.dataset = dataset

        self.classes = dataset.target_count
        self.cols = list(range(0, len(dataset.data[0])))

        self.tree = Node.Node(self.dataset.training_data, self.dataset.training_targets, self.cols, root=True)

    def create_tree(self):
        self.tree.create_next_node()

    def predict_numeric(self):
        predicted_targets = []
        for row in self.dataset.test_data:
            predicted_targets.append(self.evaluate_numeric_node(row, self.tree))

        return predicted_targets

    def evaluate_numeric_node(self, row, node):
        if node.value:
            return node.value
        else:
            upper_bound = 0
            for i in reversed(self.dataset.rules[node.feature]):
                if row[node.feature] > i:
                    break
                elif upper_bound == len(node.nodes) - 1:
                    break
                upper_bound += 1
            return self.evaluate_numeric_node(row, node.nodes[len(node.nodes) - (1 + upper_bound)])

    def predict_nominal(self):
        predicted_targets = []
        for row in self.dataset.test_data:
            predicted_targets.append(self.evaluate_nominal_node(row, self.tree))

        return predicted_targets

    def evaluate_nominal_node(self, row, node):
        if node.value != None:
            return node.value
        else:
            value_set = np.unique(self.dataset.training_data[:, node.feature]).tolist()
            indexOfNode = value_set.index(row[node.feature])
            return self.evaluate_nominal_node(row, node.nodes[indexOfNode])

    def print_tree(self):
        print("root split on feature {}".format(self.tree.feature))
        for node in self.tree.nodes:
            self.node_to_string(node)

    def node_to_string(self, node):
        print("    ", end='')
        if node.value is not None:
            print("|___{}".format(node.value))
        else:
            print("Node split on feature {}".format(node.feature))
            for node in node.nodes:
                self.node_to_string(node)

class Dataset(object):

    def __init__(self):
        self.DESCR = ""
        self.data = np.array([])
        self.target = np.array([])
        self.input_count = 0
        self.target_count = 0
        self.training_data = np.array([])
        self.test_data = np.array([])
        self.training_targets = np.array([])
        self.test_targets = np.array([])
        self.predicted_targets = np.array([])
        self.means = np.array([])
        self.standard_devs = np.array([])

    def randomize_data(self):
        reorder = np.random.permutation(len(self.data))
        self.data = self.data[reorder]
        self.target = self.target[reorder]

    def split_data(self, training_percent=70):
        # Default 70/30, can change
        training_size = round(len(self.data) * (training_percent / 100))
        self.training_data, self.test_data = np.split(self.data, [training_size])
        self.training_targets, self.test_targets = np.split(self.target, [training_size])

    def standardize_data(self):
        standardized_data = self.training_data.T
        standardized_test_data = self.test_data.T
        col_means = []
        col_stds = []

        for x in range(len(standardized_data)):
            col_means.append(np.mean(standardized_data[x]))
            col_stds.append(np.std(standardized_data[x]))
            standardized_data[x] = [(el - col_means[x]) / col_stds[x] for el in standardized_data[x]]

        for x in range(len(standardized_test_data)):
            standardized_test_data[x] = [(el - col_means[x]) / col_stds[x] for el in standardized_test_data[x]]

        self.training_data = standardized_data.T
        self.test_data = standardized_test_data.T
        self.means = np.array(col_means)
        self.standard_devs = np.array(col_stds)
        self.input_count = self.training_data.shape[1]
        self.target_count = len(np.unique(self.training_targets))

    def discretize_data(self, sections=30):
        # inefficient but... ¯\_(ツ)_/¯ Discretize!
        self.rules = []

        for col in range(len(self.training_data[0])):
            low = min(self.training_data[:, col])
            high = max(self.training_data[:, col])
            rule = np.arange(low, high, (high - low) / sections)
            self.rules.append(rule)

            for i, item in enumerate(self.training_data[:, col]):
                if item < rule[0]:
                    self.training_data[:, col][i] = -1
                    continue
                for x, bound in enumerate(rule):
                    if x == 0:
                        continue
                    if item < bound:
                        self.training_data[:, col][i] = x - 1
                        break
                    else:
                        self.training_data[:, col][i] = x
                else:
                    continue

            # Never copy and paste in code, they say
            for i, item in enumerate(self.test_data[:, col]):
                if item < rule[0]:
                    self.test_data[:, col][i] = -1
                    continue
                for x, bound in enumerate(rule):
                    if x == 0:
                        continue
                    if item < bound:
                        self.test_data[:, col][i] = x - 1
                        break
                    else:
                        self.test_data[:, col][i] = x
                else:
                    continue

    def report_accuracy(self):
        correct = 0
        for i in range(len(self.test_targets)):
            if self.test_targets[i] == self.predicted_targets[i]:
                correct += 1
        percentage = round(correct / len(self.test_targets), 2) * 100
        print("Predicting targets at {}% accuracy".format(percentage))

    # All the loads will be here. Hardcoded in so that Experiment.py is cleaner
    # Discretization is done in the load file if needed

    def load_iris(self):
        with open("datasets/iris.names.txt") as f:
            self.DESCR = f.readlines()

        raw_data = np.genfromtxt("datasets/iris.data.txt", dtype=str, delimiter=',')

        self.data = raw_data[:, :len(raw_data[0]) - 1].astype(np.float)
        self.target = np.array([0 if el == "Iris-setosa" else 2 if el == "Iris-virginica" else 1
                                for el in raw_data[:, len(raw_data[0]) - 1:].flatten()])
        self.randomize_data()
        self.split_data()
        self.standardize_data()
        self.discretize_data()
