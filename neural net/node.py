import numpy as np
import math

class Node(object):
    BIAS_X = 1
    LEARNING_RATE = 0.4

    def __init__(self):
        self.inputs = []
        self.output = 0
        self.target = 0
        self.weights = []
        self.threshold = 0
        self.error = 0

    def set_inputs(self, inputs: list):
        self.inputs = [ self.BIAS_X ]
        self.inputs.extend(inputs)

    def set_weights(self, num_weights):
        self.weights = np.random.ranf(num_weights) - 0.5

    def activation_function(self, total):
        self.output = 1 / (1 + math.e**(-1*total))

    def compute_error(self, isHidden, right_node_weights=None, right_node_errors=None):
        if isHidden:
            self.error = self.output*(1 - self.output)*sum([a*b for a, b in zip(right_node_weights, right_node_errors)])
        else:
            self.error = self.output*(1 - self.output)*(self.output - self.target)

    def update_weights(self, prev_outputs):
        for i in range(self.weights.__len__()):
            self.weights[i] -= (self.LEARNING_RATE * self.error * prev_outputs[i])
