import math
import random as rand
import os

# create a class for the neural network
class NeuralNetwork():
    def __init__(self, num_inputs, num_hidden, num_outputs):
        # set the number of nodes in each layer
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        # initialize the weights and biases
        self.weights = []
        self.biases = []

        # initialize the outputs
        self.hidden_outputs = []
        self.output_outputs = []

        # create the weights for the hidden layer
        for i in range(self.num_hidden):
            # create a list of weights for each node in the hidden layer
            weights = []
            for j in range(self.num_inputs):
                # create a random weight between -1 and 1
                weights.append(rand.uniform(-1, 1))
            # add the list of weights to the list of weights for the hidden layer
            self.weights.append(weights)

        # create the weights for the output layer
        for i in range(self.num_outputs):
            # create a list of weights for each node in the output layer
            weights = []
            for j in range(self.num_hidden):
                # create a random weight between -1 and 1
                weights.append(rand.uniform(-1, 1))
            # add the list of weights to the list of weights for the output layer
            self.weights.append(weights)

        # create the biases for the hidden layer
        for i in range(self.num_hidden):
            # create a random bias between -1 and 1
            self.biases.append(rand.uniform(-1, 1))

        # create the biases for the output layer
        for i in range(self.num_outputs):
            # create a random bias between -1 and 1
            self.biases.append(rand.uniform(-1, 1))

    # define the sigmoid function
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    # define the sigmoid derivative function
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # define the feed forward function
    def feed_forward(self, inputs):
        # create a list of the outputs for each node in the hidden layer
        self.hidden_outputs = []
        # create a list of the outputs for each node in the output layer
        self.output_outputs = []

        # calculate the output for each node in the hidden layer
        for i in range(self.num_hidden):
            # create a variable to hold the sum of the inputs multiplied by the weights
            sum = 0
            # calculate the sum of the inputs multiplied by the weights
            for j in range(self.num_inputs):
                sum += inputs[j] * self.weights[i][j]
            # add the bias to the sum
            sum += self.biases[i]
            # calculate the output for the node
            output = self.sigmoid(sum)

            # add the output to the list of outputs for the hidden layer
            self.hidden_outputs.append(output)

        # calculate the output for each node in the output layer
        for i in range(self.num_outputs):
            # create a variable to hold the sum of the inputs multiplied by the weights
            sum = 0
            # calculate the sum of the inputs multiplied by the weights
            for j in range(self.num_hidden):
                sum += self.hidden_outputs[j] * self.weights[i + self.num_hidden][j]
            # add the bias to the sum
            sum += self.biases[i + self.num_hidden]
            # calculate the output for the node
            output = self.sigmoid(sum)
            # add the output to the list of outputs for the output layer
            self.output_outputs.append(output)

        # return the output for the output layer
        return self.output_outputs

    # define the backpropagation function
    def backpropagation(self, inputs, targets):
        # create a list of the errors for each node in the output layer
        self.output_errors = []
        # create a list of the errors for each node in the hidden layer
        self.hidden_errors = []

        # calculate the error for each node in the output layer
        for i in range(self.num_outputs):
            # calculate the error for the node
            error = targets - self.output_outputs[i] # float object is not subscriptable error here - need to convert to float
            # add the error to the list of errors for the output layer
            self.output_errors.append(error)

        # calculate the error for each node in the hidden layer
        for i in range(self.num_hidden):
            # create a variable to hold the sum of the errors multiplied by the weights
            sum = 0
            # calculate the sum of the errors multiplied by the weights
            for j in range(self.num_outputs):
                sum += self.output_errors[j] * self.weights[j + self.num_hidden][i]
            # calculate the error for the node
            error = sum
            # add the error to the list of errors for the hidden layer
            self.hidden_errors.append(error)

        # update the weights and biases for the output layer
        for i in range(self.num_outputs):
            # calculate the change in weights for the node
            delta_weights = self.output_errors[i] * self.sigmoid_derivative(self.output_outputs[i])
            # update the weights for the node
            for j in range(self.num_hidden):
                self.weights[i + self.num_hidden][j] += delta_weights * self.hidden_outputs[j]
            # update the bias for the node
            self.biases[i + self.num_hidden] += delta_weights

        # update the weights and biases for the hidden layer
        for i in range(self.num_hidden):
            # calculate the change in weights for the node
            delta_weights = self.hidden_errors[i] * self.sigmoid_derivative(self.hidden_outputs[i])
            # update the weights for the node
            for j in range(self.num_inputs):
                self.weights[i][j] += delta_weights * inputs[j]
            # update the bias for the node
            self.biases[i] += delta_weights

    # define the train function
    def train(self, inputs, targets, epochs):
        # train the neural network for the specified number of epochs
        for i in range(epochs):
            # train the neural network on each input
            for j in range(len(inputs)):
                # feed forward the input
                self.feed_forward(inputs[j])
                # backpropagate the error
                self.backpropagation(inputs[j], targets[j])
                # print the error for the epoch
                print("Epoch: {} Error: {}".format(i, round(float(self.output_errors[0]), 4)))

    # define the predict function
    def predict(self, inputs):
        # feed forward the inputs
        return self.feed_forward(inputs)
    
    # define the print_weights function
    def print_weights(self):
        # print the weights for the hidden layer
        print("Hidden Layer Weights:")
        for i in range(self.num_hidden):
            print(self.weights[i])
        # print the weights for the output layer
        print("Output Layer Weights:")
        for i in range(self.num_outputs):
            print(self.weights[i + self.num_hidden])

    # define the print_biases function
    def print_biases(self):
        # print the biases for the hidden layer
        print("Hidden Layer Biases:")
        for i in range(self.num_hidden):
            print(self.biases[i])
        # print the biases for the output layer
        print("Output Layer Biases:")
        for i in range(self.num_outputs):
            print(self.biases[i + self.num_hidden])

    # define the print_outputs function
    def print_outputs(self):
        # print the outputs for the hidden layer
        print("Hidden Layer Outputs:")
        for i in range(self.num_hidden):
            try:
                print(self.hidden_outputs[i])
            except:
                print("No Hidden Layer Outputs - Model Was Loaded")
        # print the outputs for the output layer
        print("Output Layer Outputs:")
        for i in range(self.num_outputs):
            try:    
                print(self.output_outputs[i])
            except:
                print("No Output Layer Outputs - Model Was Loaded")

    # define the print_errors function
    def print_errors(self):
        # print the errors for the hidden layer
        print("Hidden Layer Errors:")
        for i in range(self.num_hidden):
            try:
                print(self.hidden_errors[i])
            except:
                print("No Hidden Layer Errors - Model Was Loaded")
        # print the errors for the output layer
        print("Output Layer Errors:")
        for i in range(self.num_outputs):
            try:
                print(self.output_errors[i])
            except:
                print("No Output Layer Errors - Model Was Loaded")

    # define the save_weights function
    def save_weights(self):
        # create a directory to store the weights if it doesn't exist
        if not os.path.exists("weights"):
            os.makedirs("weights")

        # save the weights and biases to the weights.txt file
        with open("weights/weights.txt", "w") as file:
            for layer in self.weights:
                file.write(" ".join(str(weight) for weight in layer))
                file.write("\n")
            file.write(" ".join(str(bias) for bias in self.biases))

    # define the load_weights function
    def load_weights(self):
        # check if the weights.txt file exists
        if os.path.isfile("weights/weights.txt"):
            with open("weights/weights.txt", "r") as file:
                lines = file.readlines()

            # load the weights and biases from the file
            for i in range(self.num_hidden):
                self.weights[i] = [float(weight) for weight in lines[i].split()]
            for i in range(self.num_outputs):
                self.weights[i + self.num_hidden] = [float(weight) for weight in lines[i + self.num_hidden].split()]

            self.biases = [float(bias) for bias in lines[-1].split()]


# create a neural network with 2 inputs, 3 hidden nodes, and 1 output
nn = NeuralNetwork(2, 3, 1)

# create lists to hold the inputs and targets
inputs = []
targets = []

# append the inputs and targets
inputs.append([0.0, 0.0])
inputs.append([0.0, 1.0])
inputs.append([1.0, 0.0])
inputs.append([1.0, 1.0])

# OR gate targets
# targets.append(0)
# targets.append(1)
# targets.append(1)
# targets.append(1)

# AND gate targets
# targets.append(0)
# targets.append(0)
# targets.append(0)
# targets.append(1)

# XOR gate targets
targets.append(0)
targets.append(1)
targets.append(1)
targets.append(0)

# NOT gate targets
# targets.append(1)
# targets.append(0)
# targets.append(0)
# targets.append(0)

# load the weights and biases if the weights.txt file exists; otherwise, train the neural network
if os.path.isfile("weights/weights.txt"):
    nn.load_weights()
else:
    nn.train(inputs, targets, 1000)

# print the weights
nn.print_weights()

# print the biases
nn.print_biases()

# print the outputs
nn.print_outputs()

# print the errors
nn.print_errors()

# feed it data to see what it predicts
new_input = [0, 0]
new_output = nn.predict(new_input)
print("New Input:[0, 0] New Output:[{}]".format(
    round(float(new_output[0]), 4)
))

new_input = [0, 1]
new_output = nn.predict(new_input)
print("New Input:[0, 1] New Output:[{}]".format(
    round(float(new_output[0]), 4)
))

new_input = [1, 0]
new_output = nn.predict(new_input)
print("New Input:[1, 0] New Output:[{}]".format(
    round(float(new_output[0]), 4)
))

new_input = [1, 1]
new_output = nn.predict(new_input)
print("New Input:[1, 1] New Output:[{}]".format(
    round(float(new_output[0]), 4)
))

# save the weights and biases
nn.save_weights()



