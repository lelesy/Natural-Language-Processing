import numpy as np
import sys

class PartyNN(object):
    def __init__(self, learning_rate=0.01):
        self.weights_0_1 = np.random.normal(0.0, 2 ** -0.5, (3, 4))
        self.weights_1_2 = np.random.normal(0.0, 1, (1, 3))
        self.sigmoid_mapper = np.vectorize(self.sigmoid)
        self.learning_rate = np.array([learning_rate])
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, inputs):
        inputs_1 = np.dot(self.weights_0_1, inputs)
        outputs_1 = self.sigmoid_mapper(inputs_1)

        inputs_2 = np.dot(self.weights_1_2, outputs_1)
        outputs_2 = self.sigmoid_mapper(inputs_2)


    def train(self, inputs, expected_predict):
        inputs_1 = np.dot(self.weights_0_1, inputs)
        outputs_1 = self.sigmoid_mapper(inputs_1)

        inputs_2 = np.dot(self.weights_1_2, outputs_1)
        outputs_2 = self.sigmoid_mapper(inputs_2)
        actual_predict = outputs_2[0]

        error_layer_2 = np.array([actual_predict - expected_predict])
        gradient_layer_2 = actual_predict * (1 - actual_predict)
        weights_delta_layer_2 = error_layer_2 * gradient_layer_2
        self.weights_1_2 -= (np.dot(weights_delta_layer_2, outputs_1.reshape(1, len(outputs_1)))) * self.learning_rate

        error_layer_1 = weights_delta_layer_2 * self.weights_1_2
        gradient_layer_1 = outputs_1 * (1 - outputs_1)
        weights_delta_layer_1 = error_layer_1 * gradient_layer_1
        self.weights_0_1 -= np.dot(inputs.reshape(len(inputs), 1), weights_delta_layer_1).T  * self.learning_rate

def MSE(y, Y):
    return np.mean((y-Y)**2)
train = [
        ([1, 1, 1, 1], 1),
        ([1, 1, 0, 0], 0),
        ([1, 0, 1, 1], 0),
        ([0, 0, 0, 1], 1),
        ([0, 1, 1, 1], 1),
        ([0, 0, 0, 0], 0),
    ]


# epochs = 5000
# learning_rate = 0.05
epochs = 3000
learning_rate = 0.1

network = PartyNN(learning_rate=learning_rate)

for e in range(epochs):
    inputs = []
    correct_predictions = []
    for input_stat, correct_predict in train:
        network.train(np.array(input_stat), correct_predict)
        inputs.append(np.array(input_stat))
        correct_predictions.append(np.array(correct_predict))

    # print network.predict(np.array(inputs).T
    # train_loss = MSE(network.predict(np.array(inputs).T), np.array(correct_predictions))
    # print "Progress: "+(str(100 * e/float(epochs))[:4])
    # print "Training loss: "+ str(train_loss)[:5]
    # sys.stdout.write("\\rProgress: {}, Training loss: {}\".format(str(100 * e/float(epochs))[:4], str(train_loss)[:5]))


for input_stat, correct_predict in train:
        print("For input: {} the prediction is: {}, expected: {}".format(
            str(input_stat),
            str(network.predict(np.array(input_stat)) > .5),
            str(correct_predict == 1)))
print network.weights_0_1
print network.weights_1_2
