import numpy as np
from keras.datasets import mnist
from matplotlib import pyplot

#Load in the data set into training and test variables
(train_X, train_Y), (test_X, test_Y) = mnist.load_data()

#Convert the data sets into numpy arrays
train_X = train_X.astype(np.float32) / 255
train_Y = np.array(train_Y)
test_X = test_X.astype(np.float32) / 255
test_Y = np.array(test_Y)

#print(train_X.shape)
#print(train_Y.shape)

'''for i in range(9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
    pyplot.show()'''

#Save the labels as a boolean vector of size 10 because we have 10 classes
def one_hot_encode(labels, numClasses = 10):
    return np.eye(numClasses)[labels]

train_labels_one_hot = one_hot_encode(train_Y)
test_labels_one_hot = one_hot_encode(test_Y)

#print(train_labels_one_hot)
#print(test_labels_one_hot)

class ConjugateGradientDescent:
    def __init__(self, epsilon=1e-5):
        self.epsilon = epsilon
        self.prev_gradient = None
        self.prev_direction = None

    def initialize(self, params):
        self.prev_gradient = {}
        self.prev_direction = {}
        for key, value in params.items():
            self.prev_gradient[key] = np.zeros_like(value)
            self.prev_direction[key] = np.zeros_like(value)

    def update(self, gradients, params, data):
        if self.prev_gradient is None:
            self.initialize(params)

        for key in params.keys():
            if np.linalg.norm(self.prev_gradient[key]) < self.epsilon:
                direction = -gradients[key]
            else:
                beta = np.dot(gradients[key].ravel(), gradients[key].ravel()) / np.dot(self.prev_gradient[key].ravel(), self.prev_gradient[key].ravel())
                direction = -gradients[key] + beta * self.prev_gradient[key]
                #print(gradients[key], end=" ")
                #print(direction)

            step_size = self.line_search(params, key, direction, gradients[key], data)
            #params[key] += step_size * direction.reshape(params[key].shape)
            params[key] -= gradients[key].reshape(params[key].shape)

            self.prev_gradient[key] = gradients[key]
            self.prev_direction[key] = direction

        print(self.compute_loss(params, data))
        return params

    def compute_loss(self, params, data):
        # This function should compute the loss given the model parameters and data
        model.set_params(params)
        output = model.forward(data['inputs'])
        loss = model.compute_loss(data['labels'], output)
        return loss

    def line_search(self, params, key, direction, gradient, data):
        step_size = 1.0
        alpha = 0.5
        beta = 0.8

        curr_loss = self.compute_loss(params, data)
        max_iterations = 100
        curr_iterations = 0
        #print(curr_loss)
        while curr_iterations < max_iterations:
            params_temp = params.copy()
            new_param = params[key] + step_size * direction.reshape(params[key].shape)
            params_temp[key] = new_param
            new_loss = self.compute_loss(params_temp, data)
            comp = curr_loss + alpha * step_size * np.dot(gradient.ravel(), direction.ravel())
            #print(new_loss, end = " ")
            #print(comp, end = " ")
            #print(np.dot(gradient.ravel(), direction.ravel()))
            if new_loss < curr_loss + alpha * step_size * np.dot(gradient.ravel(), direction.ravel()):
                break
            step_size *= beta
            curr_iterations+=1
        return step_size

#Basic neural network class
class NeuralNet:
    def __init__(self):
        #Randomize the weights and biases for each layer so that each layer does not learn
        #at the same rate
        self.fc1_weights = np.random.randn(784, 128) * 0.01
        self.fc1_bias = np.zeros((1, 128))
        self.fc2_weights = np.random.randn(128, 64) * 0.01
        self.fc2_bias = np.zeros((1, 64))
        self.fc3_weights = np.random.randn(64, 10) * 0.01
        self.fc3_bias = np.zeros((1, 10))

    def getParams(self):
        params = {
            'fc1_weights': self.fc1_weights,
            'fc1_bias': self.fc1_bias,
            'fc2_weights': self.fc2_weights,
            'fc2_bias': self.fc2_bias,
            'fc3_weights': self.fc3_weights,
            'fc3_bias': self.fc3_bias
        }
        return params

    #Forward propagation through the neural network
    def forward(self, x):
        #print('c')
        self.x = x
        x = x.reshape(x.shape[0], -1)
        self.z1 = np.dot(x, self.fc1_weights) + self.fc1_bias
        self.a1 = self.relu(self.z1)
        #print(self.a1)

        self.z2 = np.dot(self.a1, self.fc2_weights) + self.fc2_bias
        self.a2 = self.relu(self.z2)

        self.s3 = np.dot(self.a2, self.fc3_weights) + self.fc3_bias

        #Softmax at the end to determine the most probable outcome
        self.a3 = self.softmax(self.s3)
        #print(self.a3)
        return self.a3

    #Backwards propagation
    def backward(self, y):
        #print('d')
        m = y.shape[0]

        #Computing the gradient for the output layer
        dz3 = self.a3 - y
        #print(dz3, end="")
        self.fc3_weights_grad = np.dot(self.a2.T, dz3) / m
        #print(self.a2.T, end="")
        self.fc3_bias_grad = np.sum(dz3, axis = 0, keepdims = True) / m

        #Computing the gradient for the second layer
        da2 = np.dot(dz3, self.fc3_weights.T)
        dz2 = da2 * self.relu_derivative(self.z2)
        self.fc2_weights_grad = np.dot(self.a1.T, dz2) / m
        self.fc2_bias_grad = np.sum(dz2, axis = 0, keepdims = True) / m

        #Computing the gradient for the first layer
        da1 = np.dot(dz2, self.fc2_weights.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        self.fc1_weights_grad = np.dot(self.x.T, dz1) / m
        self.fc1_bias_grad = np.sum(dz1, axis = 0, keepdims = True) / m

        clip_value = 1.0
        self.fc3_weights_grad = np.clip(self.fc3_weights_grad, -clip_value, clip_value)
        self.fc3_bias_grad = np.clip(self.fc3_bias_grad, -clip_value, clip_value)
        self.fc2_weights_grad = np.clip(self.fc2_weights_grad, -clip_value, clip_value)
        self.fc2_bias_grad = np.clip(self.fc2_bias_grad, -clip_value, clip_value)
        self.fc1_weights_grad = np.clip(self.fc1_weights_grad, -clip_value, clip_value)
        self.fc1_bias_grad = np.clip(self.fc1_bias_grad, -clip_value, clip_value)

        gradients = {
            'fc1_weights': self.fc1_weights_grad,
            'fc1_bias': self.fc1_bias_grad,
            'fc2_weights': self.fc2_weights_grad,
            'fc2_bias': self.fc2_bias_grad,
            'fc3_weights': self.fc3_weights_grad,
            'fc3_bias': self.fc3_bias_grad,
        }
        return gradients

    #Method for computing the ReLU function
    def relu(self, z):
        return np.maximum(0, z);

    #Method for computing the ReLU function for derivatives
    def relu_derivative(self, z):
        return (z > 0).astype(float)

    #Method for computing softmax
    def softmax(self, z):
        eZ = np.exp(z - np.max(z, axis = 1, keepdims = True))
        return eZ / np.sum(eZ, axis = 1, keepdims = True)

    def compute_loss(self, y, y_hat):
        m = y.shape[0]
        return -np.sum(y * np.log(y_hat + 1e-9)) / m

    def set_params(self, params):
        self.fc1_weights = params['fc1_weights']
        self.fc1_bias = params['fc1_bias']
        self.fc2_weights = params['fc2_weights']
        self.fc2_bias = params['fc2_bias']
        self.fc3_weights = params['fc3_weights']
        self.fc3_bias = params['fc3_bias']

    def step(self, gradients, optimizer, data):
        #print('e')
        params = {
            'fc1_weights': self.fc1_weights,
            'fc1_bias': self.fc1_bias,
            'fc2_weights': self.fc2_weights,
            'fc2_bias': self.fc2_bias,
            'fc3_weights': self.fc3_weights,
            'fc3_bias': self.fc3_bias
        }

        params = optimizer.update(gradients, params, data)
        #print('f')

        self.set_params(params)

#Class for gradient descent
class GradientDescent:
    #Initialize the model for gradient descent
    def __init__(self, model, lr = 0.0025):
        self.model = model
        self.lr = lr

    #Method for each step in successive subspace correction
    def step(self, multiplier, subspace):
        #Update a certain subspace using the learning rate and weights gradient
        if(subspace == 'fc1'):
            self.model.fc1_weights_grad = self.model.fc1_weights_grad.reshape(784, 128)
            self.model.fc1_weights -= self.lr * self.model.fc1_weights_grad
            self.model.fc1_bias -= self.lr * self.model.fc1_bias_grad
        elif(subspace == 'fc2'):
            self.model.fc2_weights -= self.lr * self.model.fc2_weights_grad
            self.model.fc2_bias -= self.lr * self.model.fc2_bias_grad
        elif(subspace == 'fc3'):
            self.model.fc3_weights -= self.lr * self.model.fc3_weights_grad
            self.model.fc3_bias -= self.lr * self.model.fc3_bias_grad

def train_ssc(model, train_X, train_Y, epochs=1500, batch_size = 100):
    num_samples = train_X.shape[0]

    #Initialize variables of GradientDescent for each layer of the Neural Network
    optimizer_fc1 = GradientDescent(model)
    optimizer_fc2 = GradientDescent(model)
    optimizer_fc3 = GradientDescent(model)

    kP = 1 / 2.3
    multiplier = 1
    error = 0

    for epoch in range(epochs):
        permutation = np.random.permutation(num_samples)
        train_images_shuffled = train_X[permutation]
        train_labels_shuffled = train_Y[permutation]

        for i in range(0, num_samples, batch_size):
            batch_images = train_images_shuffled[i:i+batch_size]
            batch_labels = train_labels_shuffled[i:i+batch_size]

            output = model.forward(batch_images)

            loss = model.compute_loss(batch_labels, output)

            model.backward(batch_labels)

            optimizer_fc1.step(multiplier, 'fc1')
            optimizer_fc2.step(multiplier, 'fc2')
            optimizer_fc3.step(multiplier, 'fc3')

        error = loss - 0.01
        multiplier = kP * error
        optimizer_fc1.lr = 0.0025 * multiplier
        print(f'Epoch {epoch + 1}, Loss: {loss:.4f}, LR: {optimizer_fc1.lr}, mult: {multiplier}')

def train_ssc_cgd(model, train_X, train_Y, num_epochs = 100, epsilon = 1e-5, batch_size=100):
    num_samples = train_X.shape[0]
    optimizer = ConjugateGradientDescent(epsilon)
    params = model.getParams()
    optimizer.initialize(params)

    for epoch in range(num_epochs):
        permutation = np.random.permutation(num_samples)
        train_images_shuffled = train_X[permutation]
        train_labels_shuffled = train_Y[permutation]

        for i in range(0, num_samples, batch_size):
            #print('b')
            batch_images = train_images_shuffled[i:i + batch_size]
            batch_labels = train_labels_shuffled[i:i + batch_size]

            output = model.forward(batch_images)

            loss = model.compute_loss(batch_labels, output)
            #print(loss)

            gradients = model.backward(batch_labels)

            data = {'inputs': batch_images, 'labels': batch_labels}
            params = optimizer.update(gradients, params, data)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}")

def test_model(model, test_X, test_Y):
    outputs = model.forward(test_X)
    predictions = np.argmax(outputs, axis = 1)
    accuracy = np.mean(predictions == test_Y)
    print(f'Accuracy: {accuracy * 100:.2f}%')

model = NeuralNet()
train_ssc_cgd(model, train_X, train_labels_one_hot)
test_model(model, test_X, test_Y)