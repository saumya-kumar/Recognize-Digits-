import numpy as np

# Activation functions
identity = lambda x: x
identity.d = lambda x: np.eye(x.size)

relu = np.vectorize(lambda x: x if x >= 0.0 else 0.0)
relu.d = lambda x: np.diag((x >= 0.0).astype('float64'))

def softmax(x):
    x = np.exp(x)
    x /= x.sum()
    return x

def softmax_d(x):
    x = softmax(x)
    res = -np.tensordot(x, x, axes=0)
    for i in range(x.size):
        res[i][i] = x[i] * (1.0 - x[i])
    return res

softmax.d = softmax_d

# Loss functions
categorical_crossentropy = lambda y_true, y_pred: -(y_true * np.log(y_pred)).sum()
categorical_crossentropy.d = lambda y_true, y_pred: -(y_true / y_pred)

mse = lambda y, y_pred: ((y_pred - y) ** 2).sum()
mse.d = lambda y, y_pred: 2.0*(y_pred - y)

class Layer:
    """
    Instance variables:
    inputs -- number of input neurons
    activation -- the activation function
    z -- values of the neurons before applying the activation function
    a -- values of the neurons (after applying the activation function)
    w -- weight matrix of this layer
    b -- bias vector of this layer
    dz -- vector to store the derivative of loss w.r.t. z
    da -- vector to store the derivative of loss w.r.t. a
    dw -- matrix to store the derivative of loss w.r.t. w
    db -- vector to store the derivative of loss w.r.t. b
    """

    def __init__(self, units, activation=identity, inputs=None):
        self.units = units
        self.activation = activation
        self.inputs = inputs

    def feedforward(self, x):
        self.z = self.w @ x + self.b
        self.a = self.activation(self.z)
        return self.a

class NeuralNetwork:
    """
    Instance Variables:
    layers -- list of Layer objects in this network
    loss -- the loss function to evaluate this network
    """

    def __init__(self, layers, loss):
        self.layers = layers
        self.loss = loss
        for i in range(1, len(layers)):
            layers[i].inputs = layers[i-1].units
        for layer in layers:
            limit = np.sqrt(6 / (layer.units + layer.inputs))
            layer.w = 2 * limit * np.random.random_sample((layer.units, layer.inputs)) - limit
            layer.b = np.zeros(layer.units,)

    def predict(self, x):
        y = x
        for layer in self.layers:
            y = layer.feedforward(y)
        return y

    def evaluate(self, val_x, val_y):
        """
        Returns average loss and accuracy over val_x, val_y
        """
        loss_sum = 0.0
        num_correct = 0
        for x, y in zip(val_x, val_y):
            y_pred = self.predict(x)
            loss_sum += self.loss(y, y_pred)
            num_correct += y[y_pred.argmax()] == 1.0
        samples = val_x.shape[0]
        return loss_sum / samples, num_correct / samples

    def train_iteration(self, x, y, learning_rate):
        for layer in self.layers:
            layer.db = np.zeros(layer.b.shape)
            layer.dw = np.zeros(layer.w.shape)
        for xi, yi in zip(x, y):
            y_pred = self.predict(xi)
            last_layer = self.layers[-1]
            last_layer.da = self.loss.d(yi, y_pred)
            last_layer.dz = last_layer.activation.d(last_layer.z) @ last_layer.da
            for i in range(len(self.layers) - 2, -1, -1):
                layer = self.layers[i]
                next_layer = self.layers[i+1]
                layer.da = next_layer.dz @ next_layer.w
                layer.dz = layer.activation.d(layer.z) @ layer.da
            prev_a = xi
            for layer in self.layers:
                layer.db += layer.dz
                layer.dw += np.tensordot(layer.dz, prev_a, 0)
                prev_a = layer.a
        samples = x.shape[0]
        for layer in self.layers:
            layer.db /= samples
            layer.dw /= samples
            layer.b -= layer.db * learning_rate
            layer.w -= layer.dw * learning_rate

    def train(self, x, y, validation_split=0.0, learning_rate=0.01, batch_size=32, epochs=5):
        #Split into training set and validation set
        num_samples = x.shape[0]
        split = int(num_samples * validation_split)
        val_x = x[ : split]
        val_y = y[ : split]
        train_x = x[split : ]
        train_y = y[split : ]
        num_samples = train_x.shape[0]
        history = {"loss": [], "acc": []}
        if val_x.size:
            history["val_loss"] = []
            history["val_acc"] = []
        for i in range(epochs):
            samples = train_x.shape[0]
            perm = np.random.permutation(samples)
            train_x = train_x[perm]
            train_y = train_y[perm]
            for r in range(batch_size, num_samples+1, batch_size):
                l = r - batch_size
                self.train_iteration(train_x[l : r], train_y[l : r], learning_rate)
            train_loss, train_acc = self.evaluate(train_x, train_y)
            history["loss"].append(train_loss)
            history["acc"].append(train_acc)
            print(f"Epoch # {i+1}- loss: {train_loss:.5f}, acc: {train_acc:.3%}", end="")
            if val_x.size:
                val_loss, val_acc = self.evaluate(val_x, val_y)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                print(f", val_loss: {val_loss:.3f}, val_acc: {val_acc:.3%}", end="")
            print()
        return history
