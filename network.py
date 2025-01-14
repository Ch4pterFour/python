import numpy as np
import helpers
import pandas as pd
from sklearn.metrics import confusion_matrix

#  #  #  Tasks  #  #  #
#  Split predict function into train and test functions


# Error messages

class CustomError(Exception):
    def __init__(self, message="An error occurred"):
        self.message = message
        super().__init__(self.message)

class FormatError(CustomError):
    def __init__(self, message="An error occurred related to format of the provided input"):
        self.message = message
        super().__init__(self.message)

class NotATuple(FormatError):
    def __init__(self, message="Predictors need to be supplied as tuple"):
        super().__init__(message)

class UnequalVariables(CustomError):
    def __init__(self, error_type=None):
        if error_type == "nobs_unequal":
            message = "Network observations are unequal to data length"
        elif error_type == "in_unequal":
            message = "Number of predictors is not equal to number of input neurons"
        elif error_type == "out_unequal":
            message = "Number of outcome categories is not equal to number of output neurons"
        else:
            message = "Some variables are not equal when they should be"
        super().__init__(message)

class BatchSize(CustomError):
    def __init__(self, error_type=None):
        if error_type == "no_size":
            message = "The training set has more than 500 observations, specify the batch size using 'batch_size'"
        elif error_type == "unequal_sizes":
            message = "The batch size you provided does not result in equal batches, make sure that dividing the train \
                      set by the batch size results in an integer"
        elif error_type == "zero_size":
            message = "Batch size cannot be 0"
        super().__init__(message)


class Network:
    def __init__(self, structure: tuple, categorical=False, function="ReLu", rate=0.01, tolerance=0.0001, MAXEpoch=5000,
                 x=1, y=2, batch_size=1):
        """
        Initialize the neural network.

        :param structure: Tuple representing the number of neurons in each layer.
        :param categorical: Whether the task is classification.
        :param function: Activation function used in the network. Options: "ReLu", "Sigmoid" or "SoftPlus".
        :param rate: Learning rate for gradient descent.
        :param tolerance: Tolerance level to stop training.
        :param MAXEpoch: Maximum number of epochs for training.
        :param x: Index for feature column.
        :param y: Index for label column.
        """
        self.structure = structure
        self.categorical = categorical
        self.function = function
        self.rate = rate
        self.tolerance = tolerance
        self.MAXEpoch = MAXEpoch
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.layers = []
        self.iterator_state = 0
        self.out = False

        # Create layers
        self.create_layers()

    def create_layers(self):
        """
        Create layers for the neural network based on the structure.
        """
        num_layers = len(self.structure)

        for i in range(num_layers):
            if i == 0:
                # Input layer
                layer = InputLayer(self, i, self.structure[i], 0)
            elif i == num_layers - 1:
                # Output layer
                layer = OutputLayer(self, i, self.structure[i], self.structure[i - 1])
            else:
                # Hidden layers
                layer = Layer(self, i, self.structure[i], self.structure[i - 1])

            self.layers.append(layer)

    # Generic function to get the structure of the network
    def get_structure(self):
        for layer in self.layers:
            if layer.layer_n == 0:
                print("Input Layer")
                layer.get_params(show=True)
            elif(layer.layer_n == len(self.structure)-1):
                print("Output Layer")
                layer.get_params(show=True)
            else:
                print(f"Layer", layer.layer_n)
                layer.get_params(show=True)

    # Function to manually set the weights and biases of the network instead of random initialization
    def set_w_b(self, nweights, nbiases):
        """
        Manually set the initial weights and biases of the network.

        :param nweights: List or numpy array representing the weights, rows represent neurons in the current
        layer, columns represent the number of neurons in the previous layer.
        :param nbiases: List or numpy array representing the biases
        """
        # Check for the correct format of weights and biases input
        if not all(isinstance(w, (list, np.ndarray)) for w in nweights):
            raise ValueError("All elements of 'nweights' must be lists or arrays")
        if not all(isinstance(b, (list, np.ndarray)) for b in nbiases):
            raise ValueError("All elements of 'nbiases' must be lists or arrays")

        # Check that the lengths of nweights and nbiases match the number of layers minus one
        # (excluding the input layer)
        if len(nweights) != len(self.layers) - 1:
            raise ValueError("Length of 'nweights' does not match number of layers excluding input layer")
        if len(nbiases) != len(self.layers) - 1:
            raise ValueError("Length of 'nbiases' does not match number of layers excluding input layer")

        c = 0  # Counter to keep track of values
        for i, layer in enumerate(self.layers[1:]):
            print(f"Layer: {i+1}")
            layer.set_params(nweights[c], nbiases[c])
            c += 1

    # Predict outcome values based on the input of predictor(s), using matrix-vector multiplication (calc_pred())
    def forward_propagate(self, observations, y=1, show=False):
        cat = self.categorical # Check whether it is a classifcation task
        # Define the y arrays
        y_hat = np.zeros((len(observations), len(np.unique(y)))) if cat is True else np.zeros((len(observations), 1))
        cross = np.zeros(len(observations))
        keys = ["Predictions", "Residuals", "Error"]
        # Iterate through each observation
        for i, observation in enumerate(observations):
            #print(f"Observation {i}")
            self.iterator_state = i  # Keep track of the observation number at network level
            prev_a = observation  # Initialize the input with the current observation
            for layer in self.layers:
                if isinstance(layer, InputLayer):
                    #print("Input Layer")
                    layer.set_params(a=prev_a)
                    params = layer.get_params()
                    prev_a = params["A"][i, :] if params["A"].ndim > 1 else params["A"][i]
                else:
                    #print("Normal layer")
                    params = layer.calc_pred(prev_a, layer.get_params()["Weights"], layer.get_params()["Biases"])
                    # For the last layer of neurons we do not use an activation function on the predictions
                    prev_a = (
                        params["Z"][i, :] if params["Z"].ndim > 1 else params["Z"][i]
                    ) if isinstance(layer, OutputLayer) else (
                        params["A"][i, :] if params["A"].ndim > 1 else params["A"][i]
                    )
                    if isinstance(layer, OutputLayer):
                        #print("Output layer")
                        y_hat[i] = helpers.SoftMax(prev_a) if cat is True else prev_a
                        # Calculate cross-entropy by taking the -log of the predicted probability of the actual class
                        cross[i] = -np.log(y_hat[i, int(y[i])] + helpers.epsilon)
        self.iterator_state = 0  # Reset iterator state
        residual = y - y_hat if cat is False else None
        error = np.square(residual) if cat is False else cross
        results = {key: value for key, value in zip(keys, [y_hat, residual, np.sum(error)])}
        if show is True:  # Useful for manually calling this function
            for key, value in results.items():
                print(f"{key}: {value}\n")
        else:
            return results

    # Go backward through the network and use the chain rule to calculate derivatives
    def backward_propagate(self, observations, y, preds):
        # Add an extra iteration to update parameters by appending a row of NaNs
        observations = np.vstack([observations, np.nan * np.ones((1, observations.shape[1]))])
        step_sizes = np.zeros(len(self.structure) - 1)
        batch = len(observations) - 1
        # Loop through the observations
        for i, observation in enumerate(observations):
            #print(f"Observation {i}")
            self.iterator_state = i

            for l in range(len(self.layers) - 1, -1, -1):
                layer = self.layers[l]
                prev_layer = self.layers[l - 1] if l > 0 else None

                if isinstance(layer, InputLayer):
                    #print("Input layer")
                    continue

                self.out = isinstance(layer, OutputLayer)
                #print("Output layer" if self.out else "Normal layer")
                # For the last observation we want to get the max step size for that layer
                if i == batch:
                    step_sizes[l - 1] = layer.chain_rule(preds, y, prev_layer)
                else:
                    layer.chain_rule(preds, y, prev_layer)
        self.iterator_state = 0
        return np.max(np.abs(step_sizes))

    # Function that implements the gradient descent algorithm
    def gradient_descent(self, train, show=True):
        max_step, epochs = 1, 1
        error = None
        # Set the shape of the neuron arrays based on the number of observations in the batch
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.reshape(self.batch_size)
        # While loop to keep the algorithm going
        while max_step > self.tolerance and epochs < self.MAXEpoch:
            if show is True: print(f"Epochs: {epochs}, error: {error}")
            b = 1
            for batch in helpers.create_shuffled_batches(train, self.batch_size):
                # Process each individual observation within the batch
                if show is True: print(f"Batch: {b}, error: {error}")
                forward = self.forward_propagate(batch[:, self.x], y=batch[:, self.y], show=False)
                error = forward["Error"]
                backward = self.backward_propagate(batch[:, self.x], y=batch[:, self.y], preds=forward["Predictions"])
                max_step = backward # Evaluation goes wrong

                # Print the batch info with the current max_step
                if show is True: print(f"Batch: {b}, error: {error}, max_step: {max_step}")

                # Check if the stopping criterion is met within the batch loop

                if max_step <= self.tolerance:
                    if show is True: print("Stopping early as max_step is below the tolerance threshold.")
                    return
                b += 1
            epochs += 1


    # Train the network and predict values for a test dataset
    def predict(self, x: tuple, y, train, test, batch_size=None, show=False):
        """
        Train the network and predict values for a test set.

        :param x: Index for feature column. Multiple features are supplied in a tuple with length equal to number of
        features.
        :param y: Index for label column.
        :param train: Train dataset, a numpy array with rows equal observations and columns equal features
        :param test: Test dataset, a numpy array with rows equal observations and columns equal features
        :param batch_size: The size of the batches for the gradient descent algorithm, must be specified if train
        dataset has more than 500 observations
        """
        if len(x) != self.structure[0]:
            raise UnequalVariables(error_type="in_unequal")
        if self.categorical is True and len(np.unique(train[:, y])) != self.structure[len(self.structure) - 1]:
            raise UnequalVariables(error_type="out_unequal")
        # Check for batch size being None or zero
        if batch_size is None:
            if len(train) > 500:
                raise BatchSize(error_type="no_size")
        elif batch_size == 0:
            raise BatchSize(error_type="zero_size")
        # Ensure batch_size is not None or zero before performing modulus operation
        elif len(train) % batch_size != 0:
            raise BatchSize(error_type="unequal_sizes")
        self.x = x
        self.y = y
        if batch_size is None:
            self.batch_size = len(train)
        else:
            self.batch_size = batch_size
        self.gradient_descent(train, show=show)
        keys = ["Predictions", "Y_values", "Confusion", "Accuracy"]
        # Set the shape of the neuron arrays based on the number of observations in the test dataset
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.reshape(len(test))
        testing = self.forward_propagate(test[:, x], test[:, y])
        if show is True: print(f"Predictions: {testing['Predictions']}; actual values: {test[:, y]}")
        y_pred = testing["Predictions"] if self.categorical is False else np.argmax(testing["Predictions"], axis=1)
        y_true = test[:, y]
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        names = ("Setosa", "Versicolor", "Virginica")
        cm = pd.DataFrame(cm, index=[f"True {i}" for i in names],
                     columns = [f"Predicted {i}" for i in names])
        accuracy = (1 - np.mean(y_true - y_pred) ) if self.categorical is False else (1 - np.mean(y_true != y_pred) )
        # Print the confusion matrix and accuracy
        results = {key: value for key, value in zip(keys, [y_pred, y_true, cm, accuracy])}
        if show is True:
            for key, value in results.items():
                if key == "Confusion" or key == "Accuracy":
                    print(f"{key}: {value}\n")
        else:
            return results


class Layer:
    def __init__(self, network_instance, layer_n, n_neuron, prev_n):
        # Network instance the layer is part of in order to access network parameters
        self.network_instance = network_instance
        self.layer_n = layer_n  # Layer number
        self.neuron_n = n_neuron  # Number of neurons in layer
        self.prev_n = prev_n  # Number of neurons in the previous layer
        # Initialize layer with neurons
        self.neurons = [Neuron(self, n, prev_n) for n in range(1, (n_neuron + 1))]

    # Necessary function to allow matrix-vector multiplication during forward propagation
    def get_params(self, show=False):
        nobs = self.network_instance.batch_size
        #  Create a dictionary to store the parameters
        keys = ["Z", "A", "Weights", "Biases"]
        # Create lists to store keys information
        if self.layer_n != 0:
            # Create the arrays for the weights and biases
            weights = np.array([neuron.weights for neuron in self.neurons])
            biases = np.array([neuron.bias for neuron in self.neurons])
            # Reshape the arrays to the right format
            weights = weights.reshape(self.neuron_n, self.prev_n)
            biases = biases.reshape(self.neuron_n)
        else:
            weights, biases = None, None
        # Create the arrays for the predictions (z) and the activations (a)
        predictions = np.array([neuron.z for neuron in self.neurons])
        activations = np.array([neuron.a for neuron in self.neurons])
        # If there is only one neuron in the layer
        if self.neuron_n == 1:
            # Reshape the arrays to the right format based on batch size
            predictions, activations = predictions.reshape(nobs), activations.reshape(nobs)
        else:
            # Transposing the matrices is necessary to maintain correct dimensions
            # Z/A are always matrices with observations in the rows and neurons per layer in the columns
            #predictions, activations = predictions.reshape(self.neuron_n,nobs), activations.reshape(self.neuron_n, nobs)
            predictions, activations = predictions.transpose(), activations.transpose()
        # Put everything in a dictionary
        params = {key: value for key, value in zip(keys, [predictions, activations, weights, biases])}
        if show is True:
            for key, value in params.items():
                print(f"{key}: {value}\n")
        else:
            return params

    # Function to update the parameters in a layer format
    # It is only used by the network to update the a values
    # Otherwise this function is used when the user wants to set the weights/biases manually
    def set_params(self, weights=None, bias=None, a=None):
        for i, n in enumerate(self.neurons):
            if weights is not None and bias is not None:
                if weights.ndim == 1:
                    print(f"Weights for neuron {i} set to: {weights[i:]}")
                    print(f"Bias for neuron {i} set to: {bias[i]}")
                    n.update(weights=weights[i:], bias=bias[i])
                elif weights.ndim != 1:
                    print(f"Weights for neuron {i} set to: {weights[i, :]}")
                    print(f"Bias for neuron {i} set to: {bias[i]}")
                    n.update(weights=weights[i, :], bias=bias[i])
            else:
                if not isinstance(a, np.float64) or np.isscalar(a):
                    #print(f"Activation for neuron {i} set to: {a[i]}")
                    n.update(a=a[i], obs_ind=self.network_instance.iterator_state)  # multiple neurons
                else:
                    #print(f"Activation for neuron {i} set to: {a}")
                    n.update(a=a, obs_ind=self.network_instance.iterator_state)  # single neuron



    # Function to incorporate the matrix-vector multiplication during forward propagation
    def calc_pred(self, obs, weights, biases):
        z = np.dot(weights, obs) + biases  # Get matrix-vector product
        # Get the activation function to calculate a
        activation_function = getattr(helpers, self.network_instance.function)
        a = activation_function(z)
        # Update the z and a values
        for i, n in enumerate(self.neurons):
            if not isinstance(a, np.float64):
                n.update(z=z[i], a=a[i], obs_ind=self.network_instance.iterator_state) # multiple neurons
            else:
                n.update(z=z, a=a, obs_ind=self.network_instance.iterator_state) # single neuron
        return self.get_params()

    # Function to calculate the derivatives using the chain rule
    def chain_rule(self, y_hat, y, previous):
        # Previous is the previous layer object
        obs = self.network_instance.iterator_state
        # Create arrays to store step sizes for the parameters
        step_sizes_w, \
            step_sizes_b = np.zeros((self.prev_n, len(self.neurons))), np.zeros((self.prev_n, len(self.neurons)))
        # Loop through the neurons in the previous layer
        for n_p, prev_neuron in enumerate(previous.neurons):
            weights = np.array([])
            b_d = np.array(0)
            # Loop through the neurons in the current layer
            for n, neuron in enumerate(self.neurons):
                # Update parameters when obs is equal to the total number of obs + 1
                if obs == self.network_instance.batch_size:
                    #print(f"Last observation:", obs)
                    # Calculate new weights
                    step_sizes_w[n_p, n] = np.max(np.absolute((np.mean(neuron.w_d, axis=1) * self.network_instance.rate)))
                    n_w = neuron.weights - (np.mean(neuron.w_d, axis=1) * self.network_instance.rate)
                    # Calculate new bias
                    step_sizes_b[n_p, n] = (np.mean(neuron.b_d) * self.network_instance.rate)
                    n_b = neuron.bias - (np.mean(neuron.b_d) * self.network_instance.rate)
                    # Update the weights and bias
                    neuron.update(weights=n_w, bias=n_b)
                else:
                    # Chain rule derivative for the last layer
                    if self.network_instance.out is True:
                        neuron.b_d[obs] = -2*np.sum(y_hat[obs, :]-y[obs, ]) if self.network_instance.categorical \
                                                                               is False else helpers.CE_dx(n, y[obs,],
                                                                                                           y_hat[obs,])
                    # Chain rule derivative for other layers
                    else:
                        function_derivative = getattr(helpers, self.network_instance.function+"_dx")
                        neuron.b_d[obs] = function_derivative(neuron.z[obs]) * neuron.a_d[obs]
                    b_d = neuron.b_d[obs]
                    weights = np.append(weights, neuron.weights[n_p])
                    neuron.w_d[n_p, obs] = prev_neuron.a[obs] * neuron.b_d[obs]
            # Calculate the previous activation derivative
            if obs != self.network_instance.batch_size:
                prev_neuron.a_d[obs] = np.sum(weights * b_d) # Does not matter which b_d we pick, it is the same value
        # Get the maximum step size so this can be used by backpropagate() to determine the final max step size
        return np.max(np.absolute(np.concatenate((step_sizes_b, step_sizes_w))))

# Specific instances of a layer
class InputLayer(Layer):
    def __init__(self, network_instance, layer_n, n_neuron, prev_n):
        Layer.__init__(self, network_instance, layer_n, n_neuron, prev_n)
        self.neurons = [Neuron(self, n, prev_n, True, False) for n in range(1, (n_neuron + 1))]


class OutputLayer(Layer):
    def __init__(self, network_instance, layer_n, n_neuron, prev_n):
        Layer.__init__(self, network_instance, layer_n, n_neuron, prev_n)
        self.neurons = [Neuron(self, n, prev_n, False, True) for n in range(1, (n_neuron + 1))]


class Neuron:
    def __init__(self, layer_instance, neuron_n, prev_n, input_l=False, output_l=False):
        # a, z, a_d, b_d, w_d are None initially but are reshaped once the number of observations has been known
        self.layer_instance = layer_instance  # Layer reference
        self.neuron_n = neuron_n  # Unique neuron number
        self.prev_n = prev_n  # Get the number of neurons in previous layer
        self.first_l = input_l  # Whether the neuron is part of the input layer
        self.last_l = output_l  # Whether the neuron is part of the output layer
        self.z = None  # each neuron has a z value (activation before transformation)
        self.a, self.a_d = [None, None]
        if not input_l:
            # Initiate the weights, prev_n = the number of neurons
            self.weights, self.w_d = np.random.normal(0, 1, prev_n), None
            # in the previous layer to determine number of weights
            self.bias, self.b_d = np.zeros(1), None  # Initiate bias with a float
        else:
            # Input layer does not have weights and bias
            self.weights, self.bias = [None, None]

    # Update the parameters of the neuron
    def update(self, z=None, a=None, weights=None, bias=None, obs_ind=0):
        # Z and A are vectors with length equal to number of observations
        if z is not None:
            self.z[obs_ind] = z if isinstance(z, np.float64) else z[self.neuron_n - 1]
        if a is not None:
            if isinstance(a, np.float64) or np.isscalar(a):
                self.a[obs_ind] = a
            else:
                self.a[obs_ind] = a[self.neuron_n - 1]
        if weights is not None:
            weights = np.atleast_1d(weights)
            if weights.size != self.prev_n:
                raise ValueError(f"weights must have length {self.prev_n}")
            self.weights = weights
        if bias is not None:
            self.bias = bias

    # Function to reshape the parameters once the number observations are known
    def reshape(self, size):
        # Size refers to the size of the batch or set (number of observations)
        self.z = np.zeros(size)
        self.a = np.zeros(size)
        self.w_d = np.zeros((self.prev_n, size))
        self.b_d = np.zeros(size)
        self.a_d = np.zeros(size)


    # Generic function to check the derivatives
    def get_derivs(self):
        keys = ["a_d", "w_d", "b_d"]
        act_der = self.a_d
        wght_der = self.w_d
        bs_der = self.b_d
        derivs = {key: value for key, value in zip(keys, [act_der, wght_der, bs_der])}
        for key, value in derivs.items():
            print(f"Neuron:", self.neuron_n)
            print(f"{key}: {value}")

