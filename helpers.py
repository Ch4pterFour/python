import numpy as np


# Relu function
def ReLu(vector):
    return np.maximum(0, vector)

def ReLu_dx(vector):
    # This returns 1 for elements greater than 0, and 0 for elements less than or equal to 0
    return np.where(vector > 0, 1, 0)

# SoftPlus (Used by Josh Starmer)
def SoftPlus(vector):
    return np.log(1 + np.power(np.exp(1), vector))

def SoftPlus_dx(vector):
    # The derivative of the softplus function is the sigmoid function
    return (np.exp(vector) / (1 + np.exp(vector)))

# Sigmoid function
def Sigmoid(vector):
    return 1 / (1 + np.exp(-vector))

def Sigmoid_dx(vector):
    # First, calculate the sigmoid function
    sigmoid = 1 / (1 + np.exp(-vector))
    # Then, compute the derivative
    return sigmoid * (1 - sigmoid)

def SoftMax(vector):
    e_x = np.exp(vector - np.max(vector))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def ArgMax(vector):
    return np.argmax(vector)


def CE_dx(bias, category, probabilities):  # Probabilities is a numpy array
    if bias == category:
        dx = probabilities[bias] - 1
    else:
        dx = probabilities[bias]
    return dx


def create_shuffled_batches(data, batch_size):
    # Shuffle the data
    np.random.shuffle(data)
    # Create batches
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


epsilon = 1e-10


################################################################################
###############################   Optimization   ###############################
################################################################################

# # Test vectors
# vector = np.random.randn(1000000)  # Large vector for performance testing
#
# # Timing original implementation
# start_time = time.time()
# result_relu = ReLu(vector.copy())  # Use .copy() to avoid in-place modification affecting the next test
# relu_time = time.time() - start_time
# print(f"Relu implementation time: {relu_time:.6f} seconds")
#
# # Timing optimized implementation
# start_time = time.time()
# result_sigmoid = Sigmoid(vector)
# sigmoid_time = time.time() - start_time
# print(f"Sigmoid implementation time: {sigmoid_time:.6f} seconds")