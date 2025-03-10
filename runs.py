from network import Network
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

np.random.seed(22)

################################################################################
###############################   Iris dataset   ###############################
################################################################################


iris = datasets.load_iris()  # Load the Iris dataset
iris = np.column_stack((iris.data, iris.target))  # Combine features and categories into a single dataset

# Split the Iris dataset into training and test set
iris_train, iris_test = train_test_split(iris, test_size=0.33, random_state=42)
# Create a neural network with 3 layers (Input, output layer and a hidden layer with 4 neurons)
iris_network = Network(structure=(4, 4, 3), categorical=True, function="ReLu", rate=.01, tolerance=.001)

# Call the predict function on our network to start training and testing
result = iris_network.predict((0, 1, 2, 3), 4, iris_train, iris_test, show=False)
structure = iris_network.get_structure()
# Print the results in the console
print("Confusion matrix")
print(result["Confusion"])
print("Accuracy")
print(result["Accuracy"])
for key, value in structure.items():
    print(f"{key}, {value}")







# MAX = 10 # Change number to see average accuracy, takes 13 seconds on average to run the network
# res = np.zeros([MAX])
# for i in range(0, MAX):
#     iris_network = Network(structure=(4, 5, 3), categorical=True, function="ReLu", rate=.01, tolerance=.001)
#     res[i] = iris_network.predict((0,1,2,3),4,iris_train, iris_test)["Accuracy"]
# print(res)
# acc = np.mean(res)
# print(acc)
# iris_time = time.time() - start_time
# print(f"Network run time: {iris_time:.6f} seconds")


################################################################################
#############################   Code   Graveyard   #############################
################################################################################

##################################   _____   ###################################
##################################  /     \  ###################################
################################## | () () | ###################################
##################################  \  ^  /  ###################################
##################################   |||||   ###################################
##################################    ```    ###################################

# Dosage example
# test = [np.array([[-34.4], [-2.52]]), np.array([[-1.3, 2.28],[2.14, 1.29]])]
# josh_w = [np.array([[-34.4], [-2.52]]), np.array([-1.3, 2.28])]  # import these weights and biases to test network
# josh_b = [np.array([2.14, 1.29]), np.array([0])]
# josh = np.array([[0],[.5],[1]])  # how does the format of the input compare to that of pred val for error calc
# n = Network([1, 2, 1], nobs=3)
# n.set_w_b(josh_w, josh_b)
# forward = n.forward_propagate(josh, y=np.array([[0],[1],[0]]), show=True)

# Backprop dosage
# josh_w = [np.array([[2.74], [-1.13]]), np.array([0.36, 0.63])]  # import these weights and biases to test network
# josh_b = [np.array([0, 0]), np.array([0])]
# josh = np.array([[0],[.5],[1]])  # how does the format of the input compare to that of pred val for error calc
# n = Network([1, 2, 1], nobs=3, rate=.1, y=np.array([[0],[1],[0]]))
# n.set_w_b(josh_w, josh_b)
#
# forward = n.forward_propagate(josh, y=np.array([[0],[1],[0]]))
#
# n.backward_propagate(josh, preds=forward["Predictions"])

# Backprop multiple pred/out
# josh = np.array([[0, .1], [.4,.5], [.9, 1]])
#
# n = Network([2, 2, 3], categorical=True, function="SoftPlus", rate=.1, nobs=3, y=np.array([[0],[1],[0]]))
#
# #n.forward_propagate(josh, y=np.array([[0],[1],[0]]), show=True)
#
# forward = n.forward_propagate(josh, y=np.array([[0],[1],[0]]))
# n.backward_propagate(josh, preds=forward["Predictions"])


#n.get_structure()

#  Iris example
# josh_w = [np.array([[-2.5, 0.6], [-1.5, 0.4]]), np.array([[-0.1, 1.5], [2.4, -5.2], [-2.2, 3.7]])]
# josh_b = [np.array([1.6, 0.70]), np.array([-2, 0, 1])]
#
#
# joshiris = np.array([[.04, .42, 0], [1, .54, 1], [.5, .37, 2]])
# n = Network([2, 2, 3], categorical=True, function="ReLu", rate=.1, nobs=3)
# n.set_w_b(josh_w, josh_b)
# n.predict(train=joshiris, test=joshiris, x=(0,1), y=2)