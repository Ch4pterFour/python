import streamlit as st
from network import Network
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

# To Do:
# Add Git
# Host on Streamlit

# CSS for button styling
st.markdown("""
    <style>
        button {
            border-radius: 8px;
            padding: 20px 20px;
            font-size: 16px;
            border: none;
            transition: 0.3s;
        }
        
        button:hover{ border: none; }
        button:active { opacity: .6; color: white; }
        button[kind="primary"] { background-color: #ff4b4b; color: white; }
        button[kind="primary"]:hover { background-color: #d43f3f; }
        button[kind="secondary"] { background-color: #4CAF50; color: white; }
        button[kind="secondary"]:hover { background-color: #45a049; color: white; }
    </style>
""", unsafe_allow_html=True)

# Initialize session state variables
if "test_n" not in st.session_state:
    st.session_state.test_n = (1/3)
if "seed" not in st.session_state:
    st.session_state.seed = None
if "network_result" not in st.session_state:
    st.session_state.network_result = None
if "parameters" not in st.session_state:
    st.session_state.parameters = None
if "network_done" not in st.session_state:
    st.session_state.network_done = False

# Title and description
st.title("Simple Neural Network to Analyze the Iris Dataset")
st.write("Here you can use the simple neural network from the Frontiers for Young Minds article (NAME + URL) "
         "to analyze the Iris dataset. The network starts with different "
         "random weights each time you click the 'Run network' button (remember the biases are always 0 at the start). "
         "By entering a seed you can force the network to start with certain weights. "
         "So, if you provide the same seed and test set size, the results will be the same.")
st.write("### Exercises/examples")
st.write("1. Try to replicate the results of the paper. The seed used in the paper is 22 and "
         "the test set size is 50 (.33). Check the weights and biases.")
st.write("2. Try the seed 2 and then 111 (keep test set size .33). What is the accuracy and "
         "what do you notice about the predictions? Why do you think this might happen?")
st.write("3. Use the seed 5 and then run the network with the test sizes .9, .1 and .33. "
         "What happens to the accuracy and predictions each time your run the network?")

# Seed Input
st.write("**Change the seed**")
seed_input = st.text_input(
    "Enter a number without decimals:",
    placeholder="Enter a seed",
    value=st.session_state.seed,
    key=st.session_state.get("widget_key", "seed_input")
)

# Train / Test set input
st.write("**Adjust the slider below to choose the size of the test set**")
test_n = st.slider("Select a number:", min_value=0.1, max_value=0.9, step=0.01, key="test_n")

# Reset button
if st.button("Reset", type="primary"):
    st.session_state.clear()
    st.session_state["widget_key"] = str(np.random.rand())
    st.session_state["network_result"] = None
    st.session_state.network_done = False
    st.rerun()

# Load Iris dataset
iris = datasets.load_iris()
iris = np.column_stack((iris.data, iris.target))

@st.cache_data(show_spinner=True)
# Function to run the neural network
def run_network(seed, test_size):
    try:
        np.random.seed(seed)
        iris_train, iris_test = train_test_split(iris, test_size=test_size, random_state=42)

        # Create a neural network with 3 layers (Input, hidden layer, and output layer)
        iris_network = Network(structure=(4, 4, 3), categorical=True, function="ReLu", rate=0.01, tolerance=0.001)

        # Predict and evaluate
        result = iris_network.predict((0, 1, 2, 3), 4, iris_train, iris_test, show=False)
        parameters = iris_network.get_structure()

        return result, parameters  # Return values instead of modifying session state

    except Exception as e:
        return f"ERROR: {e}", None

# Run Neural Network button
if st.button("Run Neural Network"):
    try:
        st.session_state.network_result = None
        st.session_state.parameters = None

        # Ensure seed_input is converted to an integer safely
        if seed_input.strip():  # Check if input is not empty
            try:
                seed = int(seed_input.strip())  # Convert to integer
            except ValueError:
                st.error("Seed must be a valid integer.")
                st.stop()
        else:
            seed = np.random.randint(1, 10000)

        st.session_state.seed = seed  # Store in session state

        # Run the network and store results
        result, parameters = run_network(seed, test_n)

        # Get seed and test size
        if isinstance(result, str) and result.startswith("ERROR"):
            st.error(result)
            st.session_state.network_done = False
        else:
            st.session_state.network_result = result
            st.session_state.parameters = parameters
            st.session_state.network_done = True

    except ValueError:
        st.error("Seed must be a number.")
        st.stop()

# Display results if network is done
if st.session_state["network_done"]:
    result = st.session_state["network_result"]
    parameters = st.session_state["parameters"]

    if isinstance(result, str) and result.startswith("ERROR"):
        st.error(result)
    else:
        st.write(f"Random seed: {st.session_state.seed}")
        st.write(f"Test set size: {int(st.session_state.test_n * 150)}")
        st.write("### Results of Neural Network on Iris Dataset")
        st.write("#### Confusion Matrix")
        st.dataframe(result["Confusion"])
        st.write("#### Accuracy")
        st.write(f"{result['Accuracy']:.2f}")

        if result["Accuracy"] < 0.7:
            st.write("The accuracy may be low due to suboptimal weights. "
                     "Try a different seed for better results.")

        # change output format
        st.write("### Final Weights and Biases:")
        for layer, layer_data in parameters.items():
            st.write(f"#### {layer}")  # Display layer name (e.g., "Input layer")

            for param_name, param_value in layer_data.items():
                st.write(f"**{param_name}**")  # Display "Weights" or "Biases"

                # Convert NumPy arrays to Pandas DataFrames for better visualization
                if isinstance(param_value, np.ndarray):
                    df = pd.DataFrame(param_value)

                    # If it's a 1D array (biases), reshape to a column format
                    if param_value.ndim == 1:
                        df = pd.DataFrame(param_value, columns=["Biases"])

                    st.dataframe(df)  # Use st.table(df) for a static table
                else:
                    st.write(param_value)  # Fallback if not an array
        st.write("**Note:** the decimals of the weights and biases can be slightly different than those in the "
                 "Frontiers article Figure 2b because of the rounding algorithm used by Python.")

