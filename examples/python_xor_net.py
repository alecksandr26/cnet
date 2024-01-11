
# First I import numpy library and matplotlib used to display loss curve
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# Then I defined the inputs and structure of neural network

# These are XOR inputs
x = np.array([
    [[0], [0]],
    [[1], [0]],
    [[0], [1]],
    [[1], [1]],
    ])

# These are XOR outputs
y = np.array([[0], [1], [1], [0]])

# Number of inputs
n_x = 2

# Number of neurons in output layer
n_y = 1
# Number of neurons in hidden layer
n_h = 2

# Total training examples
m = 4

# Define weight matrices for neural network
# Initialize weights and biases at zero
w1 = np.zeros((n_h, n_x))
b1 = np.zeros((n_h, 1))

w2 = np.zeros((n_y, n_h))
b2 = np.zeros((n_y, 1))

# I didnt use bias units

# We will use this list to accumulate losses
losses = []

# Here I define the important processes as Python methods
# I used sigmoid activation function for hidden layer and output
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivate(z):
    return 1 / (2 + np.exp(z) + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def relu_derivate(z):
    return np.where(z > 0, 1, 0)

def linear(z):
    return z

def linear_derivate(z):
    return np.ones(z.shape)


# Forward propagation
def forward_prop(w1, b1, w2, b2, x):
    # Layer 1
    z1 = np.dot(w1, x) + b1
    a1 = sigmoid(z1)

    # Layer 2
    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)
    return z1, a1, z2, a2

# Backward propagation
def back_prop(m, w1, b1, w2, b2, z1, a1, z2, a2, x, y):
    # Layer 2
    da2 = 2 * (a2 - y) / m
    dz2 = da2 * sigmoid_derivate(z2)
    dw2 = np.dot(dz2, a1.T)
    db2 = dz2

    # Layer 1
    da1 = np.dot(w2.T, dz2)
    dz1 = da1 * sigmoid_derivate(z2)
    dw1 = np.dot(dz1, x.T)
    db1 = dz1
    
    return dz2, dw2, db2, dz1, dw1, db1

# Now we run the neural network for 10000 epochs and observe the loss value
epochs = 1
# Learning rate
lr = 100.0
for e in range(epochs):
    s = 0.0
    for i in range(4):
        print("------------------------")
        print(f"x{i}: ", x[i])
        print(f"y{i}: ", y[i])
        
        z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, x[i])
        
        print("a2:", a2)
        print("z2:", z2)
        print("a1:", a1)
        print("z1:", z1)
        s += np.squeeze(a2 - y[i]) ** 2
        
        da2, dw2, db2, dz1, dw1, db1 = back_prop(m, w1, b1, w2, b2, z1, a1, z2, a2, x[i], y[i])
        print(dw2)
        print(db2)

        print(dw1)
        print(db1)
        
        w2 = w2 - lr * dw2
        b2 = b2 - lr * db2
        w1 = w1 - lr * dw1
        b1 = b1 - lr * db1
        
    loss = (1 / (2 * m)) * s  # Mean Squared Error (MSE) cost function
    losses.append(loss)
    # print("loss: ", loss)

        

# We plot losses to see how our network is doing
fig, ax = plt.subplots()

# Plot the data
ax.plot(losses)

# Set a custom formatter for the y-axis labels
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.10f}"))

# Set labels and show the plot
ax.set_xlabel("EPOCHS")
ax.set_ylabel("Loss value")
# plt.show()

# Now after training we see how our neural network is doing in terms of predictions
def predict(w1, b1, w2, b2, input):
    z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, test)
    a2 = np.squeeze(a2)
    print(a2)
        

# Test predictions
print("last loss: ", losses[-1])

test = np.array([[0], [0]])
predict(w1, b1, w2, b2, test)

test = np.array([[0], [1]])
predict(w1, b1, w2, b2, test)

test = np.array([[1], [0]])
predict(w1, b1, w2, b2, test)

test = np.array([[1], [1]])
predict(w1, b1, w2, b2, test)
