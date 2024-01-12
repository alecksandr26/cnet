import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs



X, y = make_blobs(n_samples = 200, n_features = 2, cluster_std=.1,
                  centers= [(1,1), (1,0), (0,0),(0,1)])

y[y==2]=0
y[y==3]=1

print(X, y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=19)

# Create a sequential model
model = Sequential()

# Add layers
model.add(Dense(3, input_dim=2, activation='sigmoid', kernel_initializer = 'zeros'))
model.add(Dense(2, activation='sigmoid', kernel_initializer = 'zeros')) 
model.add(Dense(1, activation='sigmoid', kernel_initializer = 'zeros'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer=SGD(learning_rate = 0.1))

# Train the model
model.fit(x_train, y_train, epochs=500, verbose=0)

# Evaluate the model
loss = model.evaluate(x_test, y_test)
print(f"Loss: {loss}")

# XOR inputs and outputs
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([[0], [1], [1], [0]])

# Print predictions
predictions = model.predict(inputs)
print("Predictions:")
for i in range(len(inputs)):
    print(f"Input: {inputs[i]}, Predicted Output: {predictions[i]}")

# Print weights
print("Weights:")
for layer in model.layers:
    print(f"Weights for Layer: {layer.get_config()['name']}")
    weights, biases = layer.get_weights()
    print(f"Weights: {weights}")
    print(f"Biases: {biases}")


