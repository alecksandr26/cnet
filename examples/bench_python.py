import numpy as np
import time

# Function to measure matrix multiplication time
def measure_matrix_multiplication(n):
    # Generate two random matrices of size n x n
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)

    # Measure time for NumPy matrix multiplication
    start_time = time.time()
    result_np = np.dot(A, B)
    elapsed_time_np = time.time() - start_time

    return elapsed_time_np

# Function to measure matrix addition time
def measure_matrix_addition(n):
    # Generate two random matrices of size n x n
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)

    # Measure time for NumPy matrix addition
    start_time = time.time()
    result_np = A + B
    elapsed_time_np = time.time() - start_time

    return elapsed_time_np

# Set the size of the matrix (adjust as needed)
matrix_size = 1000

# Measure time for matrix multiplication
elapsed_time = measure_matrix_multiplication(matrix_size)

# Matrix multiplication time for size 1000 x 1000: 0.501300 seconds
print(f"Matrix multiplication time for size {matrix_size} x {matrix_size}: {elapsed_time:.6f} seconds")

# Measure time for matrix addition
elapsed_time = measure_matrix_addition(matrix_size)

# Matrix addition time for size 1000 x 1000: 0.002466 seconds
print(f"Matrix addition time for size {matrix_size} x {matrix_size}: {elapsed_time:.6f} seconds")


import tensorflow as tf

# Create a simple tensor
tensor = tf.constant([[1, 2], [3, 4]])

# Print the tensor
tf.print("Tensor:", tensor)

print(tensor)

# Set the random seed for reproducibility
tf.random.set_seed(42)

# Create random matrices
matrix_size = 1000
matrix_a = tf.random.normal((matrix_size, matrix_size))
matrix_b = tf.random.normal((matrix_size, matrix_size))

print(matrix_a)
print(matrix_b)

# Benchmark matrix multiplication
start_time = time.time()
result = tf.matmul(matrix_a, matrix_b)
print(result)
end_time = time.time()

# Print the execution time
execution_time = end_time - start_time
print(f"Matrix multiplication of size {matrix_size}x{matrix_size} took {execution_time:.4f} seconds.")

