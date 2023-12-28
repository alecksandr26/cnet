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

# Set the size of the matrix (adjust as needed)
matrix_size = 1000

# Measure time for matrix multiplication
elapsed_time = measure_matrix_multiplication(matrix_size)

# Matrix multiplication time for size 1000 x 1000: 0.501300 seconds
print(f"Matrix multiplication time for size {matrix_size} x {matrix_size}: {elapsed_time:.6f} seconds")
