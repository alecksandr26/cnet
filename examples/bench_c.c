#include <stdio.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <time.h>

void print_matrix(gsl_matrix *m) {
	for (size_t i = 0; i < m->size1; ++i) {
		for (size_t j = 0; j < m->size2; ++j) {
			printf("%g ", gsl_matrix_get(m, i, j));
		}
		printf("\n");
	}
	printf("\n");
}

int main() {
	const size_t rows = 1000;
	const size_t cols = 1000;

	gsl_matrix *A = gsl_matrix_alloc(rows, cols);
	gsl_matrix *B = gsl_matrix_alloc(cols, rows);
	gsl_matrix *C = gsl_matrix_alloc(rows, rows);

	// Initialize matrices A and B
	for (size_t i = 0; i < rows; ++i) {
		for (size_t j = 0; j < cols; ++j) {
			gsl_matrix_set(A, i, j, i * cols + j + 1);
			gsl_matrix_set(B, j, i, i * cols + j + 1);  // Transpose B
		}
	}

	// Measure the time taken for matrix multiplication
	clock_t start_time = clock();

	// Perform matrix multiplication: C = A * B
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, A, B, 0.0, C);

	clock_t end_time = clock();
	double cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;

	printf("Matrix multiplication time: %f seconds\n", cpu_time_used);

	// Optionally print the result matrix
	// printf("Matrix C (result of A * B):\n");
	// print_matrix(C);

	gsl_matrix_free(A);
	gsl_matrix_free(B);
	gsl_matrix_free(C);

	return 0;
}
