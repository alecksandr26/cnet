// To have output: make test ARGS=-V

// Run valgrind to check memory issues
// valgrind --leak-check=full --track-origins=yes -s --show-leak-kinds=all --max-stackframe=62179200

#include <iostream>
#include <gtest/gtest.h>
#include <chrono>

#include "cnet/mat.hpp"

using namespace cnet;

TEST(MatTest, TestMatInit) {
	// Just allocation but not initailization of the matrix
	mat<double> A(2, 3);
	
	ASSERT_EQ(A.get_rows(), 2);
	ASSERT_EQ(A.get_cols(), 3);

	// Initialization with zeros
	mat<double> B(3, 3, 0.0);
	
	ASSERT_EQ(B.get_rows(), 3);
	ASSERT_EQ(B.get_cols(), 3);
	
	ASSERT_EQ(B(0, 0), 0.0);

	// Initializatino with initializer lists
	mat<double> C = {
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0}
	};
	
	ASSERT_EQ(C.get_rows(), 2);
	ASSERT_EQ(C.get_cols(), 3);

	ASSERT_EQ(C(0, 0), 1.0);
	ASSERT_EQ(C(0, 1), 2.0);
	
	// Initializatino with another matrix
	// Copying the data
	mat<double> D = C;
	
	ASSERT_EQ(D.get_rows(), 2);
	ASSERT_EQ(D.get_cols(), 3);
	
	ASSERT_EQ(D(0, 0), 1.0);
}

TEST(MatTest, TestMatAddOperation) {
	mat<double> A(2, 2, 1.0);
	mat<double> B(2, 2, 2.0);

	// Simple addition
	mat<double> C = A + B;
	
	ASSERT_EQ(C.get_rows(), 2);
	ASSERT_EQ(C.get_cols(), 2);
	
	ASSERT_EQ(C(0, 0), 3.0);
	ASSERT_EQ(C(1, 1), 3.0);

	// Simple subtraction
	mat<double> D = A - B;
	
	ASSERT_EQ(D(0, 0), -1.0);
	ASSERT_EQ(D(1, 1), -1.0);
}

TEST(MatTest, TestMatMulOperation) {
	mat<double> A = {
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0}
	};
	
	mat<double> B = {
		{2.0},
		{2.0},
		{2.0}
	};

	mat<double> C = A * B;

	// Should have the number of rows of A
	// And the number of columns of B
	ASSERT_EQ(C.get_rows(), 2);
	ASSERT_EQ(C.get_cols(), 1);

	// C(0, 0) = (1.0 * 2.0) + (2.0 * 2.0) + (2.0 * 3.0) = 12.0
	ASSERT_EQ(C(0, 0), 12.0);


	// Multiplying big matrix
	mat<double> M(10, 10, 1.0);
	mat<double> N(10, 1, 2.0);
	
	mat<double> R = M * N;

	// Should have the number of rows of M
	// And the number of columns of N
	ASSERT_EQ(R.get_rows(), 10);
	ASSERT_EQ(R.get_cols(), 1);
	
	ASSERT_EQ(R(0, 0), 20.0);
}

TEST(MatTest, TestMatMulAddOperation) {
	mat<double> A(2, 2, 1.0);
	mat<double> B(2, 2, 2.0);
	mat<double> C(2, 2, 1.0);

	// A 2 rows and B 2 cols 
	mat<double> D = A * B + C;


	ASSERT_EQ(D.get_rows(), 2);
	ASSERT_EQ(D.get_cols(), 2);
	
	
	ASSERT_EQ(D(0, 0), 5.0);
}

TEST(MatTest, TestMatTimeMulOperation) {
	static constexpr int size_mat = 1000;
	
	mat<double> X(size_mat, 1);
	mat<double> A(size_mat, size_mat);
	mat<double> B(size_mat, 1);

	// Assing random values
	rand_mat(A, 0.0, 1.0);
	rand_mat(X, 0.0, 1.0);
	rand_mat(B, 0.0, 1.0);
	
	auto beg = std::chrono::high_resolution_clock::now();
	
	mat<double> X_T = X.transpose();
	mat<double> B_T = B.transpose();
	
	// X_T 1 rows and A 100 cols
	mat<double> R = X_T * A + B_T;
	
	auto end = std::chrono::high_resolution_clock::now();
	
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - beg);

	// Convert microseconds to seconds
	double seconds = duration.count() / 1e6;

	std::cout << "Elapsed Time: " << std::fixed << std::setprecision(6)
		  << seconds << " seconds" << std::endl;
}


