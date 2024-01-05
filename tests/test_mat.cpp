// To have output: make test ARGS=-V

// Run valgrind to check memory issues
// valgrind --leak-check=full --track-origins=yes -s --show-leak-kinds=all --max-stackframe=62179200

#include <iostream>
#include <gtest/gtest.h>
#include <chrono>

#include "cnet/mat.hpp"

using namespace cnet;

TEST(MatTest, TestMatInit)
{
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
	ASSERT_EQ(D(0, 1), 2.0);
}

TEST(MatTest, TestMatAddOperation)
{
	constexpr std::size_t rows = 10;
	constexpr std::size_t cols = 5;
	
	mat<double> A(rows, cols, 1.0);
	mat<double> B(rows, cols, 2.0);

	// Simple addition
	mat<double> C = A + B;
	
	ASSERT_EQ(C.get_rows(), rows);
	ASSERT_EQ(C.get_cols(), cols);

	for (std::size_t i = 0; i < rows; i++)
		for (std::size_t j = 0; j < cols; j++)
			ASSERT_EQ(C(i, j), 3.0);

	// Simple subtraction
	mat<double> D = A - B;
	
	ASSERT_EQ(D.get_rows(), rows);
	ASSERT_EQ(D.get_cols(), cols);

	for (std::size_t i = 0; i < rows; i++)
		for (std::size_t j = 0; j < cols; j++)
			ASSERT_EQ(D(i, j), -1.0);


	A.resize(rows, cols, 1.0);
	B.resize(rows, cols, 2.0);

	// Simple addition
	A += B;
	
	ASSERT_EQ(A.get_rows(), rows);
	ASSERT_EQ(A.get_cols(), cols);

	for (std::size_t i = 0; i < rows; i++)
		for (std::size_t j = 0; j < cols; j++)
			ASSERT_EQ(A(i, j), 3.0);
	
	// Simple subtraction
	A.resize(rows, cols, 1.0);
	A -= B;
	
	ASSERT_EQ(A.get_rows(), rows);
	ASSERT_EQ(A.get_cols(), cols);

	for (std::size_t i = 0; i < rows; i++)
		for (std::size_t j = 0; j < cols; j++)
			ASSERT_EQ(A(i, j), -1.0);
}

TEST(MatTest, TestMatMulOperation)
{
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

	A = {
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0}
	};
	
	B = {
		{2.0},
		{2.0},
		{2.0}
	};

	A *= B;

	// Should have the number of rows of A
	// And the number of columns of B
	ASSERT_EQ(A.get_rows(), 2);
	ASSERT_EQ(A.get_cols(), 1);

	// C(0, 0) = (1.0 * 2.0) + (2.0 * 2.0) + (2.0 * 3.0) = 12.0
	ASSERT_EQ(C(0, 0), 12.0);


	// Multiplying big matrix
	M.resize(10, 10, 1.0);
	N.resize(10, 1, 2.0);
	
	M *= N;

	// Should have the number of rows of M
	// And the number of columns of N
	ASSERT_EQ(R.get_rows(), 10);
	ASSERT_EQ(R.get_cols(), 1);
	
	ASSERT_EQ(R(0, 0), 20.0);
}

TEST(MatTest, TestMatMulAddOperation)
{
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
	
	mat<double> A(size_mat, size_mat);
	mat<double> B(size_mat, size_mat);

	// Assing random values
	A.rand(0.0, 1.0);
	B.rand(0.0, 1.0);

	auto beg = std::chrono::high_resolution_clock::now();
	
	// mat mul of 1000
	mat<double> R = A * B;
	
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
	
	// Convert microseconds to seconds 
	double seconds = duration.count() / 1e6;



	std::cout << "Elapsed Time: " << std::fixed << std::setprecision(6)
		  << seconds << " seconds" << std::endl;

	// It should be faster than 0.4 seconds
	ASSERT_TRUE(seconds < 0.4);
}


TEST(MatTest, TestMatScalarMul)
{
	mat<double> A(10, 10, 1.0);

	A = A * 10.0;

	ASSERT_EQ(A.get_rows(), 10);
	ASSERT_EQ(A.get_cols(), 10);


	for (std::size_t i = 0; i < A.get_rows(); i++)
		for (std::size_t j = 0; j < A.get_cols(); j++)
			ASSERT_EQ(A(i, j), 10.0);

	mat<double> B(10, 10, 1.0);

	B *= 10.0;

	ASSERT_EQ(B.get_rows(), 10);
	ASSERT_EQ(B.get_cols(), 10);

	for (std::size_t i = 0; i < B.get_rows(); i++)
		for (std::size_t j = 0; j < B.get_cols(); j++)
			ASSERT_EQ(B(i, j), 10.0);
}


TEST(MatTest, TestMatGrandSum)
{
	mat<double> A(10, 10, 1.0);

	std::cout << A.grand_sum() << std::endl;
	ASSERT_EQ(A.grand_sum(), 100.00);
	
	mat<double> B(10, 10, 0.0);

	std::cout << B.grand_sum() << std::endl;
	ASSERT_EQ(B.grand_sum(), 0.0);
}


TEST(MatTest, TestMatRand)
{
	mat<double> A(10, 10, 2.0);
	
	A.rand(0.0, 1.0);

	for (std::size_t i = 0; i < A.get_rows(); i++)
		for (std::size_t j = 0; j < A.get_cols(); j++)
			ASSERT_TRUE(A(i, j) <= 1.0 && A(i, j) >= 0.0);
}



TEST(MatTest, TestTranspose)
{
	mat<double> A(10, 10, 0.0);

	for (std::size_t i = 0; i < A.get_rows(); i++)
		for (std::size_t j = i; j < A.get_cols(); j++)
			A(i, j) = (i + 1) * (j + 1);
	std::cout << A << std::endl;

	mat<double> B = A.transpose();

	std::cout << B << std::endl;
	
	for (std::size_t i = 0; i < A.get_rows(); i++)
		for (std::size_t j = 0; j < A.get_cols(); j++) {
			if (j <= i)
				ASSERT_EQ(B(i, j), A(j, i));
			else
				ASSERT_EQ(B(i, j), 0.0);
		}

	A.transpose_();

	for (std::size_t i = 0; i < A.get_rows(); i++)
		for (std::size_t j = 0; j < A.get_cols(); j++)
			ASSERT_EQ(B(i, j), A(i, j));
	
}

TEST(MatTest, TestElementWiseMul)
{
	mat<double> A(10, 10, 1.0);
	mat<double> B(10, 10, 2.0);

	// Element wise mul
	mat<double> C = A ^ B;
	
	std::cout << C << std::endl;

	for (std::size_t i = 0; i < C.get_rows(); i++)
		for (std::size_t j = 0; j < C.get_cols(); j++)
			ASSERT_EQ(C(i, j), 2.0);

	A ^= B;

	for (std::size_t i = 0; i < A.get_rows(); i++)
		for (std::size_t j = 0; j < A.get_cols(); j++)
			ASSERT_EQ(A(i, j), 2.0);
}
