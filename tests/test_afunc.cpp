// To have output: make test ARGS=-V
// To have output in the error testcases: make test ARGS='--rerun-failed --output-on-failure'

// Run valgrind to check memory issues
// valgrind --leak-check=full --track-origins=yes -s --show-leak-kinds=all --max-stackframe=62179200


#include <iostream>
#include <gtest/gtest.h>
#include <chrono>

#include "cnet/afunc.hpp"
#include "cnet/mat.hpp"

using namespace cnet;
using namespace afunc;

TEST(TestAFunc, TestSigmoid) {
	mat<double> A(10, 10, 15.0);

	mat<double> B = sigmoid<double>()(A);
	
	std::cout << B << std::endl;

	for (std::size_t i = 0; i < B.get_rows(); i++)
		for (std::size_t j = 0; j < B.get_cols(); j++)
			EXPECT_NEAR(B(i, j), 1.0, 1e-6);
	
	A.resize(10, 10, 0.0);

	B = sigmoid<double>()(A);

	for (std::size_t i = 0; i < B.get_rows(); i++)
		for (std::size_t j = 0; j < B.get_cols(); j++)
			EXPECT_NEAR(B(i, j), 0.5, 1e-6);

	A.resize(10, 10, -15.0);
	
	B = sigmoid<double>()(A);

	for (std::size_t i = 0; i < B.get_rows(); i++)
		for (std::size_t j = 0; j < B.get_cols(); j++)
			EXPECT_NEAR(B(i, j), 0.0, 1e-6);
	
}


TEST(TestAFunc, TestRelu) {
	mat<double> A(10, 10, 15.0);

	mat<double> B = relu<double>()(A);
	
	std::cout << B << std::endl;
	
	for (std::size_t i = 0; i < A.get_rows(); i++)
		for (std::size_t j = 0; j < A.get_cols(); j++)
			ASSERT_EQ(B(i, j), A(i, j));

	A.resize(10, 10, - 15.0);

	B = relu<double>()(A);

	for (std::size_t i = 0; i < B.get_rows(); i++)
		for (std::size_t j = 0; j < B.get_cols(); j++)
			ASSERT_EQ(B(i, j), 0.0);
}
