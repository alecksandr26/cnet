// To have output: make test ARGS=-V
// To have output in the error testcases: make test ARGS='--rerun-failed --output-on-failure'

// Run valgrind to check memory issues
// valgrind --leak-check=full --track-origins=yes -s --show-leak-kinds=all --max-stackframe=62179200


#include <iostream>
#include <gtest/gtest.h>
#include <chrono>

#include "cnet/dtypes.hpp"
#include "cnet/mat.hpp"
#include "cnet/variable.hpp"
#include "cnet/activation.hpp"

using namespace std;
using namespace cnet;
using namespace dtypes;
using namespace mathops;
using namespace variable;
using namespace activation;

TEST(TestAFuncFloat32, TestLinear)
{
	Mat<float32> A(10, 10, 15.0);
	
	Mat<float32> B = Linear(A);
	cout << B << endl;
	
	for (size_t i = 0; i < B.get_rows(); i++)
		for (size_t j = 0; j < B.get_cols(); j++)
			ASSERT_EQ(B(i, j), A(i, j));
	
	A.resize(10, 10, 0.0);

	B = LinearDerivate(A);
	cout << B << endl;

	for (size_t i = 0; i < B.get_rows(); i++)
		for (size_t j = 0; j < B.get_cols(); j++)
			ASSERT_EQ(B(i, j), 1.0);
}

TEST(TestAFuncFloat64, TestLinear)
{
	Mat<float64> A(10, 10, 15.0);
	
	Mat<float64> B = Linear(A);
	cout << B << endl;
	
	for (size_t i = 0; i < B.get_rows(); i++)
		for (size_t j = 0; j < B.get_cols(); j++)
			ASSERT_EQ(B(i, j), A(i, j));
	
	A.resize(10, 10, 0.0);

	B = LinearDerivate(A);
	cout << B << endl;

	for (size_t i = 0; i < B.get_rows(); i++)
		for (size_t j = 0; j < B.get_cols(); j++)
			ASSERT_EQ(B(i, j), 1.0);
}

TEST(TestAFuncVar, TestLinear)
{
	Var A(10, 10, (float64) 15.0);
	Var B = Linear(A);
	
	cout << B << endl;
	
	for (size_t i = 0; i < B.get_rows(); i++)
		for (size_t j = 0; j < B.get_cols(); j++)
			ASSERT_EQ(B.at_mf64(i, j), A.at_mf64(i, j));
	
	A.resize(10, 10, (float64) 0.0);

	B = LinearDerivate(A);
	cout << B << endl;

	for (size_t i = 0; i < B.get_rows(); i++)
		for (size_t j = 0; j < B.get_cols(); j++)
			ASSERT_EQ(B.at_mf64(i, j), 1.0);

	// For float 32
	A.resize(10, 10, (float32) 15.0);
	B.resize(10, 10, (float32) 15.0);
	
	B = Linear(A);
	
	cout << B << endl;
	
	for (size_t i = 0; i < B.get_rows(); i++)
		for (size_t j = 0; j < B.get_cols(); j++)
			ASSERT_EQ(B.at_mf32(i, j), A.at_mf32(i, j));
	
	A.resize(10, 10, (float32) 0.0);

	B = LinearDerivate(A);
	cout << B << endl;

	for (size_t i = 0; i < B.get_rows(); i++)
		for (size_t j = 0; j < B.get_cols(); j++)
			ASSERT_EQ(B.at_mf32(i, j), (float32) 1.0);
}


TEST(TestAFuncFloat32, TestRelu)
{
	Mat<float32> A(10, 10, 15.0);

	Mat<float32> B = Relu(A);
	
	cout << B << endl;
	
	for (size_t i = 0; i < A.get_rows(); i++)
		for (size_t j = 0; j < A.get_cols(); j++)
			ASSERT_EQ(B(i, j), A(i, j));

	A.resize(10, 10, - 15.0);

	B = Relu(A);

	cout << B << endl;
	
	for (size_t i = 0; i < B.get_rows(); i++)
		for (size_t j = 0; j < B.get_cols(); j++)
			ASSERT_EQ(B(i, j), 0.0);

	B = ReluDerivate(A);

	cout << B << endl;
	
	for (size_t i = 0; i < B.get_rows(); i++)
		for (size_t j = 0; j < B.get_cols(); j++)
			ASSERT_EQ(B(i, j), 1.0);
}



TEST(TestAFuncFloat64, TestRelu)
{
	Mat<float64> A(10, 10, 15.0);

	Mat<float64> B = Relu(A);
	
	cout << B << endl;
	
	for (size_t i = 0; i < A.get_rows(); i++)
		for (size_t j = 0; j < A.get_cols(); j++)
			ASSERT_EQ(B(i, j), A(i, j));

	A.resize(10, 10, - 15.0);

	B = Relu(A);

	cout << B << endl;
	
	for (size_t i = 0; i < B.get_rows(); i++)
		for (size_t j = 0; j < B.get_cols(); j++)
			ASSERT_EQ(B(i, j), 0.0);

	B = ReluDerivate(A);

	cout << B << endl;
	
	for (size_t i = 0; i < B.get_rows(); i++)
		for (size_t j = 0; j < B.get_cols(); j++)
			ASSERT_EQ(B(i, j), 1.0);
}

TEST(TestAFuncVar, TestRelu)
{
	Var A(10, 10, (float64) 15.0);
	Var B = Relu(A);
	
	cout << B << endl;
	
	for (size_t i = 0; i < A.get_rows(); i++)
		for (size_t j = 0; j < A.get_cols(); j++)
			ASSERT_EQ(B.at_mf64(i, j), A.at_mf64(i, j));
	
	A.resize(10, 10, (float64) - 15.0);
	B = Relu(A);

	cout << B << endl;
	
	for (size_t i = 0; i < B.get_rows(); i++)
		for (size_t j = 0; j < B.get_cols(); j++)
			ASSERT_EQ(B.at_mf64(i, j), 0.0);

	B = ReluDerivate(A);

	cout << B << endl;
	
	for (size_t i = 0; i < B.get_rows(); i++)
		for (size_t j = 0; j < B.get_cols(); j++)
			ASSERT_EQ(B.at_mf64(i, j), 1.0);

	// For float 32
	A.resize(10, 10, (float32) 15.0);
	B = Relu(A);

	cout << B << endl;
	
	for (size_t i = 0; i < A.get_rows(); i++)
		for (size_t j = 0; j < A.get_cols(); j++)
			ASSERT_EQ(B.at_mf32(i, j), A.at_mf32(i, j));
	
	A.resize(10, 10, (float32) - 15.0);
	B = Relu(A);

	cout << B << endl;
	
	for (size_t i = 0; i < B.get_rows(); i++)
		for (size_t j = 0; j < B.get_cols(); j++)
			ASSERT_EQ(B.at_mf32(i, j), 0.0);

	B = ReluDerivate(A);

	cout << B << endl;
	
	for (size_t i = 0; i < B.get_rows(); i++)
		for (size_t j = 0; j < B.get_cols(); j++)
			ASSERT_EQ(B.at_mf32(i, j), 1.0);
}


TEST(TestAFuncFloat32, TestSigmoid)
{
	Mat<float32> A(10, 10, 15.0);

	Mat<float32> B = Sigmoid(A);
	
	cout << B << endl;

	for (size_t i = 0; i < B.get_rows(); i++)
		for (size_t j = 0; j < B.get_cols(); j++)
			EXPECT_NEAR(B(i, j), 1.0, 1e-6);
	
	A.resize(10, 10, 0.0);

	B = Sigmoid(A);

	cout << B << endl;

	for (size_t i = 0; i < B.get_rows(); i++)
		for (size_t j = 0; j < B.get_cols(); j++)
			EXPECT_NEAR(B(i, j), 0.5, 1e-6);

	
	A.resize(10, 10, -15.0);

	B = Sigmoid(A);

	cout << B << endl;

	for (size_t i = 0; i < B.get_rows(); i++)
		for (size_t j = 0; j < B.get_cols(); j++)
			EXPECT_NEAR(B(i, j), 0.0, 1e-6);
}

TEST(TestAFuncFloat64, TestSigmoid)
{
	Mat<float64> A(10, 10, 15.0);

	Mat<float64> B = Sigmoid(A);
	
	cout << B << endl;

	for (size_t i = 0; i < B.get_rows(); i++)
		for (size_t j = 0; j < B.get_cols(); j++)
			EXPECT_NEAR(B(i, j), 1.0, 1e-6);
	
	A.resize(10, 10, 0.0);

	B = Sigmoid(A);

	cout << B << endl;

	for (size_t i = 0; i < B.get_rows(); i++)
		for (size_t j = 0; j < B.get_cols(); j++)
			EXPECT_NEAR(B(i, j), 0.5, 1e-6);

	
	A.resize(10, 10, -15.0);

	B = Sigmoid(A);

	cout << B << endl;

	for (size_t i = 0; i < B.get_rows(); i++)
		for (size_t j = 0; j < B.get_cols(); j++)
			EXPECT_NEAR(B(i, j), 0.0, 1e-6);
}

TEST(TestAfuncVar, TestSigmoid)
{
	Var A(10, 10, (float64) 15.0);
	Var B = Sigmoid(A);
	
	cout << B << endl;
	
	for (size_t i = 0; i < A.get_rows(); i++)
		for (size_t j = 0; j < A.get_cols(); j++)
			EXPECT_NEAR(B.at_mf64(i, j), 1.0, 1e-6);

	A.resize(10, 10, (float64) - 0.0);
	B = Sigmoid(A);

	cout << B << endl;
	
	for (size_t i = 0; i < B.get_rows(); i++)
		for (size_t j = 0; j < B.get_cols(); j++)
			EXPECT_NEAR(B.at_mf64(i, j), 0.5, 1e-6);
	
	A.resize(10, 10, (float64) - 15.0);
	B = Sigmoid(A);

	cout << B << endl;
	
	for (size_t i = 0; i < B.get_rows(); i++)
		for (size_t j = 0; j < B.get_cols(); j++)
			EXPECT_NEAR(B.at_mf64(i, j), 0.0, 1e-6);

	B = SigmoidDerivate(A);

	cout << B << endl;
	
	for (size_t i = 0; i < B.get_rows(); i++)
		for (size_t j = 0; j < B.get_cols(); j++)
			EXPECT_NEAR(B.at_mf64(i, j), 0.0, 1e-6);

	// For float 32
	A.resize(10, 10, (float32) 15.0);
	B = Sigmoid(A);

	cout << B << endl;
	
	for (size_t i = 0; i < A.get_rows(); i++)
		for (size_t j = 0; j < A.get_cols(); j++)
			EXPECT_NEAR(B.at_mf32(i, j), 1.0, 1e-6);

	A.resize(10, 10, (float32) - 0.0);
	B = Sigmoid(A);

	cout << B << endl;
	
	for (size_t i = 0; i < B.get_rows(); i++)
		for (size_t j = 0; j < B.get_cols(); j++)
			EXPECT_NEAR(B.at_mf32(i, j), 0.5, 1e-6);
	
	A.resize(10, 10, (float32) - 15.0);
	B = Sigmoid(A);

	cout << B << endl;
	
	for (size_t i = 0; i < B.get_rows(); i++)
		for (size_t j = 0; j < B.get_cols(); j++)
			EXPECT_NEAR(B.at_mf32(i, j), 0.0, 1e-6);

	B = SigmoidDerivate(A);

	cout << B << endl;
	
	for (size_t i = 0; i < B.get_rows(); i++)
		for (size_t j = 0; j < B.get_cols(); j++)
			EXPECT_NEAR(B.at_mf32(i, j), 0.0, 1e-6);
}
