#include <iostream>
#include <gtest/gtest.h>
#include <chrono>

#include "cnet/dtypes.hpp"
#include "cnet/variable.hpp"

using namespace cnet;
using namespace variable;
using namespace dtypes;
using namespace std;

// cd ../build/ && make && make test ARGS='-V -R VarInit'

TEST(VariableTest, VarInit)
{
	Var V({4, 4}, (float32) 1.0);

	cout << V << endl;
	
	for (size_t i = 0; i < V.get_rows(); i++)
		for (size_t j = 0; j < V.get_cols(); j++)
			ASSERT_EQ(V.at_mf32(i, j), 1.0);

	V.resize({8, 8}, (float64) 2.0);

	cout << V << endl;

	for (size_t i = 0; i < V.get_rows(); i++)
		for (size_t j = 0; j < V.get_cols(); j++)
			ASSERT_EQ(V.at_mf64(i, j), 2.0);
	
	Mat<float32> M = {{1, 2, 3, 4, 5}};

	Var V2(M);

	cout << V2 << endl;

	for (size_t i = 0; i < V2.get_rows(); i++)
		for (size_t j = 0; j < V2.get_cols(); j++)
			ASSERT_EQ(V2.at_mf32(i, j), M(i, j));
	
}

TEST(VariableTest, VarMulOp)
{
	Var X({4, 1}, (float32) 2.0);
	Var W({4, 4}, (float32) 2.0);

	Var R = W * X;
	
	cout << R << endl;
	
	for (size_t i = 0; i < R.get_rows(); i++)
		for (size_t j = 0; j < R.get_cols(); j++)
			ASSERT_EQ(R.at_mf32(i, j), 16.0);
	W *= X;
	
	cout << W << endl;

	for (size_t i = 0; i < W.get_rows(); i++)
		for (size_t j = 0; j < W.get_cols(); j++)
			ASSERT_EQ(W.at_mf32(i, j), 16.0);

	X.resize({4, 1}, (float64) 2.0);
	W.resize({4, 4}, (float64) 2.0);
	
	R = W * X;
	
	cout << R << endl;
	
	for (size_t i = 0; i < R.get_rows(); i++)
		for (size_t j = 0; j < R.get_cols(); j++)
			ASSERT_EQ(R.at_mf64(i, j), 16.0);

	W *= X;

	cout << W << endl;
	
	for (size_t i = 0; i < W.get_rows(); i++)
		for (size_t j = 0; j < W.get_cols(); j++)
			ASSERT_EQ(W.at_mf64(i, j), 16.0);
}


TEST(VariableTest, VarAddOp)
{
	Var X({4, 4}, (float32) 2.0);
	Var W({4, 4}, (float32) 2.0);

	Var R = W + X;
	
	cout << R << endl;
	
	for (size_t i = 0; i < R.get_rows(); i++)
		for (size_t j = 0; j < R.get_cols(); j++)
			ASSERT_EQ(R.at_mf32(i, j), 4.0);
	W += X;
	
	cout << W << endl;

	for (size_t i = 0; i < W.get_rows(); i++)
		for (size_t j = 0; j < W.get_cols(); j++)
			ASSERT_EQ(W.at_mf32(i, j), 4.0);

	X.resize({4, 4}, (float64) 2.0);
	W.resize({4, 4}, (float64) 2.0);
	
	R = W + X;
	
	cout << R << endl;
	
	for (size_t i = 0; i < R.get_rows(); i++)
		for (size_t j = 0; j < R.get_cols(); j++)
			ASSERT_EQ(R.at_mf64(i, j), 4.0);

	W += X;

	cout << W << endl;
	
	for (size_t i = 0; i < W.get_rows(); i++)
		for (size_t j = 0; j < W.get_cols(); j++)
			ASSERT_EQ(W.at_mf64(i, j), 4.0);
}

TEST(VariableTest, VarSubOp)
{
	Var X({4, 4}, (float32) 2.0);
	Var W({4, 4}, (float32) 2.0);

	Var R = W - X;
	
	cout << R << endl;
	
	for (size_t i = 0; i < R.get_rows(); i++)
		for (size_t j = 0; j < R.get_cols(); j++)
			ASSERT_EQ(R.at_mf32(i, j), 0.0);
	W -= X;
	
	cout << W << endl;

	for (size_t i = 0; i < W.get_rows(); i++)
		for (size_t j = 0; j < W.get_cols(); j++)
			ASSERT_EQ(W.at_mf32(i, j), 0.0);

	X.resize({4, 4}, (float64) 2.0);
	W.resize({4, 4}, (float64) 2.0);
	
	R = W - X;
	
	cout << R << endl;
	
	for (size_t i = 0; i < R.get_rows(); i++)
		for (size_t j = 0; j < R.get_cols(); j++)
			ASSERT_EQ(R.at_mf64(i, j), 0.0);

	W -= X;

	cout << W << endl;
	
	for (size_t i = 0; i < W.get_rows(); i++)
		for (size_t j = 0; j < W.get_cols(); j++)
			ASSERT_EQ(W.at_mf64(i, j), 0.0);
}


TEST(VariableTest, VarWiseElementProductOp)
{
	Var X({4, 4}, (float32) 2.0);
	Var W({4, 4}, (float32) 2.0);

	Var R = W ^ X;
	
	cout << R << endl;
	
	for (size_t i = 0; i < R.get_rows(); i++)
		for (size_t j = 0; j < R.get_cols(); j++)
			ASSERT_EQ(R.at_mf32(i, j), 4.0);
	W ^= X;
	
	cout << W << endl;

	for (size_t i = 0; i < W.get_rows(); i++)
		for (size_t j = 0; j < W.get_cols(); j++)
			ASSERT_EQ(W.at_mf32(i, j), 4.0);

	X.resize({4, 4}, (float64) 2.0);
	W.resize({4, 4}, (float64) 2.0);
	
	R = W ^ X;
	
	cout << R << endl;
	
	for (size_t i = 0; i < R.get_rows(); i++)
		for (size_t j = 0; j < R.get_cols(); j++)
			ASSERT_EQ(R.at_mf64(i, j), 4.0);

	W ^= X;

	cout << W << endl;
	
	for (size_t i = 0; i < W.get_rows(); i++)
		for (size_t j = 0; j < W.get_cols(); j++)
			ASSERT_EQ(W.at_mf64(i, j), 4.0);
}

TEST(VariableTest, VarScalarProduct)
{
	Var X({4, 4}, (float32) 2.0);

	Var R = X * (float32) 2.0;

	for (size_t i = 0; i < R.get_rows(); i++)
		for (size_t j = 0; j < R.get_cols(); j++)
			ASSERT_EQ(R.at_mf32(i, j), 4.0);

	X *= (float32) 2.0;

	cout << X << endl;
	
	for (size_t i = 0; i < X.get_rows(); i++)
		for (size_t j = 0; j < X.get_cols(); j++)
			ASSERT_EQ(X.at_mf32(i, j), 4.0);

	X.resize({4, 4}, (float64) 2.0);

	R = X * (float64) 2.0;
	
	for (size_t i = 0; i < R.get_rows(); i++)
		for (size_t j = 0; j < R.get_cols(); j++)
			ASSERT_EQ(R.at_mf64(i, j), 4.0);


	X *= (float64) 2.0;
	
	for (size_t i = 0; i < X.get_rows(); i++)
		for (size_t j = 0; j < X.get_cols(); j++)
			ASSERT_EQ(X.at_mf64(i, j), 4.0);
}

