// To have output: make test ARGS=-V

// Run valgrind to check memory issues
// valgrind --leak-check=full --track-origins=yes -s --show-leak-kinds=all --max-stackframe=62179200

#include <iostream>
#include <gtest/gtest.h>
#include <chrono>
#include <random>

#include <omp.h>

#include "cnet/dtypes.hpp"
#include "cnet/variable.hpp"
#include "cnet/afuncs.hpp"
#include "cnet/layers.hpp"
#include "cnet/cfuncs.hpp"

#include <fenv.h>



using namespace std;
using namespace cnet;
using namespace variable;
using namespace cfuncs;
using namespace afuncs;
using namespace layers;

TEST(LayerTestFloat32, DenseLayerIinit)
{
	Dense D(4);
	
	ASSERT_EQ(D.get_dtype(), FLOAT_32_DTYPE);
	ASSERT_EQ(D.get_out_size(), 4);
	ASSERT_EQ(D.get_in_size(), 0);
	
	ASSERT_EQ(D.get_units(), 4);
	
	cout << D.get_out_shape() << endl;
	cout << D << endl;
}

TEST(LayerTestFloat32, DenseLayerBuild)
{
	Dense D(4);
	
	cout << D << endl;
	
	D.build(2);

	ASSERT_EQ(D.get_weights(), 8);
	ASSERT_EQ(D.get_biases(), 4);
	
	ASSERT_TRUE(D.is_built());
	ASSERT_TRUE(D.use_bias());

	ASSERT_EQ(D.get_weights_shape().rows, 4);
	ASSERT_EQ(D.get_weights_shape().cols, 2);
	
	ASSERT_EQ(D.get_biases_shape().rows, 4);
	ASSERT_EQ(D.get_biases_shape().cols, 1);
	
	cout << D << endl;
}


TEST(LayerTestFloat32, DenseLayerBuildRandomly)
{
	Dense D(4);
	
	cout << D << endl;
	
	D.build(2).rand_uniform_range(15.0, 15.0);

	// Run over the matrices
	const Mat<float32> &W = D.get_cmat_weights().get_cmf32();
	for (size_t i = 0; i < W.get_rows(); i++)
		for (size_t j = 0; j < W.get_cols(); j++)
			ASSERT_EQ(W(i, j), 15.0);

	const Mat<float32> &B = D.get_cmat_biases().get_cmf32();
	for (size_t i = 0; i < B.get_cols(); i++)
		for (size_t j = 0; j < B.get_cols(); j++)
			ASSERT_EQ(B(i, j), 15.0);
	
	cout << D << endl;
}


TEST(LayerTestFloat32, DenseLayerBuildAnotherAFunc)
{
	Dense D(4, "Sigmoid");
	
	cout << D << endl;
	
	D.build(2).rand_uniform_range(0.0, 0.0);
	
	ASSERT_EQ(D.get_afunc_name(), "Sigmoid");
	
	cout << D << endl;
}

TEST(LayerTestFloat32, DenseLayerFeedFoorwardLinear)
{
	Dense D(4);

	cout << "Linear activation function" << endl;;
	
	D.set_afunc("Linear").build(2).rand_uniform_range(15.0, 15.0);

	Input I_pos(Mat<float32>({
				{1.0},
				{1.0}
			}));

	Input I_neg(Mat<float32>({
				{-1.0},
				{-1.0}
			}));

	Output O = D(I_pos);

	Mat<float32> &M = O.get_mf32();
	for (size_t i = 0; i < M.get_rows(); i++)
		for (size_t j = 0; j < M.get_cols(); j++)
			ASSERT_EQ(M(i, j), 45.0);
	
	cout << O << endl;

	O = D(I_neg);

	M = O.get_mf32();
	for (size_t i = 0; i < M.get_rows(); i++)
		for (size_t j = 0; j < M.get_cols(); j++)
			ASSERT_EQ(M(i, j), -15.0);
	cout << O << endl;

	D.rand_uniform_range(0.0, 0.0);

	O = D(I_pos);

	M = O.get_mf32();
	for (size_t i = 0; i < M.get_rows(); i++)
		for (size_t j = 0; j < M.get_cols(); j++)
			ASSERT_EQ(M(i, j), 0.0);
	cout << O << endl;
}

TEST(LayerTestFloat32, DenseLayerFeedFoorwardSigmoid)
{
	Dense D(4);

	cout << "Sigmoid activation function" << endl;;
	
	D.set_afunc("Sigmoid").build(2).rand_uniform_range(15.0, 15.0);

	Input I_pos(Mat<float32>({
				{1.0},
				{1.0}
			}));

	Input I_neg(Mat<float32>({
				{-1.0},
				{-1.0}
			}));


	Output O = D(I_pos);

	Mat<float32> &M = O.get_mf32();
	for (size_t i = 0; i < M.get_rows(); i++)
		for (size_t j = 0; j < M.get_cols(); j++)
			ASSERT_EQ(M(i, j), 1.0);
	cout << O << endl;
	
	O = D(I_neg);
	
	M = O.get_mf32();
	for (size_t i = 0; i < M.get_rows(); i++)
		for (size_t j = 0; j < M.get_cols(); j++)
			EXPECT_NEAR(M(i, j), 0.0, 1e-6);
	cout << O << endl;

	D.rand_uniform_range(0.0, 0.0);

	O = D(I_pos);
	
	M = O.get_mf32();
	for (size_t i = 0; i < M.get_rows(); i++)
		for (size_t j = 0; j < M.get_cols(); j++)
			ASSERT_EQ(M(i, j), 0.5);
	cout << O << endl;
}


TEST(LayerTestFloat32, DenseLayerFeedFoorwardRelu)
{
	Dense D(4);

	cout << "Relu activation function" << endl;;
	
	D.set_afunc("Relu").build(2).rand_uniform_range(15.0, 15.0);

	Input I_pos(Mat<float32>({
				{1.0},
				{1.0}
			}));

	Input I_neg(Mat<float32>({
				{-1.0},
				{-1.0}
			}));


	Output O = D(I_pos);

	Mat<float32> &M = O.get_mf32();
	for (size_t i = 0; i < M.get_rows(); i++)
		for (size_t j = 0; j < M.get_cols(); j++)
			ASSERT_EQ(M(i, j), 45.0);

	O = D(I_neg);
	
	M = O.get_mf32();
	for (size_t i = 0; i < M.get_rows(); i++)
		for (size_t j = 0; j < M.get_cols(); j++)
			ASSERT_EQ(M(i, j), 0.0);
	cout << O << endl;
	
}


TEST(LayerTestFloat32, DenseLayerHotFeedFoorward)
{
	Dense D(4);
	
	D.set_afunc("Relu");
	
	Input I_pos(Mat<float32>({
				{1.0},
				{1.0}
			}));

	Input I_neg(Mat<float32>({
				{-1.0},
				{-1.0}
			}));
	
	Output O = D(I_pos);
	
	ASSERT_EQ(D.get_weights(), 8);
	ASSERT_EQ(D.get_biases(), 4);
	ASSERT_TRUE(D.is_built());

	ASSERT_EQ(D.get_weights_shape().rows, 4);
	ASSERT_EQ(D.get_weights_shape().cols, 2);
	
	ASSERT_EQ(D.get_biases_shape().rows, 4);
	ASSERT_EQ(D.get_biases_shape().cols, 1);
	
	cout << D << endl;
}


TEST(LayerTestFloat32, DenseLayerFeedForwardMatrices)
{
	Dense D(4);
	
	D.set_afunc("Relu").build(2).rand_uniform_range(15.0, 15.0);
	
	Mat<float32> input_pos = {
		{1.0},
		{1.0}
	};

	Mat<float32> input_neg = {
		{-1.0},
		{-1.0}
	};

	Mat<float32> output = D(input_pos);
	
	for (size_t i = 0; i < output.get_rows(); i++)
		for (size_t j = 0; j < output.get_cols(); j++)
			ASSERT_EQ(output(i, j), 45.0);
	
	output = D(input_neg);

	for (size_t i = 0; i < output.get_rows(); i++)
		for (size_t j = 0; j < output.get_cols(); j++)
			ASSERT_EQ(output(i, j), 0.0);
	
	cout << D << endl;
}



// Test with float 64

TEST(LayerTestFloat64, DenseLayerIinit)
{
	Dense D(4, FLOAT_64_DTYPE);
	
	ASSERT_EQ(D.get_dtype(), FLOAT_64_DTYPE);
	ASSERT_EQ(D.get_out_size(), 4);
	ASSERT_EQ(D.get_in_size(), 0);
	
	ASSERT_EQ(D.get_units(), 4);
	
	cout << D.get_out_shape() << endl;
	cout << D << endl;
}

TEST(LayerTestFloat64, DenseLayerBuild)
{
	Dense D(4, FLOAT_64_DTYPE);
	
	cout << D << endl;
	
	D.build(2);

	ASSERT_EQ(D.get_weights(), 8);
	ASSERT_EQ(D.get_biases(), 4);
	
	ASSERT_TRUE(D.is_built());
	ASSERT_TRUE(D.use_bias());

	ASSERT_EQ(D.get_weights_shape().rows, 4);
	ASSERT_EQ(D.get_weights_shape().cols, 2);
	
	ASSERT_EQ(D.get_biases_shape().rows, 4);
	ASSERT_EQ(D.get_biases_shape().cols, 1);
	
	cout << D << endl;
}


TEST(LayerTestFloat64, DenseLayerBuildRandomly)
{
	Dense D(4, FLOAT_64_DTYPE);
	
	cout << D << endl;
	
	D.build(2).rand_uniform_range(15.0, 15.0);

	// Run over the matrices
	const Mat<float64> &W = D.get_cmat_weights().get_cmf64();
	for (size_t i = 0; i < W.get_rows(); i++)
		for (size_t j = 0; j < W.get_cols(); j++)
			ASSERT_EQ(W(i, j), 15.0);

	const Mat<float64> &B = D.get_cmat_biases().get_cmf64();
	for (size_t i = 0; i < B.get_cols(); i++)
		for (size_t j = 0; j < B.get_cols(); j++)
			ASSERT_EQ(B(i, j), 15.0);
	
	cout << D << endl;
}


TEST(LayerTestFloat64, DenseLayerBuildAnotherAFunc)
{
	Dense D(4, "Sigmoid", FLOAT_64_DTYPE);
	
	cout << D << endl;
	
	D.build(2).rand_uniform_range(0.0, 0.0);
	
	ASSERT_EQ(D.get_afunc_name(), "Sigmoid");
	
	cout << D << endl;
}

TEST(LayerTestFloat64, DenseLayerFeedFoorwardLinear)
{
	Dense D(4, FLOAT_64_DTYPE);

	cout << "Linear activation function" << endl;;
	
	D.set_afunc("Linear").build(2).rand_uniform_range(15.0, 15.0);

	Input I_pos(Mat<float64>({
				{1.0},
				{1.0}
			}));

	Input I_neg(Mat<float64>({
				{-1.0},
				{-1.0}
			}));

	Output O = D(I_pos);

	Mat<float64> &M = O.get_mf64();
	for (size_t i = 0; i < M.get_rows(); i++)
		for (size_t j = 0; j < M.get_cols(); j++)
			ASSERT_EQ(M(i, j), 45.0);
	
	cout << O << endl;

	O = D(I_neg);

	M = O.get_mf64();
	for (size_t i = 0; i < M.get_rows(); i++)
		for (size_t j = 0; j < M.get_cols(); j++)
			ASSERT_EQ(M(i, j), -15.0);
	cout << O << endl;

	D.rand_uniform_range(0.0, 0.0);

	O = D(I_pos);

	M = O.get_mf64();
	for (size_t i = 0; i < M.get_rows(); i++)
		for (size_t j = 0; j < M.get_cols(); j++)
			ASSERT_EQ(M(i, j), 0.0);
	cout << O << endl;
}

TEST(LayerTestFloat64, DenseLayerFeedFoorwardSigmoid)
{
	Dense D(4, FLOAT_64_DTYPE);

	cout << "Sigmoid activation function" << endl;;
	
	D.set_afunc("Sigmoid").build(2).rand_uniform_range(15.0, 15.0);

	Input I_pos(Mat<float64>({
				{1.0},
				{1.0}
			}));

	Input I_neg(Mat<float64>({
				{-1.0},
				{-1.0}
			}));


	Output O = D(I_pos);

	Mat<float64> &M = O.get_mf64();
	for (size_t i = 0; i < M.get_rows(); i++)
		for (size_t j = 0; j < M.get_cols(); j++)
			ASSERT_EQ(M(i, j), 1.0);
	cout << O << endl;
	
	O = D(I_neg);
	
	M = O.get_mf64();
	for (size_t i = 0; i < M.get_rows(); i++)
		for (size_t j = 0; j < M.get_cols(); j++)
			EXPECT_NEAR(M(i, j), 0.0, 1e-6);
	cout << O << endl;

	D.rand_uniform_range(0.0, 0.0);

	O = D(I_pos);
	
	M = O.get_mf64();
	for (size_t i = 0; i < M.get_rows(); i++)
		for (size_t j = 0; j < M.get_cols(); j++)
			ASSERT_EQ(M(i, j), 0.5);
	cout << O << endl;
}


TEST(LayerTestFloat64, DenseLayerFeedFoorwardRelu)
{
	Dense D(4, FLOAT_64_DTYPE);

	cout << "Relu activation function" << endl;;
	
	D.set_afunc("Relu").build(2).rand_uniform_range(15.0, 15.0);

	Input I_pos(Mat<float64>({
				{1.0},
				{1.0}
			}));

	Input I_neg(Mat<float64>({
				{-1.0},
				{-1.0}
			}));


	Output O = D(I_pos);

	Mat<float64> &M = O.get_mf64();
	for (size_t i = 0; i < M.get_rows(); i++)
		for (size_t j = 0; j < M.get_cols(); j++)
			ASSERT_EQ(M(i, j), 45.0);

	O = D(I_neg);
	
	M = O.get_mf64();
	for (size_t i = 0; i < M.get_rows(); i++)
		for (size_t j = 0; j < M.get_cols(); j++)
			ASSERT_EQ(M(i, j), 0.0);
	cout << O << endl;
	
}


TEST(LayerTestFloat64, DenseLayerHotFeedFoorward)
{
	Dense D(4, FLOAT_64_DTYPE);
	
	D.set_afunc("Relu");
	
	Input I_pos(Mat<float64>({
				{1.0},
				{1.0}
			}));

	Input I_neg(Mat<float64>({
				{-1.0},
				{-1.0}
			}));
	
	Output O = D(I_pos);
	
	ASSERT_EQ(D.get_weights(), 8);
	ASSERT_EQ(D.get_biases(), 4);
	ASSERT_TRUE(D.is_built());

	ASSERT_EQ(D.get_weights_shape().rows, 4);
	ASSERT_EQ(D.get_weights_shape().cols, 2);
	
	ASSERT_EQ(D.get_biases_shape().rows, 4);
	ASSERT_EQ(D.get_biases_shape().cols, 1);
	
	cout << D << endl;
}


TEST(LayerTestFloat65, DenseLayerFeedForwardMatrices)
{
	Dense D(4, FLOAT_64_DTYPE);
	
	D.set_afunc("Relu").build(2).rand_uniform_range(15.0, 15.0);
	
	Mat<float64> input_pos = {
		{1.0},
		{1.0}
	};

	Mat<float64> input_neg = {
		{-1.0},
		{-1.0}
	};

	Mat<float64> output = D(input_pos);
	
	for (size_t i = 0; i < output.get_rows(); i++)
		for (size_t j = 0; j < output.get_cols(); j++)
			ASSERT_EQ(output(i, j), 45.0);
	
	output = D(input_neg);

	for (size_t i = 0; i < output.get_rows(); i++)
		for (size_t j = 0; j < output.get_cols(); j++)
			ASSERT_EQ(output(i, j), 0.0);
	
	cout << D << endl;
}



// Test fit layer

TEST(LayerTestFloat32, TestFitSmallLayer)
{
	constexpr size_t N = 4;
	
	// Train a perceptron of and
	Dense D(1, "Sigmoid");

	// To initialize in zeros
	D.build(2).rand_uniform_range(0.0, 0.0);
	
	cout << D<< endl;

	Mat<float32> X[N] = {
		{
			{0.0},
			{0.0}
		},
		{
			{0.0},
			{1.0}
		},
		{
			{1.0},
			{0.0}
		},
		{
			{1.0},
			{1.0}
		}
	};

	// Results
	// 51: Epoch: 142 MSE: 9.93309e-05
	// 51: Dense=(
	// 51: Weights=(Mat=(
	// 51: [[16.2869	8.53264]],
	// 51: shape=(rows=1, cols=2), dtype=float32, addrs=0x7ffd41a48c60),
	// 51: dtype=float32, addrs=0x7ffd41a48c30),
	// 51: Biases=(Mat=(
	// 51: [[-20.4841]],
	// 51: shape=(rows=1, cols=1), dtype=float32, addrs=0x7ffd41a48ca8),
	// 51: dtype=float32, addrs=0x7ffd41a48c78),
	// 51: built=true, in=(shape=(rows=2, cols=1)), out=(shape=(rows=1, cols=1)), units=1,
	// dtype=float32, use_bias=true, activation=Sigmoid)

	// And Gate
	// Mat<float32> Y[4] = {
	// 	{{0.0}},
	// 	{{0.0}},
	// 	{{0.0}},
	// 	{{1.0}}
	// };


	// 51: Epoch: 7 MSE: 9.33788e-05
	// 51: Dense=(
	// 51: Weights=(Mat=(
	// 51: [[11.7483	10.9118]],
	// 51: shape=(rows=1, cols=2), dtype=float32, addrs=0x7ffc9d608f90),
	// 51: dtype=float32, addrs=0x7ffc9d608f60),
	// 51: Biases=(Mat=(
	// 51: [[-3.94642]],
	// 51: shape=(rows=1, cols=1), dtype=float32, addrs=0x7ffc9d608fd8),
	// 51: dtype=float32, addrs=0x7ffc9d608fa8),
	// 51: built=true, in=(shape=(rows=2, cols=1)), out=(shape=(rows=1, cols=1)), units=1,
	// dtype=float32, use_bias=true, activation=Sigmoid)
	
	// Or gate
	// Mat<float32> Y[4] = {
	// 	{{0.0}},
	// 	{{1.0}},
	// 	{{1.0}},
	// 	{{1.0}}
	// };


	// 51: Epoch: 33 MSE: 5.09452e-06
	// 51: Dense=(
	// 51: Weights=(Mat=(
	// 51: [[-11.5314	-12.8734]],
	// 51: shape=(rows=1, cols=2), dtype=float32, addrs=0x7fffd7116550),
	// 51: dtype=float32, addrs=0x7fffd7116520),
	// 51: Biases=(Mat=(
	// 51: [[18.7219]],
	// 51: shape=(rows=1, cols=1), dtype=float32, addrs=0x7fffd7116598),
	// 51: dtype=float32, addrs=0x7fffd7116568),
	// 51: built=true, in=(shape=(rows=2, cols=1)), out=(shape=(rows=1, cols=1)), units=1,
	// dtype=float32, use_bias=true, activation=Sigmoid)
	
	// Nand gate
	// Mat<float32> Y[N] = {
	// 	{{1.0}},
	// 	{{1.0}},
	// 	{{1.0}},
	// 	{{0.0}},
	// };


	// 51: Epoch: 7 MSE: 9.33727e-05
	// 51: Dense=(
	// 51: Weights=(Mat=(
	// 51: [[-11.7483	-10.9118]],
	// 51: shape=(rows=1, cols=2), dtype=float32, addrs=0x7ffd4c642880),
	// 51: dtype=float32, addrs=0x7ffd4c642850),
	// 51: Biases=(Mat=(
	// 51: [[3.94645]],
	// 51: shape=(rows=1, cols=1), dtype=float32, addrs=0x7ffd4c6428c8),
	// 51: dtype=float32, addrs=0x7ffd4c642898),
	// 51: built=true, in=(shape=(rows=2, cols=1)), out=(shape=(rows=1, cols=1)), units=1,
	// dtype=float32, use_bias=true, activation=Sigmoid)
	
	// Nor gate
	Mat<float32> Y[4] = {
		{{1.0}},
		{{0.0}},
		{{0.0}},
		{{0.0}}
	};

	constexpr size_t epochs = 7;
	constexpr double lr = 100.0;
	
	Mat<float32> A[N];
	for (size_t e = 1; e <= epochs; e++) {
		cout << "---------------------\n" << endl;
		for (size_t i = 0; i < N; i++) {
			A[i] = D(X[i]);
		
			// Use Mse Derivate
			D.fit(MseDerivate(A[i], Y[i], N), X[i], lr);
		}
		
		// cout << D.get_cmat_weights() << endl;
		cout << "Epoch: " << e << " MSE: " << Mse(A, Y, N).grand_sum() << endl;
	}

	
	cout << D << endl;
	
	for (size_t i = 0; i < N; i++) {
		cout << "---------------------\n" << endl;
		A[i] = D(X[i]);
		cout << "Outpu=" << endl;
		cout << Y[i] << endl;
		cout << "Predicted=" << endl;
		cout << A[i] << endl;
	}
	// Test the precision
	for (size_t k = 0; k < N; k++)
		for (size_t i = 0; i < A[k].get_rows(); i++)
			for (size_t j = 0; j < A[k].get_cols(); j++)
				EXPECT_NEAR(A[k](i, j), Y[k](i, j), 1e-1);

}

TEST(LayerTestFloat64, TestFitSmallLayer)
{
	constexpr size_t N = 4;
	
	// Train a perceptron of and
	Dense D(1, "Sigmoid", FLOAT_64_DTYPE);

	// To initialize in zeros
	D.build(2).rand_uniform_range(0.0, 0.0);
	cout << D << endl;

	Mat<float64> X[N] = {
		{
			{0.0},
			{0.0}
		},
		{
			{0.0},
			{1.0}
		},
		{
			{1.0},
			{0.0}
		},
		{
			{1.0},
			{1.0}
		}
	};
	
	// Results
	// 52: Epoch: 32 MSE: 9.93306e-05
	// 52: Dense=(
	// 52: Weights=(Mat=(
	// 52: [[12.6266	9.4671]],
	// 52: shape=(rows=1, cols=2), dtype=float64, addrs=0x7ffd7b518440),
	// 52: dtype=float64, addrs=0x7ffd7b518410),
	// 52: Biases=(Mat=(
	// 52: [[-18.1219]],
	// 52: shape=(rows=1, cols=1), dtype=float64, addrs=0x7ffd7b518488),
	// 52: dtype=float64, addrs=0x7ffd7b518458),
	// 52: built=true, in=(shape=(rows=2, cols=1)), out=(shape=(rows=1, cols=1)), units=1, dtype=float64,
	// use_bias=true, activation=Sigmoid)

	// And Gate
	// Mat<float64> Y[4] = {
	// 	{{0.0}},
	// 	{{0.0}},
	// 	{{0.0}},
	// 	{{1.0}}
	// };

	// 52: Epoch: 7 MSE: 9.34343e-05
	// 52: Dense=(
	// 52: Weights=(Mat=(
	// 52: [[11.7485	10.9118]],
	// 52: shape=(rows=1, cols=2), dtype=float64, addrs=0x7ffd7f6c1e80),
	// 52: dtype=float64, addrs=0x7ffd7f6c1e50),
	// 52: Biases=(Mat=(
	// 52: [[-3.94612]],
	// 52: shape=(rows=1, cols=1), dtype=float64, addrs=0x7ffd7f6c1ec8),
	// 52: dtype=float64, addrs=0x7ffd7f6c1e98),
	// 52: built=true, in=(shape=(rows=2, cols=1)), out=(shape=(rows=1, cols=1)), units=1,
	// dtype=float64, use_bias=true, activation=Sigmoid)
	
	// Mat<float64> Y[4] = {
	// 	{{0.0}},
	// 	{{1.0}},
	// 	{{1.0}},
	// 	{{1.0}}
	// };


	// 52: Epoch: 32 MSE: 9.93306e-05
	// 52: Dense=(
	// 52: Weights=(Mat=(
	// 52: [[-12.6266	-9.4671]],
	// 52: shape=(rows=1, cols=2), dtype=float64, addrs=0x7ffe6b6ebb30),
	// 52: dtype=float64, addrs=0x7ffe6b6ebb00),
	// 52: Biases=(Mat=(
	// 52: [[18.1219]],
	// 52: shape=(rows=1, cols=1), dtype=float64, addrs=0x7ffe6b6ebb78),
	// 52: dtype=float64, addrs=0x7ffe6b6ebb48),
	// 52: built=true, in=(shape=(rows=2, cols=1)), out=(shape=(rows=1, cols=1)), units=1,
	// dtype=float64, use_bias=true, activation=Sigmoid)
	
	// Nand gate
	// Mat<float64> Y[N] = {
	// 	{{1.0}},
	// 	{{1.0}},
	// 	{{1.0}},
	// 	{{0.0}},
	// };


	// 52: Epoch: 7 MSE: 9.34343e-05
	// 52: Dense=(
	// 52: Weights=(Mat=(
	// 52: [[-11.7485	-10.9118]],
	// 52: shape=(rows=1, cols=2), dtype=float64, addrs=0x7ffefeddfc60),
	// 52: dtype=float64, addrs=0x7ffefeddfc30),
	// 52: Biases=(Mat=(
	// 52: [[3.94612]],
	// 52: shape=(rows=1, cols=1), dtype=float64, addrs=0x7ffefeddfca8),
	// 52: dtype=float64, addrs=0x7ffefeddfc78),
	// 52: built=true, in=(shape=(rows=2, cols=1)), out=(shape=(rows=1, cols=1)), units=1,
	// dtype=float64, use_bias=true, activation=Sigmoid)
	
	// Nor gate
	Mat<float64> Y[4] = {
		{{1.0}},
		{{0.0}},
		{{0.0}},
		{{0.0}}
	};

	constexpr size_t epochs = 7;
	constexpr double lr = 100.0;
	
	Mat<float64> A[N];
	for (size_t e = 1; e <= epochs; e++) {
		cout << "---------------------\n" << endl;
		for (size_t i = 0; i < N; i++) {
			A[i] = D(X[i]);
		
			// Use Mse Derivate
			D.fit(MseDerivate(A[i], Y[i], N), X[i], lr);
		}
		
		// cout << D.get_cmat_weights() << endl;
		cout << "Epoch: " << e << " MSE: " << Mse(A, Y, N).grand_sum() << endl;
	}

	cout << D << endl;
	for (size_t i = 0; i < N; i++) {
		cout << "---------------------\n" << endl;
		A[i] = D(X[i]);
		cout << "Outpu=" << endl;
		cout << Y[i] << endl;
		cout << "Predicted=" << endl;
		cout << A[i] << endl;
	}

	// Test the precision
	for (size_t k = 0; k < N; k++)
		for (size_t i = 0; i < A[k].get_rows(); i++)
			for (size_t j = 0; j < A[k].get_cols(); j++)
				EXPECT_NEAR(A[k](i, j), Y[k](i, j), 1e-1);
}


// Test your big dense
TEST(LayerTestFloat32, TestFitMidLayer) {
	// For the logic gates Two neurons proccessing 2 inputs 5 outputs
	constexpr size_t epochs = 64;
	constexpr double lr = 100.0;
	constexpr size_t N = 4;
	
	Dense D(5, "Sigmoid");
	
	D.build(2).rand_uniform_range(0.0, 0.0);
	
	// 52: Epoch: 64 MSE: 0.25067
	// 52: Dense=(
	// 52: Weights=(Mat=(
	// 52: [[9.03573	10.2552]
	// 52:  [11.7494	10.917]
	// 52:  [-15.8663	-7.93517]
	// 52:  [-11.7494	-10.917]
	// 52:  [7.83464	9.19671]],
	// 52: shape=(rows=5, cols=2), dtype=float32, addrs=0x7ffe36306eb0),
	// 52: dtype=float32, addrs=0x7ffe36306e80),
	// 52: Biases=(Mat=(
	// 52: [[-14.6192]
	// 52:  [-4.5013]
	// 52:  [19.7544]
	// 52:  [4.50129]
	// 52:  [-3.79126]],
	// 52: shape=(rows=5, cols=1), dtype=float32, addrs=0x7ffe36306ef8),
	// 52: dtype=float32, addrs=0x7ffe36306ec8),
	// 52: built=true, in=(shape=(rows=2, cols=1)), out=(shape=(rows=5, cols=1)),
	// units=5, dtype=float32, use_bias=true, activation=Sigmoid)
	
	Mat<float32> X[N] = {
		{{0.0}, {0.0}},	// One column and two rows
		{{0.0}, {1.0}},
		{{1.0}, {0.0}},
		{{1.0}, {1.0}}
	};

	Mat<float32> Y[N] = {
		// And, Or,    Nand,  Nor,   Xor
		{{0.0}, {0.0}, {1.0}, {1.0}, {0.0}}, 
		{{0.0}, {1.0}, {1.0}, {0.0}, {1.0}}, 
		{{0.0}, {1.0}, {1.0}, {0.0}, {1.0}}, 
		{{1.0}, {1.0}, {0.0}, {0.0}, {0.0}}
	};

	
	Mat<float32> A[N];
	for (size_t e = 1; e <= epochs; e++) {
		cout << "---------------------\n" << endl;
		for (size_t i = 0; i < N; i++) {
			A[i] = D(X[i]);
			D.fit(MseDerivate(A[i], Y[i], N), X[i], lr);
		}
		
		cout << "Epoch: " << e << " MSE: " << Mse(A, Y, N).grand_sum() << endl;
	}

	cout << D << endl;
	for (size_t i = 0; i < N; i++) {
		cout << "---------------------\n" << endl;
		A[i] = D(X[i]);
		cout << "Outpu=" << endl;
		cout << Y[i] << endl;
		cout << "Predicted=" << endl;
		cout << A[i] << endl;
	}

	// Test the precision expect in xor 
	for (size_t k = 0; k < N - 1; k++)
		for (size_t i = 0; i < A[k].get_rows(); i++)
			for (size_t j = 0; j < A[k].get_cols(); j++)
				EXPECT_NEAR(A[k](i, j), Y[k](i, j), 1e-1);
}



// Test your big dense
TEST(LayerTestFloat64, TestFitMidLayer) {
	// For the logic gates Two neurons proccessing 2 inputs 5 outputs
	constexpr size_t epochs = 64;
	constexpr double lr = 100.0;
	constexpr size_t N = 4;
	
	Dense D(5, "Sigmoid", FLOAT_64_DTYPE);
	
	D.build(2).rand_uniform_range(0.0, 0.0);

	// 54: Epoch: 64 MSE: 0.25033
	// 54: Dense=(
	// 54: Weights=(Mat=(
	// 54: [[12.8368	9.72358]
	// 54:  [11.7495	10.917]
	// 54:  [-12.8368	-9.72358]
	// 54:  [-11.7495	-10.917]
	// 54:  [7.85522	9.22704]],
	// 54: shape=(rows=5, cols=2), dtype=float64, addrs=0x7ffdb4f804c0),
	// 54: dtype=float64, addrs=0x7ffdb4f80490),
	// 54: Biases=(Mat=(
	// 54: [[-17.9117]
	// 54:  [-4.50124]
	// 54:  [17.9117]
	// 54:  [4.50124]
	// 54:  [-3.80204]],
	// 54: shape=(rows=5, cols=1), dtype=float64, addrs=0x7ffdb4f80508),
	// 54: dtype=float64, addrs=0x7ffdb4f804d8),
	// 54: built=true, in=(shape=(rows=2, cols=1)), out=(shape=(rows=5, cols=1)),
	// units=5, dtype=float64, use_bias=true, activation=Sigmoid)
	
	Mat<float64> X[N] = {
		{{0.0}, {0.0}},	// One column and two rows
		{{0.0}, {1.0}},
		{{1.0}, {0.0}},
		{{1.0}, {1.0}}
	};

	Mat<float64> Y[N] = {
		// And, Or,    Nand,  Nor,   Xor
		{{0.0}, {0.0}, {1.0}, {1.0}, {0.0}}, 
		{{0.0}, {1.0}, {1.0}, {0.0}, {1.0}}, 
		{{0.0}, {1.0}, {1.0}, {0.0}, {1.0}}, 
		{{1.0}, {1.0}, {0.0}, {0.0}, {0.0}}
	};

	
	Mat<float64> A[N];
	for (size_t e = 1; e <= epochs; e++) {
		cout << "---------------------\n" << endl;
		for (size_t i = 0; i < N; i++) {
			A[i] = D(X[i]);
			D.fit(MseDerivate(A[i], Y[i], N), X[i], lr);
		}
		
		cout << "Epoch: " << e << " MSE: " << Mse(A, Y, N).grand_sum() << endl;
	}

	cout << D << endl;
	for (size_t i = 0; i < N; i++) {
		cout << "---------------------\n" << endl;
		A[i] = D(X[i]);
		cout << "Outpu=" << endl;
		cout << Y[i] << endl;
		cout << "Predicted=" << endl;
		cout << A[i] << endl;
	}

	// Test the precision expect in xor 
	for (size_t k = 0; k < N - 1; k++)
		for (size_t i = 0; i < A[k].get_rows(); i++)
			for (size_t j = 0; j < A[k].get_cols(); j++)
				EXPECT_NEAR(A[k](i, j), Y[k](i, j), 1e-1);
}


TEST(LayerTestFloat32, TestSimpleLineaRegression)
{
	// We are going to try to teach the model to multiply by 2
	// Predict f(x) = 2.0 * x, in a range of [-10, 10]

	// Generate data for the model
	
	// Seed the random number generator
	std::random_device rd;
	std::mt19937 generator(rd());
	std::uniform_real_distribution<float> distribution(- 10, 10);
	
	feenableexcept(FE_DIVBYZERO | FE_OVERFLOW | FE_INVALID);
	
	constexpr size_t N = 20;
	Mat<float32> X[N];
	Mat<float32> Y[N];
	
	for (size_t i = 0; i < N; i++) {
		float x = distribution(generator);
		X[i].resize(1, 1, x);
		Y[i].resize(1, 1, 2.0 * x);
		cout << "f(" << x << ") =" << 2.0 * x << endl;
	}

	Dense L(1, "Linear");

	L.build(1).rand_uniform_range(0.0, 0.0);

	constexpr size_t epochs = 1000;
	constexpr double lr = 0.01;

	for (size_t e = 1; e <= epochs; e++) {
		Mat<float32> A[N];
		double tmse_derivate = 0.0;
		
		for (size_t i = 0; i < N; i++) {
			
			// Feedforward prop
			A[i] = L(X[i]);
			
			// Catch the errors			
			Mat<float32> dA = MseDerivate(A[i], Y[i], N);
			
			tmse_derivate += dA(0, 0);

			// Fit the model
			L.fit(dA, A[i], lr);
		}
		
		if (e % 10 == 0)
			cout << "Epoch = " << e << " Mse = " << Mse(A, Y, N)(0, 0)
			     << " TMseDerivate = " << tmse_derivate <<  endl;
	}

	cout << L << endl;

	constexpr size_t T = 4;

	// Test examples
	Mat<float32> TX[T] = {
		{{1.0}},
		{{-1.0}},
		{{5.0}},
		{{9.0}}
	};
	
	Mat<float32> TY[T] = {
		{{2.0}},
		{{-2.0}},
		{{10.0}},
		{{18.0}}
	};

	for (size_t i = 0; i < T; i++) {
		cout << "----------------------------" << endl;
		cout << "Testing..." << endl;
		cout << "X = " << TX[i](0, 0) << " Y[i] = " << TY[i](0, 0) << endl;
		cout << "Pred = " << L(TX[i])(0, 0) << endl;
	}
}


TEST(LayerTestFloat32, TestNotSimpleLineaRegression)
{
	// We are going to try to teach the model to multiply any pair of numbers
	// Predict f(x, c) = c * x, in a range of [-10, 10]

	// Generate data for the model
	
	// Seed the random number generator
	std::random_device rd;
	std::mt19937 generator(rd());
	std::uniform_real_distribution<float> distribution(- 10, 10);
	
	feenableexcept(FE_DIVBYZERO | FE_OVERFLOW | FE_INVALID);
	
	constexpr size_t N = 500;
	Mat<float32> X[N];
	Mat<float32> Y[N];
	
	for (size_t i = 0; i < N; i++) {
		float x = distribution(generator);
		float c = distribution(generator);
		X[i].resize(2, 1);
		X[i](0, 0) = x;
		X[i](1, 0) = c;
		Y[i].resize(1, 1, c * x);
		// cout << "f(" << x << ", " << c << ") = "
		//      << c * x << endl;
	}
	
	// To avoid getting nan
	// link: https://stackoverflow.com/questions/37232782/nan-loss-when-training-regression-network

	Dense L1(6, "Sigmoid"), L2(6, "Sigmoid"), L3(1, "Linear");
	
	L1.build(2).rand_uniform_range(-1.0, 1.0);
	L2.build(6).rand_uniform_range(-1.0, 1.0);
	L3.build(6).rand_uniform_range(-1.0, 1.0);

	constexpr size_t epochs = 2000;
	constexpr double lr = 0.01;

	for (size_t e = 1; e <= epochs; e++) {
		Mat<float32> A1[N], A2[N], A3[N];
		double tmse_derivate = 0.0;
		
		for (size_t i = 0; i < N; i++) {
			
			// Feedforward prop
			A1[i] = L1(X[i]);
			A2[i] = L2(A1[i]);
			A3[i] = L3(A2[i]);
			
			// Catch the errors			
			Mat<float32> dA3 = MseDerivate(A3[i], Y[i], N);
			Mat<float32> dA2 = L3.get_derror_dinput(dA3);
			Mat<float32> dA1 = L2.get_derror_dinput(dA2);
			
			tmse_derivate += dA3(0, 0);

			// Fit the model
			L3.fit(dA3, A2[i], lr);
			L2.fit(dA2, A1[i], lr);
			L1.fit(dA1, X[i], lr);
		}
		
		if (e % 10 == 0)
			cout << "Epoch = " << e << " Mse = " << Mse(A3, Y, N)(0, 0)
			     << " MseDerivate = " << tmse_derivate <<  endl;
	}

	cout << L1 << endl;
	cout << L2 << endl;
	cout << L3 << endl;

	constexpr size_t T = 4;

	// Test examples
	Mat<float32> TX[T] = {
		{{1.0}, {2.0}},
		{{-1.0}, {4.0}},
		{{5.0}, {1.0}},
		{{9.0}, {0.0}}
	};
	
	Mat<float32> TY[T] = {
		{{2.0}},
		{{-4.0}},
		{{5.0}},
		{{0.0}}
	};

	for (size_t i = 0; i < T; i++) {
		cout << "----------------------------" << endl;
		cout << "Testing..." << endl;
		cout << "X = " << TX[i](0, 0) << " Y[i] = " << TY[i](0, 0) << endl;
		cout << "Pred = " << L3(L2(L1(TX[i])))(0, 0) << endl;
	}
}


// TEST(LayerTestFloat32, TestMultiLayerConvergence)
// {
// 	// Predict f(x) = x^2, in a range of [-10, 10]

// 	// Generate data for the model
	
// 	// Seed the random number generator
// 	std::random_device rd;
// 	std::mt19937 generator(rd());
// 	std::uniform_real_distribution<float> distribution(- 10, 10);
			
// 	// Define the distribution for the desired range

// 	feenableexcept(FE_DIVBYZERO | FE_OVERFLOW | FE_INVALID);
	    
// 	constexpr size_t N = 20;
// 	Mat<float32> X[N];
// 	Mat<float32> Y[N];
	
// 	for (size_t i = 0; i < N; i++) {
// 		float x = distribution(generator);
// 		X[i].resize(1, 1, x);
// 		Y[i].resize(1, 1, x * x);
// 		// cout << "f(" << x << ") = " << x * x << endl;
// 	}

// 	// Build the model
// 	Dense L1(3, "Relu"), L2(3, "Relu"), L3(1, "Linear");
	
// 	// Works better if you add a little bit of randomness
// 	L1.build(1).rand_uniform_range(0.0, 0.0);
// 	L2.build(3).rand_uniform_range(0.0, 0.0);
// 	L3.build(3).rand_uniform_range(0.0, 0.0);
	
// 	// cout << L1 << endl;
// 	// cout << L2 << endl;
// 	// cout << L3 << endl;
	
// 	constexpr size_t epochs = 1000;
// 	constexpr double lr = 0.01;

// 	for (size_t e = 1; e <= epochs; e++) {
// 		Mat<float32> A1[N], A2[N], A3[N];
// 		double tmse_derivate = 0.0;
		
// 		for (size_t i = 0; i < N; i++) {
			
// 			// cout << L1 << endl;
// 			// cout << L2 << endl;
// 			// cout << L3 << endl;
			
// 			// Feedforward prop
// 			A1[i] = L1(X[i]);
// 			A2[i] = L2(A1[i]);
// 			A3[i] = L3(A2[i]);
			
// 			// Catch the errors			
// 			Mat<float32> dA3 = MseDerivate(A3[i], Y[i], N);
// 			Mat<float32> dA2 = L3.get_derror_dinput(dA3);
// 			Mat<float32> dA1 = L2.get_derror_dinput(dA2);
			
// 			tmse_derivate += dA3(0, 0);

// 			// Fit the model
// 			L3.fit(dA3, A2[i], lr);
// 			L2.fit(dA2, A1[i], lr);
// 			L1.fit(dA1, X[i], lr);
// 		}
// 		if (e % 10 == 0)
// 			cout << "Epoch = " << e << " Mse = " << Mse(A3, Y, N)(0, 0)
// 			     << " TMseDerivate = " << tmse_derivate <<  endl;

// 	}

// 	// Print the model
// 	cout << L1 << endl;
// 	cout << L2 << endl;
// 	cout << L3 << endl;
	
// 	constexpr size_t T = 4;

// 	// Test examples
// 	Mat<float32> TX[T] = {
// 		{{1.0}},
// 		{{-1.0}},
// 		{{5.0}},
// 		{{9.0}}
// 	};
	
// 	Mat<float32> TY[T] = {
// 		{{1.0}},
// 		{{1.0}},
// 		{{25.0}},
// 		{{81.0}}
// 	};

// 	for (size_t i = 0; i < T; i++) {
// 		cout << "----------------------------" << endl;
// 		cout << "Testing..." << endl;
// 		cout << "X = " << TX[i](0, 0) << " Y[i] = " << TY[i](0, 0) << endl;
// 		cout << "Pred = " << L3(L2(L1(TX[i])))(0, 0) << endl;
// 	}
// }

// We don't have the tech to face this problem :(

// TEST(LayerTestFloat32, TestMultipleLayerConvergence)
// {
// 	// Poorly results with these arquiecture
// 	// Dense L1(2, "Sigmoid"), L2(1, "Sigmoid");
	
// 	// srand(time(0));
	
// 	// L1.build(2).rand_uniform_range(-1.0, 1.0);
// 	// L2.build(2).rand_uniform_range(-1.0, 1.0);

// 	Dense L1(3, "Relu"), L2(2, "Relu"), L3(1, "Sigmoid");
	
// 	srand(time(0));

// 	// Randomness somtimes help the model
// 	L1.build(2).rand_uniform_range(-10.0, 10.0);
// 	L2.build(3).rand_uniform_range(-10.0, 10.0);
// 	L3.build(2).rand_uniform_range(-10.0, 10.0);

// 	// We need more data to solve this problem 4 examples is not enough
// 	// constexpr size_t N = 4;
// 	constexpr size_t N = 200; // a divisibly by four
	
// 	Mat<float32> X[N];
// 	Mat<float32> Y[N];

// 	// Generate random data for the xor problem
// 	for (size_t i = 0; i < N; i += 4) {
// 		// cout << "---------------------\n" << endl;
// 		// cout << "cluster: 0 ^ 0 = 0" << endl;
		
// 		X[i].resize(2, 1);
// 		Y[i].resize(1, 1);
		
// 		// Generate random inputs that tends to be zero
// 		X[i](0, 0) = TEND_TO_ZERO;
// 		X[i](1, 0) = TEND_TO_ZERO;
// 		Y[i](0, 0) = TEND_TO_ZERO;
		
// 		// cout << "Input: " << X[i] << " | Output: " << Y[i](0, 0) << endl;
			
// 		// cout << "---------------------\n" << endl;
// 		// cout << "cluster: 0 ^ 1 = 1" << endl;
		
// 		X[i + 1].resize(2, 1);
// 		Y[i + 1].resize(1, 1);
		
// 		X[i + 1](0, 0) = TEND_TO_ZERO;
// 		X[i + 1](1, 0) = TEND_TO_ONE;
// 		Y[i + 1](0, 0) = TEND_TO_ONE;
		
// 		// cout << "Input: " << X[i + 1] << " | Output: " << Y[i + 1](0, 0) << endl;
		
// 		// cout << "---------------------\n" << endl;
// 		// cout << "cluster: 1 ^ 0 = 1" << endl;
		
// 		X[i + 2].resize(2, 1);
// 		Y[i + 2].resize(1, 1);
		
// 		// Generate random inputs that tends to be one
// 		X[i + 2](0, 0) = TEND_TO_ONE;
		
// 		// Generate random inputs that tends to be zero
// 		X[i + 2](1, 0) = TEND_TO_ZERO;
		
// 		// Generate random inputs that tends to be one
// 		Y[i + 2](0, 0) = TEND_TO_ONE;
		
// 		// cout << "Input: " << X[i + 2] << " | Output: " << Y[i + 2](0, 0) << endl;
		
// 		// cout << "---------------------\n" << endl;
// 		// cout << "cluster: 1 ^ 1 = 0" << endl;
		
// 		X[i + 3].resize(2, 1);
// 		Y[i + 3].resize(1, 1);
		
		
// 		// Generate random inputs that tends to be one
// 		X[i + 3](0, 0) = TEND_TO_ONE;
// 		X[i + 3](1, 0) = TEND_TO_ONE;
		
// 		// Generate random inputs that tends to be zero
// 		Y[i + 3](0, 0) = TEND_TO_ZERO;
		
// 		// cout << "Input: " << X[i + 3] << " | Output: " << Y[i + 3](0, 0) << endl;
// 	}
	
// 	// Xor, Test examples
// 	Mat<float32> TX[4] = {
// 		{{0.0}, {0.0}},
// 		{{1.0}, {0.0}},
// 		{{0.0}, {1.0}},
// 		{{1.0}, {1.0}}
// 	};
	
// 	Mat<float32> TY[4] = {
// 		{{0.0}},
// 		{{1.0}},
// 		{{1.0}},
// 		{{0.0}}
// 	};

	
// 	constexpr size_t epochs = 15000;
// 	double lr = 1.0;
	
// 	Mat<float32> A1[N], A2[N], A3[N];
// 	for (size_t e = 1; e <= epochs; e++) {
// 		for (size_t i = 0; i < N; i++) {
// 			// The feedforward of the model
// 			A1[i] = L1(X[i]);
// 			A2[i] = L2(A1[i]);
// 			A3[i] = L3(A2[i]);

			
// 			// Compute the errors
// 			Mat<float32> dE3 = MseDerivate(A3[i], Y[i], N);
// 			Mat<float32> dE2 = L3.get_derror_dinput(dE3);
// 			Mat<float32> dE1 = L2.get_derror_dinput(dE2);
			
			
// 			// Fit the model
// 			L3.fit(dE3, A2[i], lr);
// 			L2.fit(dE2, A1[i], lr);
// 			L1.fit(dE1, X[i], lr);
// 		}

// 		if (e % 10 == 0)
// 			cout << "Epoch: " << e << " MSE: " << Mse(A3, Y, N).grand_sum()
// 			     << " lr = " << lr << endl;
// 	}

// 	cout << "\n\n---------------------\n\n" << endl;
// 	cout << " MSE: " << Mse(A3, Y, N).grand_sum() << endl;
	
// 	cout << L1 << endl;
// 	cout << "---------------------\n" << endl;
// 	cout << L2 << endl;
// 	cout << "---------------------\n" << endl;
// 	cout << L3 << endl;
	
// 	for (size_t i = 0; i < 4; i++) {
// 		cout << "---------------------\n" << endl;
// 		A3[i] = L3(L2(L1(TX[i])));
// 		cout << "Outpu = " << TY[i](0, 0) << endl;
// 		cout << "Predicted = " << A3[i](0, 0) <<  endl;
// 	}


// }

