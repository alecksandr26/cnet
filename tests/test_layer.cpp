// To have output: make test ARGS=-V

// Run valgrind to check memory issues
// valgrind --leak-check=full --track-origins=yes -s --show-leak-kinds=all --max-stackframe=62179200

#include <iostream>
#include <gtest/gtest.h>
#include <chrono>

#include "cnet/layer.hpp"
#include "cnet/cost.hpp"

using namespace cnet;
using namespace cnet::layer;

TEST(LayerTest, TestFeedForwardSigmoidOne)
{
	dense<double> L(4, "sigmoid");
	
	// Build the layer
	L.build(4, 1.0);
	
	mat<double> X = {
		{0},
		{1},
		{2.5},
		{5}
	};

	// Feedforward process
	mat<double> A = L(X);
	
	for (std::size_t i = 0; i < A.get_rows(); i++)
		for (std::size_t j = 0; j < A.get_cols(); j++)
			EXPECT_NEAR(A(i, j), 0.999, 1e-3);
}

TEST(LayerTest, TestFeedForwardSigmoidZero)
{
	
	dense<double> L(4, "sigmoid");
	
	// Dont use bias
	L.set_use_bias(false);

	// Build the layer
	L.build(4, -1.0);	// With negative as init val

	mat<double> X = {
		{0},
		{1},
		{2.5},
		{5}
	};

	mat<double> A = L(X);

	for (std::size_t i = 0; i < A.get_rows(); i++)
		for (std::size_t j = 0; j < A.get_cols(); j++)
			EXPECT_NEAR(A(i, j), 0.0, 1e-3);
}

TEST(LayerTest, TestFeedForwardReLUPositive)
{
	dense<double> L(4, "relu");
    
	L.build(4, 1.0);

	mat<double> X = {
		{0},
		{1},
		{2.5},
		{5}
	};

	mat<double> A = L(X);

	for (std::size_t i = 0; i < A.get_rows(); i++)
		for (std::size_t j = 0; j < A.get_cols(); j++)
			// ReLU of a positive value should be the value itself
			EXPECT_NEAR(A(i, j), 9.5, 1e-3);
}

TEST(LayerTest, TestFeedForwardReLUNegative)
{
	dense<double> L(4, "relu");

	// Dont use bias
	L.set_use_bias(false);
	L.build(4, -1.0);
	
	mat<double> X = {
		{0},
		{1},
		{2.5},
		{5}
	};

	mat<double> A = L(X);

	for (std::size_t i = 0; i < A.get_rows(); i++)
		for (std::size_t j = 0; j < A.get_cols(); j++)
			// ReLU of a negative value should be 0
			EXPECT_NEAR(A(i, j), 0.0, 1e-3);
}

TEST(LayerTest, TestFitLayerAnd)
{
	constexpr std::size_t epochs = 32;
	constexpr double lr = 100.0;
	
	// Train a perceptron of and
	dense<double> p_and(1, "sigmoid");

	p_and.build(2, 0.0);

	// Initailize the values randomly
	// p_and.rand_range(0.0, 1.0); 
	
	std::cout << p_and << std::endl;
	
	mat<double> X[4] = {
		{{0.0}, {0.0}},	// One column and two rows
		{{0.0}, {1.0}},
		{{1.0}, {0.0}},
		{{1.0}, {1.0}}
	};


	// W = 
	// [[9.57219	9.57219]]
	// B = 
	// [[-14.5476]]
	
	// And gate
	// mat<double> Y[4] = {
	// 	{{0.0}},
	// 	{{0.0}},
	// 	{{0.0}},
	// 	{{1.0}}
	// };


	// W = 
	// [[10.3501	10.3501]]
	// B = 
	// [[-5.1672]]
	
	// Or gate
	// mat<double> Y[4] = {
	// 	{{0.0}},
	// 	{{1.0}},
	// 	{{1.0}},
	// 	{{1.0}}
	// };

	// W = 
	// [[-9.57219	-9.57219]]
	// B = 
	// [[14.5476]]
	
	// Nand gate
	mat<double> Y[4] = {
		{{1.0}},
		{{1.0}},
		{{1.0}},
		{{0.0}}
	};
	

	// W = 
	// [[-10.3501	-10.3501]]
	// B = 
	// [[5.1672]]
	
	// Nor gate
	// mat<double> Y[4] = {
	// 	{{1.0}},
	// 	{{0.0}},
	// 	{{0.0}},
	// 	{{0.0}}
	// };
	cost::mse<double> C;
	
	mat<double> A[4];
	for (std::size_t np = 0; np < epochs; np++) {
		for (std::size_t i = 0; i < 4; i++) {
			A[i] = p_and(X[i]);
			p_and.fit_backprop(C.derivate(A[i], Y[i], 4), lr, X[i]);
		}
		
		std::cout << "Epoch: " << np << " MSE: \n" << C(A, Y, 4) << std::endl;
	}

	std::cout << "-------------" << std::endl;
	std::cout << p_and << std::endl;
	
	for (std::size_t i = 0; i < 4; i++) {
		std::cout << "-------------------" << std::endl;
		std::cout << "X[" << i << "]: " << std::endl;
		std::cout << X[i] << std::endl;
		
		A[i] = p_and(X[i]);
		std::cout << "A: " << std::endl;
		std::cout << A[i] << std::endl;
		std::cout << "Y[" << i << "]: " << std::endl;
		std::cout << Y[i] << std::endl;
		EXPECT_NEAR(A[i](0, 0), Y[i](0, 0), 1e-1);
	}
	
}

// Test your big dense
TEST(LayerTest, TestFitBigLayer) {
	// For the big dense, neurons proccessing 10 inputs 4 outputs
	// constexpr std::size_t epochs = 100;
	// constexpr double lr = 5.0;
	
	// dense<double> p_big(4, std::make_unique<afunc::sigmoid<double>>());
	// p_big.build(10);

	// For the logic gates Two neurons proccessing 2 inputs 5 outputs
	constexpr std::size_t epochs = 32;
	constexpr double lr = 100.0;
	
	dense<double> p_big(5, "sigmoid");
	
	p_big.build(2, 0.0);
	
	std::cout << p_big << std::endl;

	// mat<double> X[4] = {
	// 	{{0.1}, {0.0}, {0.0}, {0.0}, {0.5}, {0.0}, {1.0}, {0.0}, {0.0}, {1.0}},  // 1x10 input
	// 	{{0.0}, {1.0}, {2.0}, {3.0}, {4.0}, {5.0}, {6.0}, {7.0}, {8.0}, {9.0}},  // Random pattern
	// 	{{1.0}, {2.0}, {3.0}, {4.0}, {5.0}, {6.0}, {7.0}, {8.0}, {9.0}, {10.0}}, // Random pattern
	// 	{{1.0}, {1.0}, {1.0}, {1.0}, {1.0}, {1.0}, {1.0}, {1.0}, {1.0}, {1.0}}   // Uniform pattern
	// };

	// // Random pattern for the output data
	// mat<double> Y[4] = {
	// 	{{0.0}, {0.0}, {1.0}, {1.0}},
	// 	{{1.0}, {0.0}, {1.0}, {0.0}},
	// 	{{1.0}, {0.0}, {1.0}, {0.0}},
	// 	{{1.0}, {0.0}, {1.0}, {0.0}}
	// };
	
	// W = 
	// [[12.6266	9.4671]   AND
	// [11.7488	10.9135]  OR
	// [-12.6266	-9.4671]  NAND
	// [-11.7488	-10.9135]] NOR
	// B = 
	// [[-18.1219] 
	// [-4.26583]
	// [18.1219]
	// [4.26583]]

	mat<double> X[4] = {
		{{0.0}, {0.0}},	// One column and two rows
		{{0.0}, {1.0}},
		{{1.0}, {0.0}},
		{{1.0}, {1.0}}
	};
	
	mat<double> Y[4] = {
		// And, Or,    Nand,  Nor,   Xor
		{{0.0}, {0.0}, {1.0}, {1.0}, {0.0}}, 
		{{0.0}, {1.0}, {1.0}, {0.0}, {1.0}}, 
		{{0.0}, {1.0}, {1.0}, {0.0}, {1.0}}, 
		{{1.0}, {1.0}, {0.0}, {0.0}, {0.0}}
	};

	cost::mse<double> C;	

	mat<double> A[4];
	for (std::size_t np = 0; np < epochs; np++) {
		for (std::size_t i = 0; i < 4; i++) {
			A[i] = p_big(X[i]);
			p_big.fit_backprop(C.derivate(A[i], Y[i], 4), lr, X[i]);
		}

		std::cout << "Epoch: " << np << " MSE: \n" << C(A, Y, 4) << std::endl;
	}

	std::cout << "-------------" << std::endl;
	std::cout << p_big << std::endl;

	for (std::size_t i = 0; i < 4; i++) {
		std::cout << "-------------------" << std::endl;
		std::cout << "X[" << i << "]: " << std::endl;
		std::cout << X[i] << std::endl;

		A[i] = p_big(X[i]);
		std::cout << "A: " << std::endl;
		std::cout << A[i] << std::endl;
		std::cout << "Y[" << i << "]: " << std::endl;
		std::cout << Y[i] << std::endl;
		for (std::size_t k = 0; k < A[i].get_rows() - 1; k++)
			for (std::size_t j = 0; j < A[i].get_cols(); j++) {
				// It is not going to assert the xor 
				EXPECT_NEAR(A[i](k, j), Y[i](k, j), 1e-1);
			}
	}
}


TEST(LayerTest, TestInputLayer)
{
	input<double> in(4);
	
	mat<double> X = {
		{0},
		{1},
		{2.5},
		{5}
	};
	
	mat<double> Y = in(X);
	
	for (std::size_t i = 0; i < X.get_rows(); i++)
		for (std::size_t j = 0; j < X.get_cols(); j++)
			ASSERT_EQ(X(i, j), Y(i, j));
}

