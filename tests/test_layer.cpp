// To have output: make test ARGS=-V

// Run valgrind to check memory issues
// valgrind --leak-check=full --track-origins=yes -s --show-leak-kinds=all --max-stackframe=62179200

#include <iostream>
#include <gtest/gtest.h>
#include <chrono>
#include <memory>

#include "cnet/layer.hpp"
#include "cnet/cost.hpp"

using namespace cnet;
using namespace cnet::model;

TEST(LayerTest, TestFeedForwardSigmoidOne)
{
	layer<double> l(4, 4, std::make_unique<afunc::sigmoid<double>>());
	
	l.W_.resize(4, 4, 1);
	l.B_.resize(4, 1, 1);

	mat<double> X = {
		{0},
		{1},
		{2.5},
		{5}
	};

	mat<double> A = l.feedforward(X);

	for (std::size_t i = 0; i < A.get_rows(); i++)
		for (std::size_t j = 0; j < A.get_cols(); j++)
			EXPECT_NEAR(A(i, j), 0.999, 1e-3);
}

TEST(LayerTest, TestFeedForwardSigmoidZero)
{
	layer<double> l(4, 4, std::make_unique<afunc::sigmoid<double>>());
	
	l.W_.resize(4, 4, -1);
	l.B_.resize(4, 1, 0.0);

	mat<double> X = {
		{0},
		{1},
		{2.5},
		{5}
	};

	mat<double> A = l.feedforward(X);

	for (std::size_t i = 0; i < A.get_rows(); i++)
		for (std::size_t j = 0; j < A.get_cols(); j++)
			EXPECT_NEAR(A(i, j), 0.0, 1e-3);
}

TEST(LayerTest, TestFeedForwardReLUPositive)
{
	layer<double> l(4, 4, std::make_unique<afunc::relu<double>>());
    
	l.W_.resize(4, 4, 1);
	l.B_.resize(4, 1, 1);

	mat<double> X = {
		{0},
		{1},
		{2.5},
		{5}
	};

	mat<double> A = l.feedforward(X);

	for (std::size_t i = 0; i < A.get_rows(); i++)
		for (std::size_t j = 0; j < A.get_cols(); j++)
			// ReLU of a positive value should be the value itself
			EXPECT_NEAR(A(i, j), 9.5, 1e-3);
}

TEST(LayerTest, TestFeedForwardReLUNegative)
{
	layer<double> l(4, 4, std::make_unique<afunc::relu<double>>());
    
	l.W_.resize(4, 4, -1);
	l.B_.resize(4, 1, 0.0);

	mat<double> X = {
		{0},
		{1},
		{2.5},
		{5}
	};

	mat<double> A = l.feedforward(X);

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
	layer<double> p_and(2, 1, std::make_unique<afunc::sigmoid<double>>());

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
	
	mat<double> A[4];
	for (std::size_t np = 0; np < epochs; np++) {
		for (std::size_t i = 0; i < 4; i++) {
			A[i] = p_and.feedforward(X[i]);
			p_and.fit(cost::mse<double>().dfunc_da(A[i], Y[i], 4), lr, X[i]);
		}
		
		std::cout << "Epoch: " << np << " MSE: \n" << cost::mse<double>().func(A, Y, 4) << std::endl;
	}

	std::cout << "-------------" << std::endl;
	std::cout << p_and << std::endl;
	
	for (std::size_t i = 0; i < 4; i++) {
		std::cout << "-------------------" << std::endl;
		std::cout << "X[" << i << "]: " << std::endl;
		std::cout << X[i] << std::endl;
		
		A[i] = p_and.feedforward(X[i]);
		std::cout << "A: " << std::endl;
		std::cout << A[i] << std::endl;
		std::cout << "Y[" << i << "]: " << std::endl;
		std::cout << Y[i] << std::endl;
		EXPECT_NEAR(A[i](0, 0), Y[i](0, 0), 1e-1);
	}
	
}

// Test your big layer
TEST(LayerTest, TestFitBigLayer) {
	// For the big layer, neurons proccessing 10 inputs 4 outputs
	// constexpr std::size_t epochs = 100;
	// constexpr double lr = 5.0;

	// layer<double> p_big(10, 4, std::make_unique<afunc::sigmoid<double>>());

	// For the logic gates Two neurons proccessing 2 inputs 4 outputs
	constexpr std::size_t epochs = 32;
	constexpr double lr = 100.0;
	
	layer<double> p_big(2, 5, std::make_unique<afunc::sigmoid<double>>());
	
	// Randomly initialize weights
	// p_big.rand_range(0.0, 1.0);
	
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

	mat<double> A[4];
	for (std::size_t np = 0; np < epochs; np++) {
		for (std::size_t i = 0; i < 4; i++) {
			A[i] = p_big.feedforward(X[i]);
			p_big.fit(cost::mse<double>().dfunc_da(A[i], Y[i], 4), lr, X[i]);
		}

		std::cout << "Epoch: " << np << " MSE: \n" << cost::mse<double>().func(A, Y, 4) << std::endl;
	}

	std::cout << "-------------" << std::endl;
	std::cout << p_big << std::endl;

	for (std::size_t i = 0; i < 4; i++) {
		std::cout << "-------------------" << std::endl;
		std::cout << "X[" << i << "]: " << std::endl;
		std::cout << X[i] << std::endl;

		A[i] = p_big.feedforward(X[i]);
		std::cout << "A: " << std::endl;
		std::cout << A[i] << std::endl;
		std::cout << "Y[" << i << "]: " << std::endl;
		std::cout << Y[i] << std::endl;
		for (std::size_t k = 0; k < A[i].get_rows(); k++)
			for (std::size_t j = 0; j < A[i].get_cols(); j++) {
				// It is not going to assert the xor 
				if (k < 4)
					EXPECT_NEAR(A[i](k, j), Y[i](k, j), 1e-1);
			}
	}
}



