
#include <iostream>
#include "../include/cnet/afunc.hpp"

int main(void)
{
	cnet::mat<double> W = {
		{1.0, 2.0},
		{1.0, 2.0},
	};

	cnet::mat<double> X = {
		{1.0},
		{1.0},
	};

	cnet::mat<double> B = {
		{- 110.0},
		{10.0},
	};

	cnet::mat<double> A = W * X + B;
	std::cout << cnet::sigmoid(A) << std::endl;
	
	return 0;
}
