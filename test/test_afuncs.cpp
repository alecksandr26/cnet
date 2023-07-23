
#include <iostream>
#include "../include/cnet/afuncs.hpp"

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

	cnet::mat<double> A = W * X;
	A = cnet::sigmoid(A);
	std::cout << A<< std::endl;
	
	return 0;
}
