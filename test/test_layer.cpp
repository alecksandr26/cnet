
#include <iostream>
#include <cassert>
#include "../include/cnet/layer.hpp"

int main(void)
{
	cnet::layer<double> l(10, 10, cnet::CNET_RELU);
	l.rand_range(-5.0, 5.0);
	cnet::layer<double> l2(10, 5, cnet::CNET_RELU);
	l2.rand_range(-5.0, 5.0);
	cnet::layer<double> l3(5, 1, cnet::CNET_SIGMOID);
	l3.rand_range(-5.0, 5.0);
	
	cnet::mat<double> X(10, 1);
	cnet::rand_mat(X, -10.0, 10.0);
	
	std::cout << "X = \n";
	std::cout << X << std::endl;
	std::cout << l << std::endl;
	
	cnet::mat<double> A = l.feedforward(X);
	std::cout << "A = \n";
	std::cout << A << std::endl;

	std::cout << l2 << std::endl;
	
	A = l2.feedforward(A);
	std::cout << "A = \n";
	std::cout << A << std::endl;

	std::cout << l3 << std::endl;

	A = l3.feedforward(A);
	std::cout << "A = \n";
	std::cout << A << std::endl;
	return 0;
}
