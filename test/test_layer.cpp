
#include <iostream>
#include <cassert>
#include "../include/cnet/layer.hpp"

int main(void)
{
	cnet::layer<double> l(10, 10, cnet::CNET_SIGMOID);
	cnet::layer<double> l2(10, 5, cnet::CNET_SIGMOID);
	cnet::layer<double> l3(5, 1, cnet::CNET_SIGMOID);
	
	cnet::mat<double> X(10, 1);
	cnet::rand_mat(X, -10.0, 10.0);
	
	std::cout << "X = \n";
	std::cout << X << std::endl;
	std::cout << l << std::endl;
	
	cnet::mat<double> A1 = l.feedforward(X);
	std::cout << "A1 = \n";
	std::cout << A1 << std::endl;

	std::cout << l2 << std::endl;
	
	cnet::mat<double> A2 = l2.feedforward(A1);
	std::cout << "A2 = \n";
	std::cout << A2 << std::endl;

	std::cout << l3 << std::endl;

	cnet::mat<double> A3 = l3.feedforward(A2);
	std::cout << "A3 = \n";
	std::cout << A3 << std::endl;
	
	return 0;
}
