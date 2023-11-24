#include <iostream>
#include <cassert>
#include "../include/cnet/mat.hpp"
#include "../include/cnet/afunc.hpp"

void example_mul(void)
{
	cnet::mat<double> A(2, 3);
	cnet::mat<double> B(3, 1);
	cnet::mat<double> D(1, 2);

	cnet::rand_mat(A, 0.0, 2.0);
	cnet::rand_mat(B, 0.0, 2.0);
	cnet::rand_mat(D, 0.0, 2.0);

	std::cout << A << std::endl;
	std::cout << B << std::endl;
	
	cnet::mat<double> C = A * B;
	
	std::cout << C << std::endl;
	std::cout << D << std::endl;

	std::cout << C.get_cols() << " " << D.get_rows() << std::endl;
	cnet::mat<double> R = C * D;
	std::cout << R << std::endl;
}


int main(void)
{
	cnet::mat<double> A = {
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0}
	};
	
	cnet::mat<double> B = {
		{1.0},
		{1.0},
		{1.0}
	};

	cnet::mat<double> D = {
		{5.0},
		{5.0}
	};
	
	cnet::mat<double> C = A * B + D;
	cnet::mat<double> M;

	M = C;
	

	std::cout << "A = " << std::endl;
	std::cout << A << std::endl;

	std::cout << "B = " << std::endl;
	std::cout << B << std::endl;
	
	std::cout << "C = " << std::endl;
	std::cout << C << std::endl;

	std::cout << "D = " << std::endl;
	std::cout << D << std::endl;

	
	std::cout << "sig(C)"  << std::endl;
	std::cout << cnet::sigmoid(C) << std::endl;

	std::cout << "M = " << std::endl;
	std::cout << M << std::endl;
	
		
	return 0;
}






