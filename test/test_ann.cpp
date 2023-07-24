
#include <iostream>
#include "../include/cnet/cost.hpp"
#include "../include/cnet/ann.hpp"

void test_simple_forward(void)
{
	cnet::nn_arch arch[] = {2, 4, 8, 4, 1}; // {in, hidden ... hidden, out}
	cnet::ann<double> ann(arch, size_arch(arch), cnet::CNET_SIGMOID);

	cnet::mat<double> X = {
		{1.0},
		{1.0}
	};
	
	std::cout << ann.feedforward(X) << std::endl;
}


void test_and_logic_gate(void)
{
	// Logic gate AND
	cnet::nn_arch arch[] = {2, 1}; // {in, hidden ... hidden, out}
	cnet::ann<double> ann(arch, size_arch(arch), cnet::CNET_SIGMOID);

	cnet::mat<double> input[] = {
		{
			{1.0},
			{1.0}
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
			{0.0},
			{0.0}
		}
		
	};

	cnet::mat<double> output[] = {
		{
			{1.0}
		},
		{
			{0.0}
		},
		{
			{0.0}
		},
		{
			{0.0}
		}
		
	};

	std::cout << cnet::mse(ann, input, output, 4) << std::endl;
	ann.fit(input, output, 4, 0.1, 10000);

	std::cout << cnet::mse(ann, input, output, 4) << std::endl;

	std::cout << ann.feedforward({ {1.0},  {1.0}}) << std::endl;
}

int main(void)
{
	// Logic gate XOR
	cnet::nn_arch arch[] = {2, 2, 1}; // {in, hidden ... hidden, out}
	cnet::ann<double> ann(arch, size_arch(arch), cnet::CNET_SIGMOID);

	cnet::mat<double> input[] = {
		{
			{1.0},
			{1.0}
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
			{0.0},
			{0.0}
		}
		
	};

	cnet::mat<double> output[] = {
		{
			{0.0}
		},
		{
			{1.0}
		},
		{
			{1.0}
		},
		{
			{1.0}
		}
		
	};

	std::cout << cnet::mse(ann, input, output, 4) << std::endl;
	ann.fit(input, output, 4, 0.1, 10000);
	
	std::cout << cnet::mse(ann, input, output, 4) << std::endl;

	std::cout << ann.feedforward({ {1.0},  {0.0}}) << std::endl;

	std::cout << ann << std::endl;
}
