#include <iostream>

#include "cnet/layer.hpp"

using namespace cnet;
using namespace cnet::layer;

int main(void)
{
	dense<double> L(1, "sigmoid");

	L.build(2);
}


