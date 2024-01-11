#include <iostream>
#include <chrono>

#include "cnet/dtypes.hpp"
#include "cnet/variable.hpp"

using namespace cnet;
using namespace variable;
using namespace dtypes;
using namespace std;


int main(void)
{
	Mats m;
	
	m.f32 = {{1, 2}};
	cout << m.f32 << endl;

	// Take care of allocating 
	// m.f64 = {{1, 2}};
	
	return 0;
}



