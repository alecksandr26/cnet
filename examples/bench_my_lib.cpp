#include <iostream>
#include <chrono>
#include <memory>

#include "cnet/mat.hpp"
#include "cnet/afunc.hpp"
#include "cnet/layer.hpp"

using namespace cnet;
using namespace cnet::layer;
using namespace cnet::afunc;

int main(void)
{
	Mat<float> X = {	// 2 allocs
		{1, 2, 3, 4, 5, 6, 7, 8, 9},
		{10, 11, 12, 13, 14, 15, 16, 17, 18},
		{19, 20, 21, 22, 23, 24, 25, 26, 27},
		{28, 29, 30, 31, 32, 33, 34, 35, 36},
		{37, 38, 39, 40, 41, 42, 43, 44, 45},
		{46, 47, 48, 49, 50, 51, 52, 53, 54},
		{55, 56, 57, 58, 59, 60, 61, 62, 63},
		{64, 65, 66, 67, 68, 69, 70, 71, 72},
		{73, 74, 75, 76, 77, 78, 79, 80, 81}
	};

	Mat<float> Y = {	//  2 allocs
		{9, 8, 7, 6, 5, 4, 3, 2, 1},
		{18, 17, 16, 15, 14, 13, 12, 11, 10},
		{27, 26, 25, 24, 23, 22, 21, 20, 19},
		{36, 35, 34, 33, 32, 31, 30, 29, 28},
		{45, 44, 43, 42, 41, 40, 39, 38, 37},
		{54, 53, 52, 51, 50, 49, 48, 47, 46},
		{63, 62, 61, 60, 59, 58, 57, 56, 55},
		{72, 71, 70, 69, 68, 67, 66, 65, 64},
		{81, 80, 79, 78, 77, 76, 75, 74, 73}
	};

	// Results in Mat mul
	// (2565 | 2520 | 2475 | 2430 | 2385 | 2340 | 2295 | 2250 | 2205
	// 6210 | 6084 | 5958 | 5832 | 5706 | 5580 | 5454 | 5328 | 5202
	// 9855 | 9648 | 9441 | 9234 | 9027 | 8820 | 8613 | 8406 | 8199
	// 13500 | 13212 | 12924 | 12636 | 12348 | 12060 | 11772 | 11484 | 11196
	// 17145 | 16776 | 16407 | 16038 | 15669 | 15300 | 14931 | 14562 | 14193
	// 20790 | 20340 | 19890 | 19440 | 18990 | 18540 | 18090 | 17640 | 17190
	// 24435 | 23904 | 23373 | 22842 | 22311 | 21780 | 21249 | 20718 | 20187
	// 28080 | 27468 | 26856 | 26244 | 25632 | 25020 | 24408 | 23796 | 23184
	// 31725 | 31032 | 30339 | 29646 | 28953 | 28260 | 27567 | 26874 | 26181)
	
	// Mat<float> X = {
	// 	{1, 2, 3, 4},
	// 	{5, 6, 7, 8},
	// 	{9, 10, 11, 12},
	// 	{13, 14, 15, 16}
	// };


	// Mat<float> Y = {
	// 	{17, 18, 19, 20},
	// 	{21, 22, 23, 24},
	// 	{25, 26, 27, 28},
	// 	{29, 30, 31, 32}
	// };

	// |250.00000 260.00000 270.00000 280.00000|
        // |618.00000 644.00000 670.00000 696.00000|
	// |986.00000 1028.00000 1070.00000 1112.00000|
        // |1354.00000 1412.00000 1470.00000 1528.00000|
	
	// Mat<float> X = {
	// 	{1, 2},
	// 	{3, 4}
	// };
	
	// Mat<float> Y = {
	// 	{1, 2},
	// 	{3, 4}
	// };

	// [[7.000000	10.000000]
	//  [15.000000	22.000000]]

	
	// Mat<double> X = {
	// 	{1}
	// };

	// Mat<double> Y = {
	// 	{2}
	// };

	// [[2]]
	
	// 1 alloc
	Mat<float> C;
	
	// for (std::size_t i = 0; i < 10; i++)
	
	// * -> 5 allocs
	// = -> 1 allocn
	C = X * Y;
	
	// |190.00000 200.00000 210.00000|
	// |470.00000 496.00000 522.00000|
	// |750.00000 792.00000 834.00000|

	// valgrind --leak-check=full
	std::cout << C << std::endl;
	
	// std::cout << sigmoid(C) << std::endl;

	// dense<double> L(4, "sigmoid");
	
	// // L.set_use_bias(false);
	// L.build(4, 1.0);
	
	// std::cout << L << std::endl;
	
	// Mat<double> X_IN = {
	// 	{0},
	// 	{1},
	// 	{2.5},
	// 	{5}
	// };
	
	// std::cout << L(X_IN) << std::endl;
	
	// https:en.algorithmica.org/hpc/external-memory/oblivious/#algorithm

	// Measured with this cpu
	
	// Architecture:                       x86_64
	// CPU op-mode(s):                     32-bit, 64-bit
	// Address sizes:                      43 bits physical, 48 bits virtual
	// Byte Order:                         Little Endian
	// CPU(s):                             12
	// On-line CPU(s) list:                0-11
	// Vendor ID:                          AuthenticAMD
	// Model name:                         AMD Ryzen 5 2600 Six-Core Processor
	// CPU family:                         23
	// Model:                              8
	// Thread(s) per core:                 2
	// Core(s) per socket:                 6
	// Socket(s):                          1
	// Stepping:                           2
	// Frequency boost:                    enabled
	// CPU(s) scaling MHz:                 53%
	// CPU max MHz:                        3400.0000
	// CPU min MHz:                        1550.0000
	// Virtualization:                     AMD-V
	// L1d cache:                          192 KiB (6 instances)
	// L1i cache:                          384 KiB (6 instances)
	// L2 cache:                           3 MiB (6 instances)
	// L3 cache:                           16 MiB (2 instances)
	// NUMA node(s):                       1
	// NUMA node0 CPU(s):                  0-11

	

	// N = 50
	// Normal Elapsed Time: 0.016529 ~ 0.036987 seconds 
	// Strassen Vectorized version Elapsed Time: 0.006684 ~ 0.007511 seconds
	// The paralleized Strassen Vectorized version Elapsed Time: 0.003776 ~ 0.005055 seconds
	
	// N = 100
	// Normal Elapsed Time: 0.110695 ~ 0.153430 seconds
	// Strassen Elapsed Time: 0.088378 ~ 0.133848 seconds
	// Strassen Vectorized version Elapsed Time: 0.045144 ~ 0.087983 seconds
	// The paralleized Strassen Vectorized version Elapsed Time: 0.017151 ~ 0.020471 seconds

	// N = 200
	// Normal Elapsed Time: 0.872653 ~ 0.902815 seconds
	// Strassen Elapsed Time: 0.663484 ~ 0.723685 seconds
	// Strassen Vectorized version Elapsed Time: 0.328136 ~ 0.338400 seconds
	// The paralleized Strassen Vectorized version Elapsed Time: 0.077055 ~ 0.080233 seconds
	
	// N = 300
	// Normal Elapsed Time: 2.912563 ~ 3.085271 seconds
	// Strassen Elapsed Time: 5.546863 ~ 5.931044 seconds
	// Strassen Vectorized version Elapsed Time: 2.450393 ~ 2.518431 seconds
	// The paralleized Strassen Vectorized version Elapsed Time: 0.383102 ~ 0.431236 seconds

	// N = 500
	// Normal Elapsed Time: 14.304790 ~ 15.288232 seconds
	// Strassen Elapsed Time: 5.470263 ~ 5.621796 seconds
	// Strassen Vectorized version Elapsed Time: 2.493669 ~ 2.507378 seconds
	// The paralleized Strassen Vectorized version Elapsed Time: 0.402513 ~ 0.442851 seconds

	// N = 1000
	// Normal Elapsed Time: 118.052953 seconds
	// Strassen Elapsed Time: 46.645130 ~ 48.601734 seconds
	// Strassen Vectorized version Elapsed Time: 19.657362 ~ 19.764867 seconds
	// The paralleized Strassen Vectorized version Elapsed Time: 2.929702 ~ 2.996234 seconds

	// --------------------- here ----------------------------
	static constexpr int size_mat = 1000;
	
	Mat<double> A(size_mat, size_mat);
	Mat<double> B(size_mat, size_mat);
	
	// Assing random values
	A.rand(0.0, 1.0);
	B.rand(0.0, 1.0);
	
	auto beg = std::chrono::high_resolution_clock::now();
	
	Mat<double> R;
	// for (std::size_t i = 0; i < 10; i++)
	R = A * B;
	
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
	
	// Convert microseconds to seconds
	double seconds = duration.count() / 1e6;
	
	std::cout << "Mat mul Elapsed Time: " << std::fixed << std::setprecision(6)
		  << seconds << " seconds" << std::endl;


	// // // Perfom the addition

	// // N = 1000
	// // Normal version Mat add Elapsed Time: 0.117499 seconds
	// // The vectorized version Mat add Elapsed Time: 0.021676 seconds
	// // The new version almost the finall version Mat add Elapsed Time: 0.006490 seconds
	
	// beg = std::chrono::high_resolution_clock::now();
	
	// Mat<float> R1;
	// // for (std::size_t i = 0; i < 10; i++)
	// R1 = A + B;
	
	// end = std::chrono::high_resolution_clock::now();
	// duration = std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
	
	// // Convert microseconds to seconds
	// seconds = duration.count() / 1e6;
	
	// std::cout << "Mat add Elapsed Time: " << std::fixed << std::setprecision(6)
	// 	  << seconds << " seconds" << std::endl;


	// // Perform the scalar product
	// // The new version almost the finall version Mat scalar product Elapsed Time: 0.001471 seconds

	// beg = std::chrono::high_resolution_clock::now();
	
	// // for (std::size_t i = 0; i < 10; i++)
	// R1 = A * 10.0;
	
	// end = std::chrono::high_resolution_clock::now();
	// duration = std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
	
	// // Convert microseconds to seconds
	// seconds = duration.count() / 1e6;
	
	// std::cout << "Mat scalar product Elapsed Time: " << std::fixed << std::setprecision(6)
	// 	  << seconds << " seconds" << std::endl;


	// Perform the sigmoid function
	// beg = std::chrono::high_resolution_clock::now();
	
	// // for (std::size_t i = 0; i < 10; i++)
	// R1 = sigmoid<double>()(A);
	
	// end = std::chrono::high_resolution_clock::now();
	// duration = std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
	
	// // Convert microseconds to seconds
	// seconds = duration.count() / 1e6;
	
	// std::cout << "Mat sigmoid func Elapsed Time: " << std::fixed << std::setprecision(6)
	// 	  << seconds << " seconds" << std::endl;
	
	// beg = std::chrono::high_resolution_clock::now();
	
	// // for (std::size_t i = 0; i < 10; i++)
	// R1 = relu<double>()(A);
	
	// end = std::chrono::high_resolution_clock::now();
	// duration = std::chrono::duration_cast<std::chrono::microseconds>(end - beg);
	
	// // Convert microseconds to seconds
	// seconds = duration.count() / 1e6;
	
	// std::cout << "Mat relu func Elapsed Time: " << std::fixed << std::setprecision(6)
	// 	  << seconds << " seconds" << std::endl;
	
	return 0;
}


