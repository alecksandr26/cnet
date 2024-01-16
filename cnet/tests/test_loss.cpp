
#include <iostream>
#include <gtest/gtest.h>
#include <chrono>

#include "cnet/dtypes.hpp"
#include "cnet/mat.hpp"
#include "cnet/variable.hpp"
#include "cnet/activation.hpp"
#include "cnet/loss.hpp"

using namespace std;
using namespace cnet;
using namespace dtypes;
using namespace mathops;
using namespace variable;
using namespace loss;


TEST(TestCostFloat32, TestMse)
{
	constexpr size_t N = 5;
	Mat<float32> A[N] = {
		{
			{58}
		},
		{
			{71}
		},
		{
			{79}
		},
		{
			{87}
		},
		{
			{92}
		}
	};

	Mat<float32> Y[N] = {
		{
			{60}
		},
		{
			{70}
		},
		{
			{80}
		},
		{
			{85}
		},
		{
			{90}
		}
	};

	Mat<float32> C = Mse(A, Y, N);

	cout << C << endl;

	EXPECT_NEAR(C(0, 0), 2.8, 1e-2);
}


TEST(TestCostFloat64, TestMse)
{
	constexpr size_t N = 5;
	Mat<float64> A[N] = {
		{
			{58}
		},
		{
			{71}
		},
		{
			{79}
		},
		{
			{87}
		},
		{
			{92}
		}
	};

	Mat<float64> Y[N] = {
		{
			{60}
		},
		{
			{70}
		},
		{
			{80}
		},
		{
			{85}
		},
		{
			{90}
		}
	};

	Mat<float64> C = Mse(A, Y, N);

	cout << C << endl;
	EXPECT_NEAR(C(0, 0), 2.8, 1e-2);
}


TEST(TestCostVar, TestMse)
{
	constexpr size_t N = 5;
	Var A[N], Y[N];
	
	A[0] = Mat<float64>({{58}});
	A[1] = Mat<float64>({{71}});
	A[2] = Mat<float64>({{79}});
	A[3] = Mat<float64>({{87}});
	A[4] = Mat<float64>({{92}});


	Y[0] = Mat<float64>({{60}});
	Y[1] = Mat<float64>({{70}});
	Y[2] = Mat<float64>({{80}});
	Y[3] = Mat<float64>({{85}});
	Y[4] = Mat<float64>({{90}});
	
	Var C = Mse(A, Y, N);
	cout << C << endl;
	EXPECT_NEAR(C.at_mf64(0, 0), 2.8, 1e-2);
	
	A[0] = Mat<float32>({{58}});
	A[1] = Mat<float32>({{71}});
	A[2] = Mat<float32>({{79}});
	A[3] = Mat<float32>({{87}});
	A[4] = Mat<float32>({{92}});

	Y[0] = Mat<float32>({{60}});
	Y[1] = Mat<float32>({{70}});
	Y[2] = Mat<float32>({{80}});
	Y[3] = Mat<float32>({{85}});
	Y[4] = Mat<float32>({{90}});
	
	C = Mse(A, Y, N);
	
	cout << C << endl;
	EXPECT_NEAR(C.at_mf32(0, 0), 2.8, 1e-2);
}

TEST(TestCostFloat32, TestMseDerivate)
{
	Mat<float32> A = {{1.0}};
	Mat<float32> Y = {{0.0}};

	ASSERT_EQ(MseDerivate(A, Y, 4)(0, 0), 0.5);
	
	A = {{0.5}};
	Y = {{0.0}};

	ASSERT_EQ(MseDerivate(A, Y, 4)(0, 0), 0.25);
}

TEST(TestCostFloat64, TestMseDerivate)
{
	Mat<float64> A = {{1.0}};
	Mat<float64> Y = {{0.0}};

	ASSERT_EQ(MseDerivate(A, Y, 4)(0, 0), 0.5);
	
	A = {{0.5}};
	Y = {{0.0}};

	ASSERT_EQ(MseDerivate(A, Y, 4)(0, 0), 0.25);
}

