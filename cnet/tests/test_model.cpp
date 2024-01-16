// To have output: make test ARGS=-V

// Run valgrind to check memory issues
// valgrind --leak-check=full --track-origins=yes -s --show-leak-kinds=all --max-stackframe=62179200

#include <iostream>
#include <gtest/gtest.h>
#include <chrono>

#include "cnet/layer.hpp"
#include "cnet/model.hpp"

using namespace cnet;
using namespace cnet::model;
using namespace cnet::layer;

TEST(ModelTest, TestConstructo)
{
	sequential model({dense<double>(4, "sigmoid")});
}
