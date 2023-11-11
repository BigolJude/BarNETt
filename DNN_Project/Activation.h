#pragma once
#include <list>
using namespace std;

static class Activation
{
public:
	static double ReLu(double weight);
	static double LReLu(double weight);
	static list<double> SoftMax(list<double> weights);
	static double Sigmoid(double weight);
	static double Step(double weight);
};