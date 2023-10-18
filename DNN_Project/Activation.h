#pragma once
#include <list>
using namespace std;

static class Activation
{
public:
	static float ReLu(float weight);
	static list<float> SoftMax(list<float> weights);
	static float Sigmoid(float weight);
	static float Step(float weight);
};