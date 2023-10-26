#pragma once
#include <list>
using namespace std;

static class Loss
{
public:
	static float crossEntropy(list<float> predicted, list<float> expected);
};

