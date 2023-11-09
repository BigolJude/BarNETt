#pragma once
#include <list>
using namespace std;

static class Loss
{
public:
	static double crossEntropy(list<double> predicted, list<double> expected);
};

