#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#ifdef _DEBUG
#ifndef DBG_NEW
#define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
#define new DBG_NEW
#endif
#endif  // _DEBUG

#include "Initialisation.h"
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <random>
using namespace std;

/// <summary>
/// Returns a float array of the upper and lower bounds of weights.
/// </summary>
/// <param name="previousLayerNodeCount"></param>
/// <param name="nextLayerNodeCount"></param>
/// <returns></returns>
double* Initialisation::Xavier(double previousLayerNodeCount, double nextLayerNodeCount)
{
	double* results = new double[2];
	results[0] = -(sqrt(6.0) / sqrt(previousLayerNodeCount + nextLayerNodeCount));
	results[1] = (sqrt(6.0) / sqrt(previousLayerNodeCount + nextLayerNodeCount));
	return results;
}

double Initialisation::He(float previousLayerNodeCount) { return sqrt(2.0 / previousLayerNodeCount); }

double Initialisation::Random(float upperBound, float lowerBound)
{
	random_device device;
	mt19937 random(device());
	uniform_real_distribution<float> distribution(lowerBound, upperBound);
	return distribution(random);
}