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
float* Initialisation::Xavier(float previousLayerNodeCount, float nextLayerNodeCount)
{
	float* results = new float[2];
	results[0] = -(sqrt(6.0) / sqrt(previousLayerNodeCount + nextLayerNodeCount));
	results[1] = (sqrt(6.0) / sqrt(previousLayerNodeCount + nextLayerNodeCount));
	return results;
}

float Initialisation::He(float previousLayerNodeCount)
{
	return sqrt(2.0 / previousLayerNodeCount);
}

float Initialisation::Random(float upperBound, float lowerBound)
{
	random_device device;
	mt19937 random(device());
	uniform_real_distribution<float> distribution(lowerBound, upperBound);
	return distribution(random);
}