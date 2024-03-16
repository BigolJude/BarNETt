#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#ifdef _DEBUG
#ifndef DBG_NEW
#define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
#define new DBG_NEW
#endif
#endif  // _DEBUG

#include "Activation.h"
#include <cmath>
#include <list>
#include <iostream>
using namespace std;

/// <summary>
/// Standard relu activation.
/// </summary>
/// <param name="weight">The dot product of all the weights</param>
/// <returns>0 if the value is below 0 and the value if the value is above.</returns>
double Activation::ReLu(double weight) { return (weight > 0) ? weight : 0; }

/// <summary>
/// Leaky relu activation based of off research <research here>
/// </summary>
/// <param name="weight"></param>
/// <returns></returns>
double Activation::LReLu(double weight) { return (weight > 0) ? weight : 0.01 * weight; }

/// <summary>
/// Standard softmax activation.
/// </summary>
/// <param name="weights"></param>
/// <returns></returns>
list<double> Activation::SoftMax(list<double> weights)
{
	list<double>::iterator weightsIt = weights.begin();
	
	double denominator = 0;

	for (int i = 0; i < weights.size(); ++i)
	{
		denominator = denominator + exp(*weightsIt);
		advance(weightsIt, 1);
	}

	weightsIt = weights.begin();
	list<double> outputs;

	for (int i = 0; i < weights.size(); ++i)
	{
		double weight = exp(*weightsIt) / denominator;
		outputs.push_back(weight);
		advance(weightsIt, 1);
	}

	return outputs;
}

/// <summary>
/// Standard step activation.
/// </summary>
/// <param name="weight"></param>
/// <returns></returns>
double Activation::Step(double weight)
{
	return static_cast<double>(weight > 0);
}