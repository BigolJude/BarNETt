#include "Activation.h"
#include <cmath>
#include <list>
#include <iostream>
using namespace std;

double Activation::ReLu(double weight) { return (weight > 0) ? weight : 0; }

double Activation::LReLu(double weight) { return (weight > 0) ? weight : 0.01 * weight; }

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
		float weight = exp(*weightsIt) / denominator;
		outputs.push_back(weight);
		advance(weightsIt, 1);
	}

	return outputs;
}

double Activation::Step(double weight)
{
	return static_cast<double>(weight > 0);
}