#include "Activation.h"
#include <cmath>
#include <list>
#include <iostream>
using namespace std;

float Activation::ReLu(float weight)
{
	if (weight > 0)
	{
		return weight;
	}
	else
	{
		return 0;
	}
}

list<float> Activation::SoftMax(list<float> weights)
{
	list<float>::iterator weightsIt = weights.begin();
	
	float denominator = 0;

	for (int i = 0; i < weights.size(); ++i)
	{
		denominator = denominator + exp(*weightsIt);
		advance(weightsIt, 1);
	}

	weightsIt = weights.begin();
	list<float> outputs;

	cout << "Denominator" << denominator << endl;

	for (int i = 0; i < weights.size(); ++i)
	{
		float weight = exp(*weightsIt) / denominator;
		outputs.push_back(weight);
		advance(weightsIt, 1);
		cout << "Weight from softmax" << weight << endl;
	}

	return outputs;
}

float Activation::Step(float weight)
{
	return static_cast<float>(weight > 0);
}