#include "Neuron.h"
#include "Initialisation.h"
#include "Activation.h"
#include <cmath>
#include <list>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>

/// <summary>
/// Initialises the Neuron with random weights to start.
/// </summary>
/// <param name="inputs"></param>
Neuron::Neuron(int neuronCount) 
{
	this->populateWeights(neuronCount);
}

/// <summary>
/// Initialises the Neuron with preset weights.
/// </summary>
/// <param name="inputs"></param>
/// <param name="weights"></param>
Neuron::Neuron(list<float> weights)
{
	this->weights = weights;
}

/// <summary>
/// Sets the sum of all weights in the Neuron in 'weights'. 
/// </summary>
void Neuron::weigh(list<float> inputs)
{
	list<float>::iterator weightsIt = weights.begin();
	list<float>::iterator inputsIt = inputs.begin();
	float output = 0;

	for (int i = 0; i < inputs.size(); ++i)
	{
		output = output + (*inputsIt * *weightsIt);
		advance(weightsIt, 1);
		advance(inputsIt, 1);
	}
	this->weight = Activation::ReLu(output);
}

/// <summary>
/// Runs an initialiser to get the weight ranges.
/// </summary>
void Neuron::populateWeights(int neuronCount)
{
	for (int i = 0; i != neuronCount; ++i)
	{
		int randNumber = rand() % 100 + 1;
		float weightedRange = Initialisation::He(randNumber);
		float weightedNumber = Initialisation::Random(weightedRange, -weightedRange);
		this->weights.push_back(weightedNumber);
	}
	this->weights.push_back(1);
}

/// <summary>
/// Trains the singular neuron.
/// </summary>
/// <param name="learningRate"></param>
/// <param name="error"></param>
void Neuron::train(float learningRate, float error)
{	
	list<float>::iterator weightsIt = weights.begin();

	for (int i = 0; i < weights.size(); ++i)
	{
		*weightsIt = *weightsIt - (learningRate * error);

		advance(weightsIt, 1);
	}
}

/// <summary>
/// Prints the weights of the Neuron.
/// </summary>
void Neuron::printWeights()
{
	list<float>::iterator weightsIt = weights.begin();

	for (int i = 0; i < weights.size(); ++i)
	{
		cout << *weightsIt << endl;
		advance(weightsIt, 1);
	}
}

float Neuron::getWeight()
{
	return this->weight;
}

list<float> Neuron::getWeights()
{
	return this->weights;
}

void Neuron::setWeights(list<float> weights)
{
	this->weights = weights;
}