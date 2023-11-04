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

	// Including the weighted bias.
	output = output + (1 * *weightsIt);

	this->weight = output;
	this->activationOutput = Activation::ReLu(this->weight);
}

/// <summary>
/// Runs an initialiser to get the weight ranges.
/// </summary>
void Neuron::populateWeights(int neuronCount)
{
	float weightedRange = Initialisation::He(neuronCount);
	for (int i = 0; i != neuronCount; ++i)
	{
		float weightedNumber = Initialisation::Random(weightedRange, -weightedRange);
		this->weights.push_back(weightedNumber);
	}
	//TODO implement bias this is not how it works.
	float biasWeightedNumber = Initialisation::Random(weightedRange, -weightedRange);
	this->weights.push_back(biasWeightedNumber);
}

/// <summary>
/// Trains the singular neuron.
/// </summary>
/// <param name="learningRate"></param>
/// <param name="error"></param>
void Neuron::trainWeight(int weightIndex, float learningRate, float error)
{	
	list<float>::iterator weightsIt = weights.begin();
	advance(weightsIt, weightIndex);
	*weightsIt = *weightsIt - (learningRate * error);
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

float Neuron::getActivationOutput()
{
	return this->activationOutput;
}

list<float> Neuron::getWeights()
{
	return this->weights;
}

void Neuron::setWeights(list<float> weights)
{
	this->weights = weights;
}