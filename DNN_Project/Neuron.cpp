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
Neuron::Neuron(list<double> weights)
{
	this->weights = weights;
}

/// <summary>
/// Sets the sum of all weights in the Neuron in 'weights'. 
/// </summary>
void Neuron::weigh(list<double> inputs, double biasWeight)
{
	list<double>::iterator weightsIt = weights.begin();
	list<double>::iterator inputsIt = inputs.begin();
	float output = 0;

	for (int i = 0; i < inputs.size(); ++i)
	{
		output = output + (*inputsIt * *weightsIt);
		//cout << "  " << i << "." << "Weight: " << *weightsIt << endl;
		advance(weightsIt, 1);
		advance(inputsIt, 1);
	}

	// Including the weighted bias.
	//output = output + (1 * biasWeight);
	//cout << "Bias Weight: "<< *weightsIt << endl;
	//cout << "Output: " << output << endl;
	//cout << "-------" << endl;

	this->weight = output;
	this->activationOutput = Activation::LReLu(this->weight);
}

/// <summary>
/// Runs an initialiser to get the weight ranges.
/// </summary>
void Neuron::populateWeights(int neuronCount)
{
	double weightedRange = Initialisation::He(neuronCount);
	for (int i = 0; i != neuronCount; ++i)
	{
		double weightedNumber = Initialisation::Random(weightedRange, -weightedRange);
		this->weights.push_back(weightedNumber);
	}
}

/// <summary>
/// Trains the singular neuron.
/// </summary>
/// <param name="learningRate"></param>
/// <param name="error"></param>
void Neuron::trainWeight(int weightIndex, float learningRate, double gradient)
{	
	//cout << "-------" << endl;
	list<double>::iterator weightsIt = weights.begin();
	advance(weightsIt, weightIndex);
	//cout << "Weight before: " << *weightsIt << endl;
	*weightsIt = *weightsIt - learningRate * gradient;
	//cout << "Weight after: " << *weightsIt << endl;
	if (isnan(*weightsIt))
	{
		//Breakpoint
	}
}

/// <summary>
/// Prints the weights of the Neuron.
/// </summary>
void Neuron::printWeights()
{
	list<double>::iterator weightsIt = weights.begin();

	for (int i = 0; i < weights.size(); ++i)
	{
		cout << *weightsIt << endl;
		advance(weightsIt, 1);
	}
}

/// <summary>
/// Gets a specific weight given it's index.
/// </summary>
/// <param name="weightIndex"></param>
/// <returns></returns>
double Neuron::getWeight(int weightIndex)
{
	list<double>::iterator weightsIt = weights.begin();
	advance(weightsIt, weightIndex);
	return *weightsIt;
}

/// <summary>
/// Gets the output of the neuron before activation.
/// </summary>
/// <returns>The output of the neuron before activation.</returns>
double Neuron::getOutput()
{
	return this->weight;
}

/// <summary>
/// Gets the output of the neuron after activation.
/// </summary>
/// <returns>The output of the neuron after activation.</returns>
double Neuron::getActivationOutput()
{
	return this->activationOutput;
}

/// <summary>
/// Gets all of the weights within the neuron.
/// </summary>
/// <returns></returns>
list<double> Neuron::getWeights()
{
	return this->weights;
}

/// <summary>
/// Sets all of the weights within the neuron.
/// </summary>
/// <param name="weights"></param>
void Neuron::setWeights(list<double> weights)
{
	this->weights = weights;
}