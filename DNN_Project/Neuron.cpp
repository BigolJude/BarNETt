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
Neuron::Neuron(list<float> inputs) 
{
	this->inputs = inputs;
	this->populateWeights();
	this->weigh();
}

/// <summary>
/// Initialises the Neuron with preset weights.
/// </summary>
/// <param name="inputs"></param>
/// <param name="weights"></param>
Neuron::Neuron(list<float> inputs, list<float> weights)
{
	this->inputs = inputs;
	this->weights = weights;
	this->weigh();
}

/// <summary>
/// Sets the sum of all weights in the Neuron in 'weights'. 
/// </summary>
void Neuron::weigh()
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
	this->weight = output;
}

/// <summary>
/// Runs an initialiser to get the weight ranges.
/// </summary>
void Neuron::populateWeights()
{
	for (int i = 0; i != inputs.size(); ++i)
	{
		int randNumber = rand() % 100 + 1;
		float weightedRange = Initialisation::He(randNumber);
		float weightedNumber = Initialisation::Random(weightedRange, -weightedRange);
		std::cout << weightedNumber << "\n";
		this->weights.push_back(weightedNumber);
	}
}

/// <summary>
/// Trains the singular neuron.
/// </summary>
/// <param name="learningRate"></param>
/// <param name="desired"></param>
void Neuron::train(float learningRate, float desired)
{
	float error = 1;
	float guess = 0;
	this->weigh();
	guess = Activation::ReLu(this->weight);
	error = desired - guess;
	cout << "error: " << error << endl;
	
	list<float>::iterator weightsIt = weights.begin();
	list<float>::iterator inputsIt = inputs.begin();
	
	for (int i = 0; i < inputs.size(); ++i)
	{
		*weightsIt = *weightsIt + (learningRate * error * *inputsIt);
		advance(weightsIt, 1);
		advance(inputsIt, 1);
	}
	cout << "error: " << error << endl;
}

/// <summary>
/// Prints the weights of the Neuron.
/// </summary>
void Neuron::printWeights()
{
	list<float>::iterator weightsIt = weights.begin();

	for (int i = 0; i < inputs.size(); ++i)
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

void Neuron::setInputs(list<float> inputs)
{
	this->inputs = inputs;
}