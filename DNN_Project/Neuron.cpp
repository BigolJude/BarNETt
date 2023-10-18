#include "Neuron.h"
#include "Initialisation.h"
#include "Activation.h"
#include <cmath>
#include <list>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>

Neuron::Neuron(list<float> inputs) 
{
	this->inputs = inputs;
	this->populateWeights();
	this->weigh();
}

Neuron::Neuron(list<float> inputs, list<float> weights)
{
	this->inputs = inputs;
	this->weights = weights;
	this->weigh();
}

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
void Neuron::train(float learningRate, float desired)
{
	float error = 1;
	float guess = 0;
	while(abs(error) > 0.000001)
	{
		this->weigh();
		guess = Activation::ReLu(this->weight);
		cout << "guess: " << guess << endl;
		error = desired - guess;
		cout << "error: " << error << endl;

		list<float>::iterator weightsIt = weights.begin();
		list<float>::iterator inputsIt = inputs.begin();

		for (int i = 0; i < inputs.size(); ++i)
		{
			cout << i << "." << *weightsIt << endl;
			*weightsIt = *weightsIt + (learningRate * error * *inputsIt);
			advance(weightsIt, 1);
			advance(inputsIt, 1);
		}
	}
	cout << "guess: " << guess << endl;
	cout << "error: " << error << endl;
}

float Neuron::getWeight()
{
	return this->weight;
}