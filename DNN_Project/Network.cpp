#include "Network.h"
#include "Layer.h"
#include "Activation.h"
#include "Neuron.h"
#include "Loss.h"
#include <List>
#include <algorithm>
#include <iostream>
using namespace std;

//BarNett
Network::Network(list<Layer> layers)
{
	this->layers = layers;
	this->mError = 1;
}

Network::Network()
{
	this->mError = 1;
}

/// <summary>
/// Trains the network given a valid set of inputs and valid set of outputs.
/// </summary>
/// <remarks>Current network is fixed to softmax.</remarks>
/// <param name="inputs"> - list of inputs of the same sizes as first layer.</param>
/// <param name="learningRate"> - The learning rate.</param>
/// <param name="expected"> - list of expected values of the same size as the last layer.</param>
void Network::train(list<double> inputs, float learningRate, list<double> expected)
{
	this->learningRate = learningRate;
	this->predict(inputs);
	list<double> predictions = Activation::SoftMax(this->getPrediction());
	double loss = Loss::crossEntropy(predictions, expected);
	outputNueronErrors = {};

	list<double>::iterator predictionsIt = predictions.begin();
	list<double>::iterator expectedIt = expected.begin();

	for (int i = 0; i < predictions.size(); ++i)
	{
		cout << *predictionsIt << " - " << *expectedIt << endl;
		outputNueronErrors.push_back(*predictionsIt - *expectedIt);
		advance(predictionsIt, 1);
		advance(expectedIt, 1);
	}

	this->mError = loss;
	traverseLayer(0, 0, loss);
	predictions.clear();
}

void Network::predict(list<double> inputs)
{
	list<Layer>::iterator layersIt = layers.begin();

	for (int i = 0; i < layers.size(); ++i)
	{
		//cout << "Layer: " << i << endl;
		layersIt->weigh(inputs);
		inputs = layersIt->getActivationOutputs();
		advance(layersIt, 1);
	}
}

list<double> Network::getPrediction()
{
	Layer outputLayer = layers.back();
	return outputLayer.getActivationOutputs();
}

/// <summary>
/// Adds a layer to the network given a layer object.
/// </summary>
/// <param name="layer"></param>
void Network::addLayer(Layer layer)
{
	layers.push_back(layer);
}

/// <summary>
/// Creates and adds a layer to the network given the layer's properties.
/// </summary>
/// <param name="previousLayerCount"></param>
/// <param name="neuronCount"></param>
/// <param name="activation"></param>
void Network::addLayer(int previousLayerCount, int neuronCount, string activation)
{
	Layer* layer = new Layer(previousLayerCount, neuronCount, activation);
	layers.push_back(*layer);
	delete(layer);
}

int Network::getMax(list<double> numbers)
{
	list<double>::iterator numbersIt = numbers.begin();

	double largestNumber = 0;

	for (int i = 0; i < numbers.size(); ++i)
	{
		if (largestNumber > *numbersIt)
		{
			largestNumber = i;
		}
		advance(numbersIt, 1);
	}
	return largestNumber;
}

/// <summary>
/// 
/// </summary>
/// <param name="layerCount"></param>
/// <param name="weightIndex"></param>
void Network::traverseLayer(int layerCount, int weightIndex, double error)
{
	if (layerCount >= layers.size())
	{
		return;
	}

	Layer layer = this->getLayer((layers.size() - 1) - layerCount);

	// Traversing all values on the last layer.
	if (layerCount == 0)
	{
		list<Neuron*> neurons = layer.getNeurons();
		list<double>::iterator softmaxDerivationsIt = outputNueronErrors.begin();
		for (int i = 0; i < neurons.size(); ++i)
		{
			errors.push_back(*softmaxDerivationsIt);
			traverseNeuron(layer, i, layerCount, error);
			errors.pop_back();
			advance(softmaxDerivationsIt, 1);
		}

		neurons.clear();
	}
	else
	{
		traverseNeuron(layer, weightIndex, layerCount, error);
	}
}

void Network::traverseNeuron(Layer layer, int neuronIndex, int layerCount, double error)
{
	Neuron* neuron = layer.getNeuron(neuronIndex);
	list<double> weights = neuron->getWeights();
	list<double>::iterator weightsIt = weights.begin();

	for (int weightIndex = 0; weightIndex < weights.size(); ++weightIndex)
	{
		double activationDerivative = neuron->getActivationOutput() * (1 - neuron->getActivationOutput());
		errors.push_front(activationDerivative);
		double gradient = backpropogate();
		neuron->trainWeight(weightIndex, learningRate, gradient);

		//cout << "Layer: " << layerCount << " - Neuron: " << neuronIndex << " - Weight: " << weightIndex << endl;
		//cout <<	" Proportional Error: " << gradient << endl;

		if (weightIndex < weights.size() - 1)
		{
			traverseLayer(layerCount + 1, weightIndex, error);
		}
		errors.pop_front();
		advance(weightsIt, 1);
	}

	neuron = nullptr;
	delete(neuron);
	weights.clear();
}

double Network::backpropogate()
{
	list<double>::iterator errorsIt = errors.begin();
	double weightedTotal = 1;
	for (int i = 0; i < errors.size(); ++i)
	{
		weightedTotal = weightedTotal * *errorsIt;
		advance(errorsIt, 1);
	}

	return weightedTotal;
}

Layer Network::getLayer(int index)
{
	list<Layer>::iterator layersIt = layers.begin();

	if(index != 0)
	{
		advance(layersIt, index);
	}

	return *layersIt;
}

double Network::getError()
{
	return mError;
}