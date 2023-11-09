#include "Network.h"
#include "Layer.h"
#include "Activation.h"
#include "Neuron.h"
#include "Loss.h"
#include <List>
#include <algorithm>
#include <iostream>
using namespace std;

Network::Network(list<Layer> layers)
{
	this->layers = layers;
	this->error = 1;
}

Network::Network()
{
	this->error = 1;
}

void Network::train(list<double> inputs, float learningRate, list<double> expected)
{
	this->learningRate = learningRate;
	this->predict(inputs);
	list<double> predictions = Activation::SoftMax(this->getPrediction());
	double error = Loss::crossEntropy(predictions, expected);

	list<double>::iterator predictionsIt = predictions.begin();
	list<double>::iterator expectedIt = expected.begin();

	for (int i = 0; i < predictions.size(); ++i)
	{
		cout << *predictionsIt << " - " << *expectedIt << endl;
		advance(predictionsIt, 1);
		advance(expectedIt, 1);
	}
	cout << "Error: " << error;
	cout << "-------" << endl;

	this->error = error;
	traverseLayer(0, 0, error);
	predictions.clear();
}

void Network::predict(list<double> inputs)
{
	list<Layer>::iterator layersIt = layers.begin();

	for (int i = 0; i < layers.size(); ++i)
	{
		layersIt->weigh(inputs);
		inputs = layersIt->getActivationOutputs();
		advance(layersIt, 1);
	}
}

list<double> Network::getPrediction()
{
	Layer outputLayer = layers.back();
	return outputLayer.getNeuronWeights();
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

	Layer layer = this->getLayer((layers.size() - 1)- layerCount);

	// Traversing all values on the last layer.
	if (layerCount == 0)
	{
		list<Neuron*> neurons = layer.getNeurons();
		list<Neuron*>::iterator neuronsIt = neurons.begin();

		for (int i = 0; i < neurons.size(); ++i)
		{
			traverseNeuron(layer, weightIndex, layerCount, error);
		}
		neurons.clear();
	}
	else
	{
		traverseNeuron(layer, weightIndex, layerCount, error);
	}
}

void Network::traverseNeuron(Layer layer, int weightIndex, int layerCount, double error)
{
	Neuron* neuron = layer.getNeuron(weightIndex);
	list<double> weights = neuron->getWeights();
	list<double>::iterator weightsIt = weights.begin();

	for (int j = 0; j < weights.size(); ++j)
	{
		double proportionalError = backpropogate(neuron, *weightsIt, error);
		neuron->trainWeight(j, learningRate, proportionalError);
		errors.push_front(proportionalError);

		if (j < weights.size()-1)
		{
			//cout << "Layer: " << layerCount <<" - Neuron: " << weightIndex << " - Weight: " << j << " - Proportional Error: " << proportionalError << endl;
			traverseLayer(layerCount + 1, j, error);
			advance(weightsIt, 1);
		}
		errors.pop_front();
	}
	neuron = nullptr;
	weights.clear();
}

double Network::backpropogate(Neuron* neuron, double weight, double error)
{
	list<double>::iterator errorsIt = errors.begin();
	double weightedTotal = 1;
	for (int i = 0; i < errors.size(); ++i)
	{
		weightedTotal = weightedTotal * *errorsIt;
		advance(errorsIt, 1);
		//cout << "Weighted Total:" << weightedTotal << endl;
	}
	return weightedTotal * weight * neuron->getWeight() * error;
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
	return error;
}