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
}

Network::Network(){}

void Network::train(list<float> inputs, float learningRate, list<float> expected)
{
	this->learningRate = learningRate;
	this->predict(inputs);
	list<float> predictions = Activation::SoftMax(this->getPrediction());
	float error = Loss::crossEntropy(predictions, expected);
	cout << "error: " << error << endl;
	traverseLayer(0, 0, error);
}

void Network::predict(list<float> inputs)
{
	list<Layer>::iterator layersIt = layers.begin();

	for (int i = 0; i < layers.size(); ++i)
	{
		layersIt->weigh(inputs);
		inputs = layersIt->getNeuronWeights();
		advance(layersIt, 1);
	}
}

list<float> Network::getPrediction()
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

int Network::getMax(list<float> numbers)
{
	list<float>::iterator numbersIt = numbers.begin();

	float largestNumber = 0;

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
void Network::traverseLayer(int layerCount, int weightIndex, float error)
{
	if (layerCount >= layers.size())
	{
		return;
	}

	Layer layer = this->getLayer((layers.size() - 1)- layerCount);

	// Traversing all values on the last layer.
	if (layerCount == 0)
	{
		list<Neuron> neurons = layer.getNeurons();
		list<Neuron>::iterator neuronsIt = neurons.begin();

		for (int i = 0; i < neurons.size(); ++i)
		{
			traverseNeuron(layer, weightIndex, layerCount, error);
		}
	}
	else
	{
		traverseNeuron(layer, weightIndex, layerCount, error);
	}
}

void Network::traverseNeuron(Layer layer, int weightIndex, int layerCount, float error)
{
	Neuron neuron = layer.getNeuron(weightIndex);
	list<float> weights = neuron.getWeights();
	list<float>::iterator weightsIt = weights.begin();

	for (int j = 0; j < weights.size(); ++j)
	{
		float proportionalError = backpropogate(neuron, *weightsIt, error);
		neuron.train(learningRate, proportionalError);
		errors.push_front(proportionalError);

		cout << "layer:" << layerCount << " neuron: " << weightIndex << " weight: " << j << endl;

		traverseLayer(layerCount + 1, j, error);

		advance(weightsIt, 1);
		errors.pop_front();
	}
}

float Network::backpropogate(Neuron neuron, float weight, float error)
{
	list<float>::iterator errorsIt = errors.begin();
	float weightedTotal = 1;
	for (int i = 0; i < errors.size(); ++i)
	{
		weightedTotal = weightedTotal * *errorsIt;
		advance(errorsIt, 1);
		cout << "Weighted Total:" << weightedTotal << endl;
	}
	return weightedTotal * weight * neuron.getWeight();
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