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
	this->predict(inputs);
	list<float> predictions = Activation::SoftMax(this->getPrediction());
	float error = Loss::crossEntropy(predictions, expected);
	float outNet = error * (1 - error);

	list<Layer>::iterator layersIt = layers.begin();

	for (int i = 0; i < layers.size(); ++i)
	{
		layersIt->train(learningRate, 0.1);
		inputs = layersIt->getNeuronWeights();
		advance(layersIt, 1);
	}
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
void Network::traverseLayers(int layerCount, int weightIndex)
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
			Neuron neuron = layer.getNeuron(i);
			list<float> weights = neuron.getWeights();

			for (int j = 0; j < weights.size(); ++j)
			{
				cout << "layer:" << layerCount << " neuron: " << i << " weight: " << j << endl;
				traverseLayers(layerCount + 1, j);
			}
		}
	}
	else
	{
		Neuron neuron = layer.getNeuron(weightIndex);
		list<float> weights = neuron.getWeights();

		for (int j = 0; j < weights.size(); ++j)
		{
			cout << "layer:" << layerCount << " neuron: " << weightIndex << " weight: " << j << endl;
			traverseLayers(layerCount + 1, j);
		}
	}
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