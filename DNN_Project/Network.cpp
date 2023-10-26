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

void Network::addLayer(Layer layer)
{
	layers.push_back(layer);
}

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

void Network::traverseLayers(Layer layer)
{
}

Layer Network::getLayer(int index)
{
	list<Layer>::iterator layersIt = layers.begin();
	advance(layersIt, index);
	return *layersIt;
}