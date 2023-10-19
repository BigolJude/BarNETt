#include "Network.h"
#include "Layer.h"
#include <List>

Network::Network(list<Layer> layers)
{
	this->layers = layers;
}

Network::Network(){}

void Network::train(list<float> inputs, float learningRate, float expected)
{
	list<Layer>::iterator layersIt = layers.begin();

	for (int i = 0; i < layers.size(); ++i)
	{
		layersIt->train(inputs, learningRate, expected);
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