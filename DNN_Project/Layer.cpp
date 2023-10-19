#include "Layer.h"

Layer::Layer(int previousLayerCount, int neuronCount, string activation)
{
	this->activation = activation;
	this->generateNeurons(previousLayerCount, neuronCount);
}

Layer::Layer(list<Neuron> neurons, string activation)
{
	this->activation = activation;
	this->neurons = neurons;
}

void Layer::generateNeurons(int previousLayerCount,int neuronCount)
{
	for (int i = 0; i < neuronCount; ++i)
	{
		Neuron* neuron = new Neuron(previousLayerCount);
		neurons.push_back(*neuron);
		delete(neuron);
	}
}

list<float> Layer::getNeuronWeights()
{
	list<Neuron>::iterator neuronsIt = neurons.begin();
	list<float>* weights = new list<float>();

	for (int i = 0; i < neurons.size(); ++i)
	{
		weights->push_back(neuronsIt->getWeight());
		advance(neuronsIt, 1);
	}
	return *weights;
}

void Layer::train(list<float> inputs, float learningRate, float expected)
{
	list<Neuron>::iterator neuronsIt = neurons.begin();

	for (int i = 0; i < neurons.size(); ++i)
	{
		neuronsIt->train(inputs, learningRate, expected);
		advance(neuronsIt, 1);
	}
}

void Layer::weigh(list<float> inputs)
{
	list<Neuron>::iterator neuronsIt = neurons.begin();

	for (int i = 0; i < neurons.size(); ++i)
	{
		neuronsIt->weigh(inputs);
		advance(neuronsIt, 1);
	}
}