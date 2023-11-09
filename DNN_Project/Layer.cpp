#include "Layer.h"
#include <List>
using namespace std;

Layer::Layer(int previousLayerCount, int neuronCount, string activation)
{
	this->activation = activation;
	this->generateNeurons(previousLayerCount, neuronCount);
}

Layer::Layer(list<Neuron*> neurons, string activation)
{
	this->activation = activation;
	this->neurons = neurons;
}

void Layer::generateNeurons(int previousLayerCount,int neuronCount)
{
	for (int i = 0; i < neuronCount; ++i)
	{
		Neuron* neuron = new Neuron(previousLayerCount);
		neurons.push_back(neuron);
	}
}

list<double> Layer::getNeuronWeights()
{
	list<Neuron*>::iterator neuronsIt = neurons.begin();
	list<double>* weights = new list<double>();

	for (int i = 0; i < neurons.size(); ++i)
	{
		double weight = neuronsIt._Ptr->_Myval->getWeight();

		weights->push_back(weight);
		advance(neuronsIt, 1);
	}
	return *weights;
}

list<double> Layer::getActivationOutputs()
{
	list<Neuron*>::iterator neuronsIt = neurons.begin();
	list<double>* outputs = new list<double>();

	for (int i = 0; i < neurons.size(); ++i)
	{
		outputs->push_back(neuronsIt._Ptr->_Myval->getActivationOutput());
		advance(neuronsIt, 1);
	}
	return *outputs;
}

void Layer::weigh(list<double> inputs)
{
	list<Neuron*>::iterator neuronsIt = neurons.begin();

	for (int i = 0; i < neurons.size(); ++i)
	{
		neuronsIt._Ptr->_Myval->weigh(inputs);
		advance(neuronsIt, 1);
	}
}

Neuron* Layer::getNeuron(int index)
{
	list<Neuron*>::iterator neuronIt = neurons.begin();
	advance(neuronIt, index);
	return *neuronIt;
}

list<Neuron*> Layer::getNeurons()
{
	return this->neurons;
}