#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#ifdef _DEBUG
#ifndef DBG_NEW
#define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
#define new DBG_NEW
#endif
#endif  // _DEBUG

#include "Layer.h"
#include <List>
#include <iostream>
using namespace std;

/// <summary>
/// Instantiates a new layer and generates the neurons.
/// </summary>
/// <param name="previousLayerCount"> - The neuron count of the previous layer (place input count here if first layer)</param>
/// <param name="neuronCount"> - The intended amount of neurons for this layer.</param>
/// <param name="activation"> - Currently a non functional label for clarity.</param>
Layer::Layer(int previousLayerCount, int neuronCount, string activation)
{
	this->activation = activation;
	this->generateNeurons(previousLayerCount, neuronCount);
}

/// <summary>
/// Instantiates a new Layer with a list pre-set neurons.
/// </summary>
/// <param name="neurons"> - List of pointers to neurons to use within the layer.</param>
/// <param name="activation"> - Currently a non functional label for clarity.</param>
Layer::Layer(list<Neuron*> neurons, string activation)
{
	this->neurons = neurons;
	this->activation = activation;
}

/// <summary>
/// Generates a set neurons.
/// </summary>
/// <param name="previousLayerCount"> - The neuron count of the previous layer (place input count here if first layer)</param>
/// <param name="neuronCount"> - The intended amount of neurons for this layer.</param>
void Layer::generateNeurons(int previousLayerCount,int neuronCount)
{
	for (int i = 0; i < neuronCount; ++i)
	{
		Neuron* neuron = new Neuron(previousLayerCount);
		neurons.push_back(neuron);
	}
}

/// <summary>
/// Gets all of the weights (pre-activation) of each neuron in the layer.
/// </summary>
/// <returns></returns>
list<double> Layer::getNeuronWeights()
{
	list<Neuron*>::iterator neuronsIt = neurons.begin();
	list<double>* weights = new list<double>();

	for (int i = 0; i < neurons.size(); ++i)
	{
		double weight = neuronsIt._Ptr->_Myval->getOutput();

		weights->push_back(weight);
		advance(neuronsIt, 1);
	}
	return *weights;
}

/// <summary>
/// Gets all of the activation outputs for each neuron in the layer
/// </summary>
/// <returns></returns>
list<double> Layer::getActivationOutputs()
{
	list<Neuron*>::iterator neuronsIt = neurons.begin();
	list<double> outputs = list<double>();

	for (int i = 0; i < neurons.size(); ++i)
	{
		outputs.push_back(neuronsIt._Ptr->_Myval->getActivationOutput());
		advance(neuronsIt, 1);
	}
	return outputs;
}

/// <summary>
/// Goes through every neuron and weighs them to the inputs.
/// </summary>
/// <param name="inputs"> - the inputs for the neurons to weigh.</param>
void Layer::weigh(list<double> inputs)
{
	list<Neuron*>::iterator neuronsIt = neurons.begin();

	for (int i = 0; i < neurons.size(); ++i)
	{
		//std::cout << " Neuron: " << i << endl;
		neuronsIt._Ptr->_Myval->weigh(inputs);
		advance(neuronsIt, 1);
	}
}

/// <summary>
/// Gets the pointer of a nueron within the layer given it's index.
/// </summary>
/// <param name="index"> - The index of the neuron.</param>
/// <returns>The ptr to a neuron.</returns>
Neuron* Layer::getNeuron(int index)
{
	list<Neuron*>::iterator neuronIt = neurons.begin();
	advance(neuronIt, index);
	return *neuronIt;
}

/// <summary>
/// Gets all of the pointers to neurons within a layer.
/// </summary>
/// <returns>A list of pointers to neurons.</returns>
list<Neuron*> Layer::getNeurons()
{
	return this->neurons;
}