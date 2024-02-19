#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#ifdef _DEBUG
#ifndef DBG_NEW
#define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
#define new DBG_NEW
#endif
#endif  // _DEBUG

#include "Network.h"
#include "Layer.h"
#include "Activation.h"
#include "Neuron.h"
#include "Loss.h"
#include <List>
#include <algorithm>
#include <iostream>
using namespace std;

//BarNETt
Network::Network(list<Layer> layers)
{
	this->layers = layers;
	this->mError = 1;
}

Network::Network()
{
	this->mError = 1;
}

Network::~Network()
{
	list<Layer>::iterator layersIt = layers.begin();

	for (int i = 0; i < layers.size(); ++i)
	{
		list<Neuron*> neurons = layersIt->getNeurons();
		list<Neuron*>::iterator neuronsIt = neurons.begin();
		for (int ii = 0; ii < neurons.size(); ++ii)
		{
			delete(*neuronsIt);
			*neuronsIt = nullptr;
			advance(neuronsIt, 1);
		}
		advance(layersIt, 1);
		neurons.clear();
	}
	layers.clear();
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
	this->inputs = inputs;
	this->learningRate = learningRate;
	this->predict(inputs);
	list<double> predictions = Activation::SoftMax(this->getPrediction());
	double loss = Loss::crossEntropy(predictions, expected);
	outputNueronErrors = {};

	list<double>::iterator predictionsIt = predictions.begin();
	list<double>::iterator expectedIt = expected.begin();

	for (int i = 0; i < predictions.size(); ++i)
	{
		//cout << *predictionsIt << " - " << *expectedIt << endl;
		outputNueronErrors.push_back(*predictionsIt - *expectedIt);
		advance(predictionsIt, 1);
		advance(expectedIt, 1);
	}
	//cout << "----" << endl;
	this->mError = loss;
	traverseLayer(0, 0, loss);
	predictions.clear();
}

/// <summary>
/// Weighs all of the neurons in the network.
/// </summary>
/// <param name="inputs"></param>
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

/// <summary>
/// Gets the last layers activations, hence the 'prediction'.
/// </summary>
/// <returns></returns>
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
/// Recursively traverses to the next neuron in the pre-order network traversal.
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
		list<double>::iterator outputNueronErrorsIt = outputNueronErrors.begin();
		for (int i = 0; i < neurons.size(); ++i)
		{
			traverseNeuron(layer, i, layerCount, *outputNueronErrorsIt);
			advance(outputNueronErrorsIt, 1);
		}

		neurons.clear();
	}
	else
	{
		traverseNeuron(layer, weightIndex, layerCount, error);
	}
}

/// <summary>
/// Recursively traverses each weight, calculates the delta, and traverses to the next layer if applicable.
/// </summary>
/// <param name="layer"></param>
/// <param name="neuronIndex"></param>
/// <param name="layerCount"></param>
/// <param name="error"></param>
void Network::traverseNeuron(Layer layer, int neuronIndex, int layerCount, double error)
{
	Neuron* neuron = layer.getNeuron(neuronIndex);
	list<double> weights = neuron->getWeights();
	list<double>::iterator weightsIt = weights.begin();

	for (int weightIndex = 0; weightIndex < weights.size(); ++weightIndex)
	{
		double gradient;
		double activationDerivative = 0;
		if (weightIndex == weights.size() - 1)
		{
			// Bias weight.
			activationDerivative = neuron->getActivationOutput() * (1 - neuron->getActivationOutput());
			errors.push_front(activationDerivative);
			gradient = this->backpropogate();
			neuron->trainWeight(weightIndex, learningRate, gradient);
		}
		else if (layerCount == 0)
		{
			// Last layer
			Layer layer = this->getLayer(layers.size() - 2);
			Neuron* connectingNeuron = layer.getNeuron(weightIndex);

			double outputDerivative = error * connectingNeuron->getActivationOutput();
			activationDerivative = error * neuron->getWeight(weightIndex);
			errors.push_front(activationDerivative);

			neuron->trainWeight(weightIndex, learningRate, outputDerivative);
		}
		else if (layerCount == layers.size() - 1)
		{
			// First Layer
			list<double>::iterator inputsIt = inputs.begin();
			advance(inputsIt, weightIndex);

			activationDerivative = neuron->getActivationOutput() * (1 - neuron->getActivationOutput());
			activationDerivative = activationDerivative * *inputsIt;
			errors.push_front(activationDerivative);
			gradient = this->backpropogate();
			//cout << gradient << endl;

			neuron->trainWeight(weightIndex, learningRate, gradient);
		}
		else
		{
			// Middle Layer
			Layer layer = this->getLayer((layers.size() - 2) - layerCount);
			Neuron* connectingNeuron = layer.getNeuron(weightIndex);

			activationDerivative = neuron->getActivationOutput() * (1 - neuron->getActivationOutput());
			activationDerivative = activationDerivative * connectingNeuron->getActivationOutput();
			errors.push_front(activationDerivative * neuron->getWeight(weightIndex));

			gradient = this->backpropogate();
			errors.pop_front();

			errors.push_front(activationDerivative * neuron->getWeight(weightIndex));
			neuron->trainWeight(weightIndex, learningRate, gradient);
		}

		//cout << "Layer: " << layerCount << " - Neuron: " << neuronIndex << " - Weight: " << weightIndex << endl;
		//cout <<	"Gradient: " << gradient << endl;
		//cout << "--------" << endl;

		if (weightIndex < weights.size() - 1)
		{
			traverseLayer(layerCount + 1, weightIndex, error);
		}
		errors.pop_front();
		advance(weightsIt, 1);
	}

	neuron = nullptr;
	weights.clear();
}

/// <summary>
/// Multiplies each delta together and returns the output.
/// </summary>
/// <returns></returns>
double Network::backpropogate()
{
	list<double>::iterator errorsIt = errors.begin();
	double weightedTotal = 1;
	for (int i = 0; i < errors.size(); ++i)
	{
		//cout << "derivative: " << *errorsIt << endl;
		weightedTotal = weightedTotal * *errorsIt;
		advance(errorsIt, 1);
	}
	//cout << "------" << endl;
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