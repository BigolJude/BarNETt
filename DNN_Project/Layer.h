#pragma once
#include "Neuron.h"
#include <list>
#include <string>
using namespace std;

class Layer
{
public:
	Layer(int previousLayerCount, int neuronCount, double biasWeight, string activation);
	Layer(list<Neuron*> neurons, double biasWeight, string activation);
	list<double> getNeuronWeights();
	list<double> getActivationOutputs();
	void weigh(list<double> inputs);
	Neuron* getNeuron(int index);
	list<Neuron*> getNeurons();
private:
	void generateNeurons(int previousLayerCount, int neuronCount);
	list<Neuron*> neurons;
	string activation;
	double biasWeight;
};

