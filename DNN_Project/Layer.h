#pragma once
#include "Neuron.h"
#include <list>
#include <string>
using namespace std;

class Layer
{
public:
	Layer(int previousLayerCount, int neuronCount, string activation);
	Layer(list<Neuron> neurons, string activation);
	list<float> getNeuronWeights();
	void train(list<float> inputs, float learningRate, float expected);
	void weigh(list<float> inputs);
private:
	void generateNeurons(int previousLayerCount, int neuronCount);
	list<Neuron> neurons;
	string activation;
};

