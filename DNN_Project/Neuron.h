#pragma once
#include <list>
using namespace std;

class Neuron
{
public:
	Neuron(int inputs);
	Neuron(list<float> weights);
	void weigh(list<float> inputs);
	void trainWeight(int weightIndex, float learningRate, float error);
	void setWeights(list<float> weights);
	float getWeight();
	void printWeights();
	list<float> getWeights();
	void populateWeights(int neuronCount);
	float getActivationOutput();
private:
	float weight;
	float activationOutput;
	list<float> weights;
};