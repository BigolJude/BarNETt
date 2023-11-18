#pragma once
#include <list>
using namespace std;

class Neuron
{
public:
	Neuron(int inputs);
	Neuron(list<double> weights);
	void weigh(list<double> inputs, double biasWeight);
	void trainWeight(int weightIndex, float learningRate, double gradient);
	void setWeights(list<double> weights);
	float getWeight();
	void printWeights();
	list<double> getWeights();
	void populateWeights(int neuronCount);
	float getActivationOutput();
private:
	double weight;
	double activationOutput;
	list<double> weights;
};