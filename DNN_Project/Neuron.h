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
	double getOutput();
	double getWeight(int weightIndex);
	void printWeights();
	list<double> getWeights();
	void populateWeights(int neuronCount);
	double getActivationOutput();
private:
	double weight;
	double activationOutput;
	list<double> weights;
};