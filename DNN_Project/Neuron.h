#pragma once
#include <list>
using namespace std;

class Neuron
{
public:
	Neuron(int inputs);
	Neuron(list<double> weights);
	void weigh(list<double> inputs);
	void trainWeight(int weightIndex, float learningRate, double gradient);
	void setWeights(list<double> weights);
	double getOutput();
	double getWeight(int weightIndex);
	void printWeights();
	list<double> getWeights();
	void populateWeights(int neuronCount);
	double getActivationOutput();
private:
	double output;
	double activationOutput;
	list<double> weights;
};