#pragma once
#include <list>
using namespace std;

class Neuron
{
public:
	Neuron(int inputs);
	Neuron(list<float> weights);
	void weigh(list<float> inputs);
	void train(float learningRate, float error);
	float getWeight();
	void setWeights(list<float> weights);
	void printWeights();
	list<float> getWeights();
	void populateWeights(int neuronCount);
private:
	float weight;
	list<float> weights;
};