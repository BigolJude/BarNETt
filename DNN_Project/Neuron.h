#pragma once
#include <list>
using namespace std;

class Neuron
{
public:
	Neuron(list<float> inputs);
	Neuron(list<float> inputs, list<float> weights);
	void setWeights(list<float> weights);
	void setInputs(list<float> weights);
	float getWeight();
	list<float> getWeights();
	void weigh();
	void printWeights();
	void populateWeights();
	void train(float learningRate, float desired);
private:
	float weight;
	list<float> weights;
	list<float> inputs;
};