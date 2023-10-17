#pragma once
#include <list>
using namespace std;

class Neuron
{
public:
	Neuron(list<float> inputs);
	Neuron(list<float> inputs, list<float> weights);
	
	void weigh();
	void populateWeights();
private:
	bool activated;
	float weight;
	list<float> weights;
	list<float> inputs;
};