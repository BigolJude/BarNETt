#include <iostream>
#include <list>
#include "Neuron.h"
#include "Initialisation.h"
#include "Activation.h"


int main()
{
	std::cout << "Hello World" << endl;
	list<float> inputs {0.5, 4.0, 1};
	list<float> weights{ -0.5, 0.2 , 0.4};
	
	Neuron* neuron = new Neuron(inputs);

	float activation = Activation::ReLu(neuron->getWeight());


	neuron->train(0.001, 0.1);

	//list<float> reluValues = Activation::SoftMax(weights);

	delete(neuron);
	inputs.clear();
	weights.clear();

}