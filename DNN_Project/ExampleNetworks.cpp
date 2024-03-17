#include "ExampleNetworks.h"
#include "Layer.h"
#include "Neuron.h"
#include "CSV.h"
#include <random>
using namespace std;

/// <summary>
/// A 4 layer network made for the iris dataset.
/// </summary>
/// <returns>A 4 layer network.</returns>
Network* ExampleNetworks::irisDatasetNetwork()
{
	Layer* layer1 = new Layer(4, 4, "relu");
	Layer* layer2 = new Layer(4, 10, "relu");
	Layer* layer3 = new Layer(10, 4, "relu");
	Layer* layer4 = new Layer(4, 3,  "relu");

	Network* network = new Network();

	network->addLayer(*layer1);
	network->addLayer(*layer2);
	network->addLayer(*layer3);
	network->addLayer(*layer4);

	return network;
}

/// <summary>
/// A 2 by 2 network with preset values.
/// </summary>
/// <returns>Network with preset values.</returns>
Network* ExampleNetworks::network2by2()
{
	Neuron* neuron1 = new Neuron({ 0.1, 0.2 });
	Neuron* neuron2 = new Neuron({ 0.3, 0.4 });
	Neuron* neuron3 = new Neuron({ 0.5, 0.6 });
	Neuron* neuron4 = new Neuron({ 0.7, 0.8 });	
	
	list<Neuron*> neurons1 = { neuron1, neuron2 }; 
	list<Neuron*> neurons2 = { neuron3, neuron4 };

	Layer* layer1 = new Layer(neurons1, "relu");
	Layer* layer2 = new Layer(neurons2, "relu");
	
	Network* network = new Network();

	network->addLayer(*layer1);
	network->addLayer(*layer2);

	return network;
}

/// <summary>
/// A 3 by 2 network with initalised values.
/// </summary>
/// <returns>Network with initalised values.</returns>
Network* ExampleNetworks::network3by2()
{
	Layer* layer1 = new Layer(2, 2, "relu");
	Layer* layer2 = new Layer(2, 2, "relu");
	Layer* layer3 = new Layer(2, 2, "relu");

	Network* network = new Network();

	network->addLayer(*layer1);
	network->addLayer(*layer2);
	network->addLayer(*layer3);

	return network;
}