#include <iostream>
#include <list>
#include "layer.h"
#include "Initialisation.h"
#include "Activation.h"
#include "Network.h"


int main()
{
	list<float> inputs1{ 8.4, 7.3, 192, 0.55 };

	Layer* layer1 = new Layer(3, 3, "relu");
	Layer* layer2 = new Layer(3, 4, "relu");
	Layer* layer3 = new Layer(4, 3, "relu");
	Network* network = new Network();

	network->addLayer(*layer1);
	network->addLayer(*layer2);
	network->addLayer(*layer3);

	network->traverseLayers(0,0);

	cout << "finished!" << endl;

	delete(layer1);
	delete(layer2);
	inputs1.clear();
}