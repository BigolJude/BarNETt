#include <iostream>
#include <list>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include "layer.h"
#include "loss.h"
#include "Initialisation.h"
#include "Activation.h"
#include "Network.h"
#include "CSV.h"
using namespace std;

int main()
{

	list<list<float>> values = CSV::read("C:\\iris.csv");



	//list<float> inputs1{ 8.4, 7.3, 192};
	//
	//Layer* layer1 = new Layer(3, 3, "relu");
	//Layer* layer2 = new Layer(3, 4, "relu");
	//Layer* layer3 = new Layer(4, 3, "relu");
	//Network* network = new Network();
	//
	//network->addLayer(*layer1);
	//network->addLayer(*layer2);
	//network->addLayer(*layer3);
	//
	//network->train(inputs1, 0.001, {0, 1, 0});
	//
	//float crossEntropyTests = Loss::crossEntropy({ 0.775,0.116,00.39,0.07 }, { 1,0,0,0 });
	//cout << "loss: " << crossEntropyTests << endl;
	//cout << "finished!" << endl;
	//
	//delete(layer1);
	//delete(layer2);
	//inputs1.clear();
}

//string getFile(string file)
//{
//}