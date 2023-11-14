#pragma once
#include <list>
#include <string>
#include "Layer.h"

class Network
{
public:
	Network(list<Layer> layers);
	Network();
	void train(list<double> inputs, float learningRate, list<double>expected);
	void predict(list<double> inputs);
	list<double> getPrediction();
	void addLayer(Layer layer);
	void addLayer(int previousLayerCount, int neuronCount, string activation);
	void traverseLayer(int layerCount, int weightIndex, double error);
	double backpropogate(Neuron* neuron, double weight, double error);
	void traverseNeuron(Layer layer, int neuronIndex ,int layerCount, double error);
	double getError();
private:
	list<Layer> layers;
	list<Layer> tempLayers;
	int getMax(list<double> predictions);
	Layer getLayer(int index);
	list<double> errors;
	float learningRate;
	double mError;
};