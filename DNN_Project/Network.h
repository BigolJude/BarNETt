#pragma once
#include <list>
#include <string>
#include "Layer.h"

class Network
{
public:
	Network(list<Layer> layers);
	Network();
	void train(list<float> inputs, float learningRate, list<float>expected);
	void predict(list<float> inputs);
	list<float> getPrediction();
	void addLayer(Layer layer);
	void addLayer(int previousLayerCount, int neuronCount, string activation);
	void traverseLayer(int layerCount, int weightIndex, float error);
	float backpropogate(Neuron neuron, float weight, float error);
	void traverseNeuron(Layer layer, int weightIndex ,int layerCount, float error);
private:
	list<Layer> layers;
	int getMax(list<float> predictions);
	Layer getLayer(int index);
	list<float> errors;
	float learningRate;
};