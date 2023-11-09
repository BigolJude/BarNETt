#pragma once
static class Initialisation
{
public:
	static double* Xavier(double previousLayerNodeCount, double nextLayerNodeCount);
	static double He(float previousLayerNodeCount);
	static double Random(float upperBound, float lowerBound);
};

