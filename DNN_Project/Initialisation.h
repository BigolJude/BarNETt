#pragma once
static class Initialisation
{
public:
	static float* Xavier(float previousLayerNodeCount, float nextLayerNodeCount);
	static float He(float previousLayerNodeCount);
	static float Random(float upperBound, float lowerBound);
};

