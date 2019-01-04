#pragma once
#include <vector>
#include "Neuron.h"

class NeuronLayer
{
	double * activation;
	double * error;
	
public:

	int size;
	int inputPerNeuron;
	Neuron* neurons;

	void reset() const;

	double* getActivation() const;
	double* getError() const;

	NeuronLayer(int inputPerNeuron, Neuron* neurons,int size);
	NeuronLayer(const NeuronLayer& nl);
	NeuronLayer(NeuronLayer&& nl) noexcept;
	
	~NeuronLayer();
};

