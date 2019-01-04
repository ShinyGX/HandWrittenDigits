#include "Neuron.h"
#include "NeuronUtil.h"


void Neuron::reset()
{
	activation = -1;
	error = -1;

	for (int i = 0;i <= numInputPerNerous;i++)
	{
		weight[i] = randomClamped();
	}
}




Neuron::Neuron(const int numInputPerNerous): numInputPerNerous(numInputPerNerous)
{
	weight = new double[numInputPerNerous + 1];
	for (int i = 0; i <= numInputPerNerous; i++)
	{
		weight[i] = randomClamped();
	}
}

Neuron::Neuron(Neuron&& n) noexcept
	:numInputPerNerous(n.numInputPerNerous),activation(n.activation),error(n.error),weight(n.weight)
{
	weight = nullptr;
}


Neuron::~Neuron()
{
	_delete(weight);
}
